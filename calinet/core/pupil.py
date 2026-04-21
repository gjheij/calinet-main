# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import re
import os
import math
import numpy as np
import pandas as pd
from calinet.utils import (
    update_kwargs,
    ensure_timestamp,
    get_session_and_task_name,
)

from calinet.core.io import (
    save_json,
    load_json,
    find_smi_txt_files,
    convert_all_edfs_to_asc,
    read_physio_tsv_headerless,
    write_physio_tsv_gz_headerless,
)

from calinet.templates.common import (
    EYE_JSON_TEMPLATE,
    EYE_PHYSIO_EVENTS_JSON_TEMPLATE
)

from calinet.config import available_labs, config
from calinet.core.utils import (
    pupil_unit_to_mm,
    gaze_pixel_to_mm,
    pupil_summary
)

import logging
logger = logging.getLogger(__name__)

from typing import Any, Union
from pathlib import Path
from copy import deepcopy


def set_events_to_nan(
        eye_input,
        physioevents_input,
        write_files: bool = False,
        eye_output: Union[str, None] = None,
        trial_types: tuple[str, ...] = ("blink", "saccade"),
        eye_value_columns: Union[list[str], None] = None,
        event_flag_columns: Union[list[str], None] = None,
    ) -> pd.DataFrame:
    """
    Set eye-signal samples to NaN where paired physioevents mark blink/saccade periods.

    Parameters
    ----------
    eye_input : str | Path | pd.DataFrame
        Eye physio file or already loaded dataframe.
    physioevents_input : str | Path | pd.DataFrame
        Paired physioevents file or already loaded dataframe.
    write_files : bool
        If True and `eye_input` is a path, write the modified dataframe back out.
    eye_output : str | None
        Optional output path. If None and write_files=True, overwrite/use eye_input path.
    trial_types : tuple[str, ...]
        Values in physioevents['trial_type'] that should trigger masking.
    eye_value_columns : list[str] | None
        Columns in the eye dataframe to mask. Default: all columns except timestamp.
    event_flag_columns : list[str] | None
        Additional physioevents columns whose truthy values should trigger masking.
        Example: ['blink'].

    Returns
    -------
    pd.DataFrame
        Eye dataframe with marked rows set to NaN.
    """

    # load inputs
    eye_path = None
    if isinstance(eye_input, (str, Path)):
        eye_path = Path(eye_input)
        eye_df = read_physio_tsv_headerless(eye_path)
    else:
        eye_df = eye_input.copy()

    if isinstance(physioevents_input, (str, Path)):
        phys_df = read_physio_tsv_headerless(physioevents_input)
    else:
        phys_df = physioevents_input.copy()

    if eye_df.empty or phys_df.empty:
        return eye_df

    if "timestamp" not in eye_df.columns:
        raise ValueError(f"Eye dataframe must contain a 'timestamp' column ({eye_df.columns})")

    if "onset" not in phys_df.columns or "duration" not in phys_df.columns:
        raise ValueError("Physioevents dataframe must contain 'onset' and 'duration' columns")

    # decide which eye columns to mask
    if eye_value_columns is None:
        eye_value_columns = [c for c in eye_df.columns if c != "timestamp"]

    # build event mask
    mask = pd.Series(False, index=phys_df.index)

    if "trial_type" in phys_df.columns:
        mask |= phys_df["trial_type"].astype(str).str.lower().isin(
            [x.lower() for x in trial_types]
        )

    if event_flag_columns is None:
        event_flag_columns = [c for c in ("blink", "saccade") if c in phys_df.columns]

    for col in event_flag_columns:
        vals = phys_df[col]

        if pd.api.types.is_bool_dtype(vals):
            mask |= vals.fillna(False)
        else:
            vals_num = pd.to_numeric(vals, errors="coerce")
            mask |= vals_num.fillna(0).astype(bool)

    events_to_mask = phys_df.loc[mask, ["onset", "duration"]].copy()
    if events_to_mask.empty:
        if write_files and eye_path is not None:
            out_path = Path(eye_output) if eye_output else eye_path
            write_physio_tsv_gz_headerless(eye_df, out_path)
        return eye_df

    # ensure numeric
    events_to_mask["onset"] = pd.to_numeric(events_to_mask["onset"], errors="coerce")
    events_to_mask["duration"] = pd.to_numeric(events_to_mask["duration"], errors="coerce")
    events_to_mask = events_to_mask.dropna(subset=["onset", "duration"])

    out = eye_df.copy()
    t = pd.to_numeric(out["timestamp"], errors="coerce").to_numpy()

    for _, row in events_to_mask.iterrows():
        start = row["onset"]
        stop = start + row["duration"]
        sample_mask = (t >= start) & (t < stop)
        out.loc[sample_mask, eye_value_columns] = pd.NA

    if write_files and eye_path is not None:
        out_path = Path(eye_output) if eye_output else eye_path
        write_physio_tsv_gz_headerless(out, out_path)

    return out


def mask_eye_recordings_with_physioevents(
        output_dir: Union[str, Path],
        overwrite: bool = True,
        **kwargs
    ) -> list[tuple[str, str]]:
    """
    Apply event-based masking to eye-tracking recordings using paired physioevents files.

    This function searches a directory for eye-tracking recordings (``*_physio.tsv.gz``)
    and their corresponding physioevents files (``*_physioevents.tsv.gz``). For each
    matched pair, it applies masking to the eye data by setting samples corresponding
    to specified event periods (e.g., blinks or saccades) to NaN.

    Masking is performed via :func:`set_events_to_nan`, which uses event onset and
    duration information from the physioevents file to blank out affected samples
    in the eye recording.

    Parameters
    ----------
    output_dir : str or pathlib.Path
        Directory containing eye-tracking physio files and their corresponding
        physioevents files. Files are expected to follow the naming convention:
        ``*_physio.tsv.gz`` and ``*_physioevents.tsv.gz``.

    overwrite : bool, default=True
        If True, the original eye recording files are overwritten with masked data.
        If False, new files are written with the suffix ``"_physio_masked.tsv.gz"``
        appended to the original filename.

    **kwargs
        Additional keyword arguments passed directly to
        :func:`set_events_to_nan`. This can be used to control which event types
        are masked (e.g., ``trial_types=("blink", "saccade")``), which columns are
        affected, or other masking behavior.

    Returns
    -------
    processed : list of tuple of str
        List of tuples containing the processed file pairs:
        ``(eye_file_path, physioevents_file_path)``.

    Notes
    -----
    - Only files with matching ``*_physio.tsv.gz`` and
      ``*_physioevents.tsv.gz`` names are processed.
    - If a physioevents file is missing for a given eye recording, that file is
      skipped and a warning is logged.
    - Masking is performed by setting selected data columns (e.g., pupil size,
      gaze coordinates) to NaN during event intervals.
    - The exact masking behavior (e.g., blink-only vs. blink + saccade) depends
      on the arguments passed via ``**kwargs``.
    - This function operates at the file level and modifies or writes TSV.GZ
      files using the project's I/O utilities.

    Examples
    --------
    Mask both blinks and saccades in all eye recordings in a directory:

    >>> mask_eye_recordings_with_physioevents(
    ...     "derivatives/eyetracking",
    ...     trial_types=("blink", "saccade"),
    ...     event_flag_columns=["blink"]
    ... )

    Mask only blink periods and write to new files:

    >>> mask_eye_recordings_with_physioevents(
    ...     "derivatives/eyetracking",
    ...     overwrite=False,
    ...     trial_types=("blink",),
    ...     event_flag_columns=["blink"]
    ... )
    """

    output_dir = Path(output_dir)
    processed = []

    eye_files = sorted(output_dir.glob("*recording-eye*_physio.tsv.gz"))

    for eye_file in eye_files:
        physioevents_file = Path(
            str(eye_file).replace("_physio.tsv.gz", "_physioevents.tsv.gz")
        )

        if not physioevents_file.exists():
            logger.warning(f"No paired physioevents file found for '{eye_file.name}'")
            continue

        logger.info(
            f"Masking eye file '{eye_file.name}' using '{physioevents_file.name}'"
        )

        out_path = eye_file if overwrite else eye_file.with_name(
            eye_file.name.replace("_physio.tsv.gz", "_physio_masked.tsv.gz")
        )

        set_events_to_nan(
            eye_file,
            physioevents_file,
            eye_output=str(out_path),
            **kwargs
        )

        # check if range is reasonable
        logger.info(f"Retrieving summary stats after masking for '{eye_file}'")
        _ = pupil_summary(eye_file)

        processed.append((str(eye_file), str(physioevents_file)))

    return processed


def fetch_and_write_eye_data(
        eye_file,
        output_base: str=None,
        meta_dict: dict=None,
        stim_pres: dict=None,
        write_files: bool=True,
        **kwargs
    ) -> dict:
    """
    Parse, validate, convert, and optionally write eye-tracking data.

    This function reads eye-tracking data from an EyeLink (.asc) or SMI (.txt)
    file, processes the data for each eye, performs unit conversions (pupil size
    and gaze position), validates signal quality, and optionally writes the
    processed data to compressed TSV files along with corresponding metadata.

    Parameters
    ----------
    eye_file : str
        Path to the eye-tracking data file. Supported formats are:
        - ``.asc`` (EyeLink)
        - ``.txt`` (SMI)
    output_base : str, optional
        Base path for output files. Output files will be named as
        ``f"{output_base}{eye}_physio.tsv.gz"`` for each eye.
    meta_dict : dict
        Dictionary containing metadata for each eye (e.g., "eye1", "eye2").
        This dictionary is updated with conversion settings during processing.
    stim_pres : dict
        Dictionary containing stimulus presentation parameters. Expected keys:
        - ``"StimulusPresentation"["ScreenSize"]`` (in mm)
        - ``"StimulusPresentation"["ScreenResolution"]`` (in pixels)
        - ``"StimulusPresentation"["ScreenDistance"]`` (in mm)
    write_files : bool, default=True
        If True, writes processed data to disk as compressed TSV files and
        associated JSON metadata files.
    **kwargs
        Additional keyword arguments passed to :func:`gaze_pixel_to_mm`.

    Returns
    -------
    eye_conv : dict
        Dictionary mapping eye identifiers (e.g., "eye1", "eye2") to processed
        pandas DataFrames. Each DataFrame contains:
        - Pupil size in millimeters (if applicable)
        - Gaze coordinates in millimeters
        - A timestamp column

    Raises
    ------
    TypeError
        If `meta_dict` or `stim_pres` is not a dictionary.
    NotImplementedError
        If the file extension is not supported.
    Exception
        If parsing the eye-tracking file fails.

    Notes
    -----
    Processing steps for each eye include:

    1. Validation:
       - Skips empty DataFrames
       - Skips signals with all NaNs
       - Skips flat signals (no variance)

    2. Pupil conversion:
       - Converts pupil measurements to millimeters using
         :func:`pupil_unit_to_mm` if measurement type is "AREA" or "DIAMETER"

    3. Quality check:
       - Logs a warning if mean pupil size exceeds a configurable threshold

    4. Gaze conversion:
       - Converts gaze coordinates from pixels to millimeters using
         :func:`gaze_pixel_to_mm`

    5. Timestamp handling:
       - Ensures a timestamp column using :func:`ensure_timestamp`

    6. Output:
       - Optionally writes processed data to ``.tsv.gz`` files
       - Saves updated metadata as JSON

    The function does not modify the input data structures in place except for
    updating the provided `meta_dict`.

    Examples
    --------
    >>> result = fetch_and_write_eye_data(
    ...     "subject01.asc",
    ...     output_base="sub-01_",
    ...     meta_dict=meta,
    ...     stim_pres=stimulus_info
    ... )

    >>> result = fetch_and_write_eye_data(
    ...     "recording.txt",
    ...     meta_dict=meta,
    ...     stim_pres=stimulus_info,
    ...     write_files=False
    ... )
    """

    # check inputs
    if not isinstance(meta_dict, dict):
        raise TypeError(f"'meta_dict' must be a dict, not {type(meta_dict)}")
    
    if not isinstance(stim_pres, dict):
        raise TypeError(f"'stim_pres' must be a dict, not {type(stim_pres)}")
    

    # get function depending on file extension
    if eye_file.endswith(".asc"):
        from calinet.imports.eyelink import asc_to_df
        parse_func = asc_to_df
        mode = "EyeLink"
    elif eye_file.endswith(".txt"):
        from calinet.imports.smi import smi_txt_to_df
        parse_func = smi_txt_to_df
        mode = "SMI"
    else:
        raise NotImplementedError(f"Only eye-tracking files with extension 'asc' or 'txt' can be parsed")
    
    # run
    logger.info(f"Reading '{os.path.basename(eye_file)}' with {parse_func} [mode={mode}]")
    try:
        eye_dict, measurement_type = parse_func(eye_file)
    except Exception as e:
        raise Exception(f"Dataframe creation failed: {e}") from e
    
    logger.info("Done reading data")

    eye_conv = {}
    for key, val in eye_dict.items():
        logger.info(f"Processing '{key}'")

        # Case 1: truly empty DataFrame
        if len(val) == 0:
            logger.warning(f"Dataframe for '{key}' is empty, assuming data was not recorded")
            continue

        # Compute variance once
        var = val.var(skipna=True)

        # Case 2: all NaNs
        if var.isna().all():
            logger.warning(f"'{key}' contains only NaNs, skipping")
            continue

        # Case 3: flat signal (no variation)
        if (var.fillna(0) == 0).all():
            logger.warning(f"'{key}' has no variation (flat signal), skipping")
            continue

        # Otherwise keep it
        logger.info(f"'{key}' contains valid data")
        eye_conv[key] = val
        
        # get metadata for eye
        curr_meta = meta_dict[key]

        screen_size_mm = stim_pres.get("StimulusPresentation").get("ScreenSize")
        screen_res_px = stim_pres.get("StimulusPresentation").get("ScreenResolution")
        screen_dist_mm = stim_pres.get("StimulusPresentation").get("ScreenDistance")
        
        # convert pupil_size to mm
        if measurement_type.upper() in ["AREA", "DIAMETER"]:

            val, settings = pupil_unit_to_mm(
                val,
                camera_eye_distance=screen_dist_mm,
                measurement_type=measurement_type,
                overwrite=True
            )

            # update pupil settings
            curr_meta.update(settings)
        else:
            logger.info(f"Measurement type={measurement_type}; will not convert to mm")

        # check if range is reasonable
        _ = pupil_summary(val)

        # convert gaze to mm
        logger.info(f"Converting gaze data to mm; screen={screen_size_mm}mm | resolution={screen_res_px}px")
        eye_mm = gaze_pixel_to_mm(
            val,
            screen_size_mm=screen_size_mm,
            screen_resolution_px=screen_res_px,
            overwrite=True,
            **kwargs
        )

        # add timestamp
        eye_mm_ts, _ = ensure_timestamp(
            eye_mm,
            fs=curr_meta.get("SamplingFrequency"),
            force=True
        )

        eye_conv[key] = eye_mm_ts

        # write file
        if write_files:
            output_path = f"{output_base}{key}_physio.tsv.gz"
            logger.info(f"Writing {output_path}")
            write_physio_tsv_gz_headerless(
                eye_mm_ts.copy(),
                output_path
            )

            # save json
            json_path = output_path.replace(".tsv.gz", ".json")
            save_json(json_path, curr_meta)

    return eye_conv


def fetch_eye_metadata(
        eye_file: str,
        lab_name: str,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """
    Extract metadata for both eyes and stimulus presentation settings.

    This function reads metadata from an eye-tracking file (EyeLink `.asc` or
    SMI `.txt`) and returns structured metadata for each eye along with
    stimulus presentation information. It internally calls
    `get_eyetracker_setup_info` for both left and right eyes.

    Parameters
    ----------
    eye_file : str
        Path to the eye-tracking file. Supported formats are:
        - ``.asc`` (EyeLink)
        - ``.txt`` (SMI)
    lab_name : str
        Name of the lab or setup, used to populate metadata templates.

    Returns
    -------
    eye_info : dict of dict
        Dictionary containing metadata for each eye:
        - ``"eye1"``: metadata for the left eye
        - ``"eye2"``: metadata for the right eye

        Each entry is itself a dictionary populated using
        `get_eyetracker_setup_info` and the `EYE_JSON_TEMPLATE`.
    stim_pres : dict
        Dictionary containing stimulus presentation settings extracted from
        the eye-tracking file (e.g., screen size, resolution, distance).

    Raises
    ------
    TypeError
        If the file extension is not supported or if the returned stimulus
        presentation data is not a dictionary.

    Notes
    -----
    - The function assumes a two-eye setup:
      - ``"eye1"`` corresponds to the left eye ("LEFT")
      - ``"eye2"`` corresponds to the right eye ("RIGHT")
    - Stimulus presentation metadata is extracted once (from the first eye)
      and reused for both.
    - The returned `eye_info` structure is compatible with downstream
      processing functions such as `fetch_and_write_eye_data`.

    Examples
    --------
    >>> eye_info, stim = fetch_eye_metadata(
    ...     "subject01.asc",
    ...     lab_name="NeuroLab"
    ... )

    >>> eye_info["eye1"]["SamplingFrequency"]
    1000
    """

    eye_path = Path(eye_file)
    suffix = eye_path.suffix.lower()

    if suffix == ".asc":
        from calinet.imports.eyelink import get_eyetracker_setup_info
    elif suffix == ".txt":
        from calinet.imports.smi import get_eyetracker_setup_info
    else:
        raise TypeError(
            f"Invalid extension for eye-tracking file: {suffix!r}. "
            "Expected '.asc' for EyeLink or '.txt' for SMI."
        )

    logger.info("Fetching metadata")

    eye_info: dict[str, dict[str, Any]] = {}
    stim_pres: dict[str, Any] | None = None

    for eye_key, eye_label in (("eye1", "LEFT"), ("eye2", "RIGHT")):
        info, stim = get_eyetracker_setup_info(
            eye_file,
            eye_label,
            EYE_JSON_TEMPLATE,
            lab_name=lab_name,
        )
        eye_info[eye_key] = info

        if stim_pres is None:
            stim_pres = stim

    if not isinstance(stim_pres, dict):
        raise TypeError(
            f"'stim_pres' is {type(stim_pres).__name__}, expected dict. "
            "Something went wrong in 'get_eyetracker_setup_info()'."
        )

    return eye_info, stim_pres


def create_physioevents_files(
        eye_file: str,
        output_base: str=None,
        eye_mm: dict=None,
        write_files: bool=True,
        onsets: Union[str, pd.DataFrame]=None,
        **kwargs
    ) -> dict:
    """
    Generate physiological event files (e.g., blinks, saccades) from eye-tracking data.

    This function extracts physiological events for each eye from an eye-tracking
    recording, aligns them with external onset markers (if provided), and optionally
    writes the results to compressed TSV files along with JSON metadata.

    Parameters
    ----------
    eye_file : str
        Path to the eye-tracking file. Supported formats are:
        - ``.asc`` (EyeLink)
        - ``.txt`` (SMI)
    output_base : str, optional
        Base path for output files. Output files will be named as
        ``f"{output_base}{eye}_physioevents.tsv.gz"``.
    eye_mm : dict, optional
        Dictionary mapping eye identifiers (e.g., "eye1", "eye2") to DataFrames
        containing preprocessed eye-tracking data (in millimeters).
    write_files : bool, default=True
        If True, writes physiological event files and corresponding JSON metadata.
    onsets : str, pandas.DataFrame, optional
        Path or dataFrame containing onset times (e.g., from experimental markers).
        Used to align eye-tracking time with physiological recordings.    
    **kwargs
        Additional keyword arguments passed to `fetch_physioevents`.

    Returns
    -------
    event_dict : dict
        Dictionary mapping eye identifiers to DataFrames of physiological events
        (e.g., blinks, saccades).

    Raises
    ------
    TypeError
        If the file extension is not supported.

    Notes
    -----
    Processing steps for each eye include:

    1. Event extraction:
       - Uses `fetch_physioevents` from the appropriate backend:
         - EyeLink: operates directly on the raw file
         - SMI: requires the preprocessed DataFrame and additional settings

    2. Input preparation:
       - Merges default inputs with lab-specific keyword arguments via
         `update_kwargs`

    3. Time alignment:
       - Computes the offset between physiological time and eye-tracking time:
         ``diff = t_mark - t_phys``
       - This offset is stored as the ``StartTime`` field in the output JSON
       - Ensures alignment between eye-tracking and physiological modalities

    4. Output:
       - Writes ``.tsv.gz`` files containing physiological events
       - Writes accompanying JSON files based on
         `EYE_PHYSIO_EVENTS_JSON_TEMPLATE`
       - Updates corresponding ``*_physio.json`` files with aligned ``StartTime``

    Important:
        The alignment step is critical for ensuring that eye-tracking data and
        physiological recordings (e.g., SCR) share a common temporal reference.

    Examples
    --------
    >>> events = create_physioevents_files(
    ...     eye_file="subject01.asc",
    ...     output_base="sub-01_",
    ...     eye_mm=eye_data,
    ...     onsets=markers
    ... )

    >>> events = create_physioevents_files(
    ...     eye_file="recording.txt",
    ...     eye_mm=eye_data,
    ...     write_files=False
    ... )
    """

    event_dict = {}
    for key, val in eye_mm.items():

        # set function depending on input
        if eye_file.endswith(".asc"):
            from calinet.imports.eyelink import fetch_physioevents
        elif eye_file.endswith(".txt"):
            from calinet.imports.smi import fetch_physioevents
        else:
            raise TypeError(f"Invalid extension for eye-tracking file. Must be 'asc' for EyeLink or '.txt' from SMI")
            
        # define input dict
        inputs = {
            "df": val,
            "raw_file": eye_file
        }

        # fetch saccade/blink settings from config
        saccade_settings = config.get("blink_detection_settings")
        logger.debug(f"Settings for saccades/blink: {saccade_settings}")
        
        # combine inputs
        inputs = {**inputs, **saccade_settings}

        # update lab-specific inputs
        for k, v in inputs.items():
            kwargs = update_kwargs(
                kwargs,
                k,
                v
            )

        # create blink/saccade events | pass mm-converted dataframe
        logger.info(f"Fetching physioevents for '{key}'")
        phys_events, phys_settings = fetch_physioevents(**kwargs)
        event_dict[key] = phys_events.copy()
        
        # ----------------------------------------------------------------------
        # IMPORTANT STEP!!

        # find offset between physio/markers and eye-tracking based on physioevents.
        #   - First we extract when the first event occurred in physiology-time
        #   - Then we extract when this event occurred in eye-tracking time via
        #     MSG key (differs for smi/edf, but both have 'CS' in the MSG)
        #   - The offset between eye-tracking and physiology = t_phys - t_eye
        #   - This is effectively the 'StartTime' for eye-tracking
        #   - When we then later apply trimming based on the markers, we are
        #     pulling all data into the same temporal framework, so events 
        #     in *events.tsv file actually make sense for eye-tracking data 
        #     too
        if onsets is None:
            logger.warning(f"No onsets specified, cannot align eye-tracking and physiological data..")
            diff = 0
        else:
            # read dataframe
            if isinstance(onsets, str):
                logger.info(f"Reading events from file: '{onsets}'")
                onsets = pd.read_csv(onsets, delimiter="\t")

            t_phys = first_event_in_physioevents(phys_events)
            t_mark = float(onsets.iloc[0, 0])
            diff = t_mark-t_phys

            logger.info(f"First event at t={round(t_mark, 3)} == t={round(t_phys, 3)} in eye-tracking data; diff={round(diff, 3)}. Setting StartTime-key.")

        # write file
        if write_files:
            output_path = f"{output_base}{key}_physioevents.tsv.gz"
            logger.info(f"Writing {output_path}")

            write_physio_tsv_gz_headerless(
                phys_events,
                output_path
            )

            # copy template
            phys_tpl = deepcopy(EYE_PHYSIO_EVENTS_JSON_TEMPLATE)

            # this one is VERY important for eye-tracking with physiological data during the trimming stage!
            phys_tpl["StartTime"] = diff

            # add algorithm settings for SMI-files
            if isinstance(phys_settings, dict):
                logger.info(f"Updating EYE_PHYSIO_EVENTS_JSON_TEMPLATE with blink/saccade settings")
                phys_tpl.update(phys_settings)

            # Create a JSON file for physio events
            output_json = output_path.replace(".tsv.gz", ".json")
            logger.info(f"Writing {output_json}")
            save_json(
                output_json,
                phys_tpl
            )

            # ------------------------------------------------------------------
            # IMPORTANT FOR ALIGNING EYE-TRACKING AND SCR
            
            # also update corresponding eye?_physio.json
            if diff > 0:
                phys_json = output_json.replace("physioevents", "physio")
                assert os.path.exists(phys_json), f"Corresponding physio file '{phys_json}' does not exist."

                logger.info(f"Updating StartTime in {phys_json} to {round(diff, 3)}s")
                phys_data = load_json(phys_json)
                phys_data["StartTime"] = diff
                save_json(phys_json, phys_data)

    return event_dict        


def process_eyetracker_file(
        eye_file: str=None,
        output_base: str=None,
        lab_name: str=None,
        onsets: Union[str, list]=None,
        write_files: bool=True,
        **kwargs
    ) -> None:
    """
    End-to-end processing of an eye-tracking file into analysis-ready outputs.

    This function orchestrates the full processing pipeline for an eye-tracking
    recording. It extracts metadata, converts raw data into calibrated units,
    generates physiological event files (e.g., blinks, saccades), and optionally
    writes all outputs to disk.

    Parameters
    ----------
    eye_file : str, optional
        Path to the eye-tracking file. Supported formats are:
        - ``.asc`` (EyeLink)
        - ``.txt`` (SMI)
    output_base : str, optional
        Base path for all generated output files (TSV and JSON).
    lab_name : str, optional
        Name of the lab or setup, used for metadata generation.
    onsets : str, list, optional
        String point to events file or list/array of onset times used to align 
        eye-tracking data with external physiological recordings.
    write_files : bool, default=True
        If True, writes all generated outputs (TSV and JSON files) to disk.
    **kwargs
        Additional keyword arguments passed to downstream processing functions,
        such as `fetch_and_write_eye_data`.

    Notes
    -----
    The processing pipeline consists of three main steps:

    1. Metadata extraction:
       - Calls :func:`fetch_eye_metadata` to obtain per-eye metadata and
         stimulus presentation settings.

    2. Data conversion:
       - Calls :func:`fetch_and_write_eye_data` to:
         - Parse raw eye-tracking data
         - Convert pupil size to millimeters
         - Convert gaze coordinates to millimeters
         - Add timestamps
         - Optionally write ``.tsv.gz`` and JSON files

    3. Event generation:
       - Calls :func:`create_physioevents_files` to extract physiological events
         (e.g., blinks, saccades) and align them with external onsets.

    If no valid eye-tracking data are detected, the event generation step is
    skipped.

    Examples
    --------
    >>> process_eyetracker_file(
    ...     eye_file="subject01.asc",
    ...     output_base="sub-01_",
    ...     lab_name="NeuroLab"
    ... )

    >>> process_eyetracker_file(
    ...     eye_file="recording.txt",
    ...     output_base="sub-02_",
    ...     lab_name="NeuroLab",
    ...     onsets=[0.5, 1.2, 2.0],
    ...     write_files=False
    ... )
    """

    # Create JSON files for eye1 and eye2
    meta_dict, stim_pres = fetch_eye_metadata(
        eye_file,
        lab_name=lab_name
    )
    
    # Convert ASC data to TSV files for eye1 and eye2 recordings
    logger.info(f"Generating tsv.gz-files")
    eye_dict_mm = fetch_and_write_eye_data(
        eye_file,
        output_base=output_base,
        meta_dict=meta_dict,
        stim_pres=stim_pres,
        lab_name=lab_name,
        write_files=write_files,
        **kwargs
    )

    # Create physioevents files for eye1 and eye2
    if len(eye_dict_mm)>0:
        logger.info(f"Creating physioevents for '{eye_file}'")
        _ = create_physioevents_files(
            eye_file,
            output_base,
            eye_mm=eye_dict_mm,
            onsets=onsets,
            write_files=write_files
        )
    else:
        logger.warning(f"No valid eye-tracking data detected (see messages above), skipping physioevents.")

    # add stimulus presentation field to json
    if isinstance(onsets, str):
        onsets_json = onsets.replace(".tsv", ".json")
        if not os.path.exists(onsets_json):
            logger.warning(f"Could not find '{onsets_json}'. Cannot update 'StimulusPresentation'")

        logger.info(f"Updating 'StimulusPresentation' in '{onsets_json}'")
        ev_meta = load_json(onsets_json)
        ev_meta.update(stim_pres)
        save_json(onsets_json, ev_meta)

    # after physio + physioevents have been written | needs json file for columns
    if write_files and output_base is not None:
        out_dir = os.path.dirname(output_base)

        logger.info(
            "Applying physioevents-based masking to eye recordings | "
            f"directory='{out_dir}' | overwrite=True | write_files={write_files}"
        )

        processed = mask_eye_recordings_with_physioevents(
            out_dir,
            overwrite=True,
            write_files=write_files
        )

        logger.info(
            f"Completed masking of {len(processed)} eye recording file(s) in '{out_dir}'"
        )


def find_eye_files(raw_path):
    eye_files = []
    for root, _, files in os.walk(raw_path):
        for filename in files:
            if filename.lower().endswith(".asc"):
                eye_file = os.path.join(root, filename)
                eye_files.append(eye_file)

    return eye_files


def first_event_in_physioevents(
        df: pd.DataFrame,
        search_for: str="CS",
        secondary:str="CS_Start_time"
    ) -> float:
    """
    Extract the timestamp of the first matching event from a physioevents DataFrame.

    This function searches the last column of the input DataFrame for the first
    occurrence of a string containing a specified substring (e.g., "CS"). It then
    extracts the corresponding event time, either directly from the first column
    or by parsing a timestamp embedded within the string.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing physiological event data. The last column is expected
        to contain event labels or metadata strings, and the first column typically
        contains timestamps.
    search_for : str, default="CS"
        Substring used to identify the target event within the last column.
    secondary : str, default="CS_Start_time"
        Key used to extract a timestamp from the event string when the string
        contains embedded timing information.

    Returns
    -------
    float
        Timestamp of the first matching event.

    Raises
    ------
    IndexError
        If no matching event is found in the DataFrame.
    AttributeError
        If the expected timestamp pattern cannot be extracted from the string.

    Notes
    -----
    The function operates as follows:

    1. Scans the last column for entries containing `search_for`.
    2. Identifies the first matching row.
    3. If the matched string contains "Time", extracts the timestamp using a
       regular expression based on `secondary`.
    4. Otherwise, returns the value from the first column of that row.

    The expected string format for embedded timestamps is:

        "'<secondary>': <float_value>"

    Examples
    --------
    >>> first_event_in_physioevents(df)
    0.532

    >>> first_event_in_physioevents(df, search_for="US")
    1.245
    """
    
    arr = df.iloc[:, -1].to_numpy()
    idx = np.where([isinstance(x, str) and search_for in x for x in arr])[0][0]
    first_value = arr[idx]

    if "Time" in first_value:
        return float(re.search(rf"'{secondary}': ([\d\.]+)", first_value).group(1))
    else:
        return float(df.iloc[idx, 0])
    

def handle_eyetracking(
        raw_path: str=None,
        conv_path: str=None,
        subject_name: str=None,
        overwrite: bool=False,
        lab_name: str=None,
        onsets_dict: dict=None,
        **kwargs
    ):
    """
    Locate, process, and convert eye-tracking files for a subject.

    This function identifies eye-tracking files based on lab configuration,
    processes each file through the full pipeline, and generates standardized
    output files (TSV and JSON) for eye-tracking data and physiological events.

    Parameters
    ----------
    raw_path : str, optional
        Path to the directory containing raw input files (e.g., EDF, ASC, or TXT).
    conv_path : str, optional
        Path to the directory where converted output files will be stored.
    subject_name : str, optional
        Identifier for the subject. Used in output file naming.
    overwrite : bool, default=False
        If True, overwrites existing converted files when applicable (e.g.,
        during EDF → ASC conversion).
    lab_name : str, optional
        Name of the lab. Used to determine the eye-tracking format via
        `available_labs`.
    onsets_dict : dict, optional
        Dictionary mapping task names to onset information. Each value is passed
        to downstream processing for aligning eye-tracking and physiological data.
    **kwargs
        Additional keyword arguments passed to :func:`process_eyetracker_file`.

    Returns
    -------
    None
        This function does not return any value. It processes files and writes
        outputs to disk.

    Notes
    -----
    The function performs the following steps:

    1. Determine eye-tracking format:
       - Uses `available_labs` configuration to check whether the lab provides:
         - "asc" → EyeLink data (EDF files converted to ASC)
         - "txt" → SMI data

    2. Locate files:
       - EyeLink: converts all EDF files to ASC using
         `convert_all_edfs_to_asc`
       - SMI: finds all TXT files using `find_smi_txt_files`

    3. Iterate over files:
       - Extracts session and task name using `get_session_and_task_name`
       - Constructs output paths in ``<conv_path>/physio/``

    4. Process each file:
       - Calls :func:`process_eyetracker_file` to:
         - Extract metadata
         - Convert data to millimeters
         - Generate physioevents
         - Write outputs

    If no eye-tracking data are available for the specified lab or subject,
    the function exits early with a warning.

    Examples
    --------
    >>> handle_eyetracking(
    ...     raw_path="raw/sub-01/",
    ...     conv_path="derivatives/",
    ...     subject_name="sub-01",
    ...     lab_name="NeuroLab"
    ... )

    >>> handle_eyetracking(
    ...     raw_path="raw/sub-02/",
    ...     conv_path="derivatives/",
    ...     subject_name="sub-02",
    ...     lab_name="SMILab",
    ...     onsets_dict={"acquisition": onsets_df}
    ... )
    """
    
    # find eyetracking files
    lab_clean = lab_name.replace(" ", "").lower()
    
    # read from config.available_labs what type of eyetracking is available
    try:
        eye_suffix = available_labs.get(lab_clean).get("has_eyetrack")
    except:
        eye_suffix = None

    # find suffix-specific files; return if None -> no eyetracking
    eye_files = []
    if eye_suffix == "asc":
        logger.info(f"Lab={lab_clean}; assuming EyeLink asc-files")
        eye_files = convert_all_edfs_to_asc(
            raw_path,
            overwrite=overwrite
        )
    elif eye_suffix == "txt":
        # SMI txt files
        logger.info(f"Lab={lab_clean}; assuming SMI txt-files")
        eye_files = find_smi_txt_files(raw_path)
    else:
        logger.warning(f"'{subject_name}' from lab='{lab_name}' does not have eye-tracking files")
        return

    if len(eye_files)>0:
        logger.info(f"Converting {len(eye_files)} eyetracking files to tsv.gz-files: {eye_files}")

        for eye_file in eye_files:

            # get task name (acquisition|extinction)
            (_, task_name) = get_session_and_task_name(eye_file)

            # output directory
            output_path = os.path.join(
                conv_path, 
                "physio"
            ) 
            os.makedirs(output_path, exist_ok=True)
            
            # define single output
            output_base = os.path.join(
                output_path,
                f"{subject_name}_task-{task_name}_recording-"
            )

            # process
            logger.info(f"Start processing file '{eye_file}'")
            try:
                process_eyetracker_file(
                    eye_file=eye_file,
                    output_base=output_base,
                    lab_name=lab_name,
                    onsets=onsets_dict.get(task_name),
                    **kwargs
                )
            except Exception as e:
                raise Exception(f"Error processing file '{eye_file}': {e}") from e

            logger.info("Created eyetracker files")
    else:
        logger.warning(f"No eyetracking-files found for {subject_name}")

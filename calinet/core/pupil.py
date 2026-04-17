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

import logging
logger = logging.getLogger(__name__)

from typing import Any, Tuple, Union
from pathlib import Path
from copy import deepcopy


def diameter_to_mm(
        inputs: float,
        camera_eye_distance: float=None,
        measurement_type="AREA"
    ) -> tuple:
    """
    Convert pupil measurements to millimeters.

    This function converts EyeLink pupil measurements (either AREA or DIAMETER)
    into estimated pupil diameter in millimeters using a PsPM-style calibration
    approach. The conversion applies a scaling factor based on the camera-eye
    distance relative to a reference distance.

    Parameters
    ----------
    inputs : float or array-like
        Input pupil measurements. If `measurement_type="AREA"`, values are assumed
        to represent pupil area and will be square-root transformed. If
        `measurement_type="DIAMETER"`, values are treated as diameter directly.
    camera_eye_distance : float, optional
        Distance between the camera and the eye in millimeters. This value is used
        to scale the pupil measurement relative to a reference distance defined
        in the configuration.
    measurement_type : {"AREA", "DIAMETER"}, default="AREA"
        Type of input measurement:
        - "AREA": input values represent pupil area (will be square-rooted)
        - "DIAMETER": input values represent pupil diameter directly

    Returns
    -------
    pupil_mm : float or ndarray
        Estimated pupil diameter in millimeters.
    conversion_info : dict
        Dictionary containing metadata about the conversion, including:
        - "Name": Name of the conversion function
        - "Formula": String representation of the applied formula
        - "Parameters": Dictionary of parameters used (multiplier, reference distance, screen distance)
        - "Description": Brief description of the conversion method

    Raises
    ------
    ValueError
        If `measurement_type` is not one of {"AREA", "DIAMETER"}.

    Notes
    -----
    The conversion formula is:

        diameter_mm = multiplier * (camera_eye_distance / reference_distance) * f(inputs)

    where:
        - f(inputs) = sqrt(inputs) if measurement_type == "AREA"
        - f(inputs) = inputs if measurement_type == "DIAMETER"

    The `multiplier` and `reference_distance` are retrieved from the global
    configuration (`config["pupil_multiplication"]`).

    Examples
    --------
    >>> diameter_to_mm(4000, camera_eye_distance=600, measurement_type="AREA")
    (value_in_mm, {...})

    >>> diameter_to_mm(np.array([3.2, 3.5]), camera_eye_distance=600, measurement_type="DIAMETER")
    (array([...]), {...})
    """

    if measurement_type == "AREA":
        use_vals = np.sqrt(inputs)
        ffunc = "sqrt(input)"
    elif measurement_type == "DIAMETER":
        use_vals = inputs.copy()
        ffunc = "input"
    else:
        raise ValueError(f"measurement_type must be one of 'AREA' or 'DIAMETER', not '{measurement_type}'")

    pupil_config = config.get("pupil_multiplication")
    reference_distance = pupil_config.get("reference_distance")
    multiplier = pupil_config.get(measurement_type)

    logger.info(f"Converting {measurement_type.upper()} to mm; screen distance = {camera_eye_distance}mm | multiplier={multiplier}")

    pupil_mm = (
        multiplier * (camera_eye_distance / reference_distance) * use_vals
    )

    conversion_info = {
        "Name": "diameter_to_mm",
        "Formula": f"diameter_mm = multiplier * (camera_eye_distance / reference_distance) * {ffunc}",
        "Parameters": {
            "Multiplier": multiplier,
            "ReferenceDistance": reference_distance,
            "ScreenDistance": camera_eye_distance
        },
        "Description": "Converts EyeLink pupil AREA units to estimated pupil diameter in mm using PsPM-style calibration."
    }

    return pupil_mm, conversion_info


def pupil_unit_to_mm(
        df: pd.DataFrame,
        overwrite=True,
        column_name="pupil_size",
        **kwargs
    ) -> Tuple[pd.DataFrame, dict]:
    """
    Convert a DataFrame column of pupil measurements to millimeters.

    This function applies :func:`diameter_to_mm` to a specified column in a
    pandas DataFrame and appends the converted values as a new column. Optionally,
    the original column can be overwritten with the converted values.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing pupil measurements.
    overwrite : bool, default=True
        If True, replaces the original column (`column_name`) with the converted
        values in millimeters and removes the temporary column. If False, retains
        both the original and converted columns.
    column_name : str, default="pupil_size"
        Name of the column in `df` containing pupil measurements to convert.
    **kwargs
        Additional keyword arguments passed to :func:`diameter_to_mm`, such as
        `camera_eye_distance` and `measurement_type`.

    Returns
    -------
    out : pandas.DataFrame
        A copy of the input DataFrame with the converted pupil size values.
        If `overwrite=True`, the original column is replaced. Otherwise, a new
        column named ``f"{column_name}_mm"`` is added.
    settings : dict
        Dictionary containing metadata about the conversion, as returned by
        :func:`diameter_to_mm`.

    Notes
    -----
    This function does not modify the input DataFrame in place; a copy is always
    created.

    The conversion is performed element-wise on the specified column using
    :func:`diameter_to_mm`.

    Examples
    --------
    >>> df_mm, info = pupil_unit_to_mm(df, camera_eye_distance=600)
    
    >>> df_mm, info = pupil_unit_to_mm(
    ...     df,
    ...     column_name="pupil_area",
    ...     measurement_type="AREA",
    ...     overwrite=False
    ... )
    """

    out = df.copy()
    out[f"{column_name}_mm"], settings = diameter_to_mm(
        out[column_name],
        **kwargs
    )

    # overwrite column
    if overwrite:
        out[column_name] = out[f"{column_name}_mm"]
        out.drop(columns=[f"{column_name}_mm"], inplace=True)

    return out, settings


def correct_to_fixation_hist_peak(
        df: pd.DataFrame,
        screen_mm: Union[tuple, list],
        x_col: str="x_coordinate",
        y_col: str="y_coordinate",
        bin_size_mm: float=2.0,
        return_shift: bool=True,
    ) -> Tuple[pd.DataFrame, tuple]:
    """
    Recenters gaze coordinates to screen fixation using a 2D histogram-peak (mode) estimator.

    This method estimates the fixation location as the densest region of gaze samples
    in physical screen space (millimeters) and applies a rigid translation so that
    this fixation cluster aligns with the geometric center of the screen.

    Rationale
    ---------
    During fixation tasks, the majority of gaze samples cluster around a central
    fixation target. However, global statistics such as the mean or median can be
    biased by:
        - saccades to stimuli
        - asymmetric exploration
        - blinks or signal dropouts
        - drift or calibration offsets

    Instead of using a global median, this method estimates the fixation location
    as the statistical *mode* of the 2D gaze distribution using a coarse spatial
    histogram. The bin with the highest sample count is taken as the fixation
    cluster center.

    Method
    ------
    1. Extract valid (finite) gaze samples in millimeters.
    2. Construct a 2D histogram over the full physical screen extent with
       square bins of size `bin_size_mm`.
    3. Identify the histogram bin with the maximum count (density peak).
    4. Compute the center of that bin (fx, fy).
    5. Compute the geometric screen center (cx, cy) = (W/2, H/2).
    6. Compute translation shift:
           dx = fx - cx
           dy = fy - cy
    7. Apply rigid translation:
           x' = clip(x - dx, 0, W)
           y' = clip(y - dy, 0, H)

    This is a pure translational correction (no rotation or scaling).

    Parameters
    ----------
    df : pandas.DataFrame
        Gaze dataframe with coordinates in millimeters.
    screen_mm : tuple(float, float)
        Physical screen size in millimeters as (width_mm, height_mm).
    x_col : str
        Column name containing horizontal gaze coordinates (mm).
    y_col : str
        Column name containing vertical gaze coordinates (mm).
    bin_size_mm : float
        Size of square bins used in histogram estimation (mm).
        Typical values: 2–5 mm.
        Smaller bins increase spatial precision but may increase noise sensitivity.
    return_shift : bool
        If True, also return the applied translation (dx, dy).

    Returns
    -------
    df_corr : pandas.DataFrame
        Copy of input dataframe with corrected gaze coordinates.
    (dx, dy) : tuple of float
        Translation shift applied in millimeters (optional).

    Assumptions
    -----------
    - Gaze coordinates are already expressed in physical screen space (mm).
    - The dominant cluster corresponds to fixation.
    - The physical screen size is correctly specified.
    - No rotation or scaling errors are present (only translation drift).

    Limitations
    -----------
    - If fixation is not the dominant gaze state, the method may center
      on a task-relevant stimulus instead.
    - Very small bin sizes can overfit noise.
    - Does not correct non-linear distortions or gain errors.

    Notes
    -----
    This method is robust to asymmetric outliers and saccadic excursions
    because it relies on the distribution mode rather than mean/median.
    It is particularly suitable for large fixation-heavy datasets.
    """

    W, H = screen_mm
    cx, cy = W / 2.0, H / 2.0

    df_corr = df.copy()

    x = df_corr[x_col].to_numpy()
    y = df_corr[y_col].to_numpy()

    # Remove invalid samples
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    # Define histogram bins
    x_bins = np.arange(0, W + bin_size_mm, bin_size_mm)
    y_bins = np.arange(0, H + bin_size_mm, bin_size_mm)

    H2d, x_edges, y_edges = np.histogram2d(x, y, bins=[x_bins, y_bins])

    # Find peak bin
    peak_idx = np.unravel_index(np.argmax(H2d), H2d.shape)

    # Convert bin index to center coordinate
    fx = (x_edges[peak_idx[0]] + x_edges[peak_idx[0] + 1]) / 2
    fy = (y_edges[peak_idx[1]] + y_edges[peak_idx[1] + 1]) / 2

    # Compute shift
    dx = fx - cx
    dy = fy - cy

    # Apply correction
    df_corr[x_col] = np.clip(df_corr[x_col] - dx, 0, W)
    df_corr[y_col] = np.clip(df_corr[y_col] - dy, 0, H)

    if return_shift:
        return df_corr, (dx, dy)
    else:
        return df_corr


def gaze_pixel_to_mm(
        df: pd.DataFrame,
        screen_size_mm=(311.25, 249.09),
        screen_resolution_px=(1152, 864),
        center=False,
        overwrite=True,
        **kwargs
    ) -> pd.DataFrame:
    """
    Convert gaze coordinates from pixel units to millimeters using the physical screen size.

    This function performs a linear pixel-to-metric conversion for 2D gaze coordinates.
    It assumes that the gaze coordinates in `df[x_coordinate, y_coordinate]` are defined
    in a screen coordinate system where pixel (0,0) corresponds to the top-left of the
    active display area, and pixel values increase rightward (x) and downward (y).

    The conversion is performed by computing mm-per-pixel scaling factors separately
    for x and y based on the provided physical screen dimensions and the pixel
    resolution:

        mm_per_px_x = screen_width_mm  / screen_width_px
        mm_per_px_y = screen_height_mm / screen_height_px

    Then:

        x_mm = x_px * mm_per_px_x
        y_mm = y_px * mm_per_px_y

    Optionally, the gaze coordinates can be corrected so that the origin is
    at the screen center instead of the top-left corner via :function:`calinet.import.eyelink.correct_to_fixation_hist_peak()`:

        Instead of using a global median, this method estimates the fixation location
        as the statistical *mode* of the 2D gaze distribution using a coarse spatial
        histogram. The bin with the highest sample count is taken as the fixation
        cluster center.

    This produces coordinates in a “centered mm” frame where (0,0) corresponds to
    the geometric center of the screen.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data containing gaze coordinates in pixel units in columns:
        `x_coordinate` and `y_coordinate` (unless you adapt the column names).
    screen_size_mm : tuple(float, float)
        Physical screen size in millimeters as (width_mm, height_mm). This should
        reflect the actual active display area corresponding to the pixel coordinates.
    screen_resolution_px : tuple(int, int)
        Pixel resolution as (width_px, height_px) corresponding to the coordinate
        system of the gaze data (e.g., from EyeLink GAZE_COORDS).
    center : bool
        If True, recenter the converted mm coordinates so the screen center is (0,0).
        If False (default), the origin remains the top-left corner of the screen.
    overwrite : bool
        If True (default), overwrite `x_coordinate` and `y_coordinate` with the
        millimeter coordinates, and drop the temporary `x_mm`, `y_mm` columns.
        If False, keep `x_coordinate`/`y_coordinate` unchanged and add `x_mm`, `y_mm`.

    Returns
    -------
    pandas.DataFrame
        A copy of the input dataframe with gaze coordinates converted to millimeters.

    Notes
    -----
    - This conversion is a pure linear scaling (+ optional translation). It does not
      correct calibration drift, rotation, or non-linear distortions.
    - X/Y scaling are computed independently; if pixels are square and the physical
      dimensions are consistent with the resolution, mm_per_px_x ≈ mm_per_px_y.
    - The function does not clip values to screen bounds; it assumes the input data
      are valid within the pixel coordinate range.

    Comparison to PsPM (pspm_convert_gaze) in "millimeter mode"
    -----------------------------------------------------------
    When `pspm_convert_gaze` is used with:
        conversion.from   = 'pixel'
        conversion.target = 'mm'
        conversion.screen_width / screen_height set (mm)

    PsPM calls `pspm_convert_pixel2unit_core`, which:
      1) scales pixel coordinates to metric units using the provided screen lengths, and
      2) also converts/updates the *channel range* metadata accordingly.

    This Python function matches the essential *numerical scaling* step (pixel→mm),
    but differs in two ways:

      - Metadata/range handling:
        PsPM updates channel header.range consistently; this function does not update
        any external metadata unless you do so elsewhere.

      - Centering behavior:
        In PsPM, centering is primarily relevant when converting to degrees (visual
        angle), because degrees are defined relative to screen center. Your `center=True`
        option provides a centered-mm coordinate frame (useful, but not strictly the
        same as PsPM’s degree conversion pathway).

    If you want strict PsPM equivalence for pixel→mm, use:
      - center=False
      - consistent screen_size_mm and screen_resolution_px derived from the EyeLink
        coordinate system (GAZE_COORDS / DISPLAY_COORDS).
    """

    w_mm = screen_size_mm[0]
    h_mm = screen_size_mm[1]
    w_px, h_px = screen_resolution_px

    mm_per_px_x = w_mm / float(w_px)
    mm_per_px_y = h_mm / float(h_px)

    out = df.copy()
    
    out["x_mm"] = out["x_coordinate"].astype(float) * mm_per_px_x
    out["y_mm"] = out["y_coordinate"].astype(float) * mm_per_px_y

    if overwrite:
        out["x_coordinate"] = out["x_mm"]
        out["y_coordinate"] = out["y_mm"]
        out.drop(["x_mm", "y_mm"], axis=1, inplace=True)

    # center using histograms
    if center:
        defs = {
            "bin_size_mm": 2.0,
        }

        for key, val in defs.items():
            kwargs = update_kwargs(
                kwargs,
                key,
                val
            )

        out = correct_to_fixation_hist_peak(
            out,
            screen_size_mm,
            x_col="x_coordinate",
            y_col="y_coordinate",
            **kwargs
        )

    return out


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
            mask |= vals.fillna(0).astype(float).astype(bool)

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
    ) -> list[tuple[str, str]]:
    """
    Find eye recording files and their paired physioevents files in a directory,
    then set marked event periods to NaN in the eye recordings.

    Returns
    -------
    list of (eye_file, physioevents_file)
        Pairs that were processed.
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
            write_files=True,
            eye_output=str(out_path),
        )

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
        n = len(val["pupil_size"])
        pupil_m = val["pupil_size"].mean()
        pupil_sd = val["pupil_size"].std()
        
        warning_at_mm = config.get("warning_at_mm", 9)
        if pupil_m>warning_at_mm:
            logger.warning(f"Average pupil size ({round(pupil_m, 3)}±{round(pupil_sd, 3)}) > plausible threshold ({warning_at_mm}) over n={n} samples. Output may be invalid..")
        else:
            logger.warning(f"Pupil range is reasonable: {round(pupil_m, 3)}±{round(pupil_sd, 3)}. This is within plausible limit ({warning_at_mm})")

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
            
            # eyelink doesn't need actual dataframe
            inputs = {
                "raw_file": eye_file
            }

        elif eye_file.endswith(".txt"):
            from calinet.imports.smi import fetch_physioevents
            
            # smi needs dataframe to calculate saccades/blinks
            saccade_settings = config.get("smi_settings")
            logger.debug(f"Settings for saccades/blink: {saccade_settings}")
            inputs = {
                "df": val,
                "raw_file": eye_file
            }

            inputs = {**inputs, **saccade_settings}

        else:
            raise TypeError(f"Invalid extension for eye-tracking file. Must be 'asc' for EyeLink or '.txt' from SMI")
        
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
                logger.info(f"Updating EYE_PHYSIO_EVENTS_JSON_TEMPLATE with SMI-information")
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
        logger.info(f"Setting physioevents to NaN in '{out_dir}'")
        _ = mask_eye_recordings_with_physioevents(out_dir, overwrite=True)


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


def estimate_screen_dimensions(
        diagonal_inch: float,
        width_px: int,
        height_px: int
    ) -> tuple:
    """
    Estimate physical screen dimensions from diagonal size and resolution.

    This function computes the physical width and height of a display (in
    millimeters) based on its diagonal size (in inches) and pixel resolution.
    The calculation assumes square pixels and uses the aspect ratio derived
    from the resolution.

    Parameters
    ----------
    diagonal_inch : float
        Screen diagonal in inches.
    width_px : int
        Horizontal resolution of the screen in pixels.
    height_px : int
        Vertical resolution of the screen in pixels.

    Returns
    -------
    width_mm : float
        Estimated screen width in millimeters.
    height_mm : float
        Estimated screen height in millimeters.

    Notes
    -----
    The computation proceeds as follows:

    1. Convert diagonal from inches to millimeters:
       ``diagonal_mm = diagonal_inch * 25.4``

    2. Compute aspect ratio:
       ``r = width_px / height_px``

    3. Solve for height using the Pythagorean relation:
       ``height_mm = diagonal_mm / sqrt(r^2 + 1)``

    4. Compute width:
       ``width_mm = r * height_mm``

    Assumes:
        - Square pixels
        - Accurate diagonal measurement
        - No display scaling

    Examples
    --------
    >>> estimate_screen_dimensions(24, 1920, 1080)
    (531.3..., 298.8...)

    >>> estimate_screen_dimensions(13.3, 2560, 1600)
    (286.4..., 179.0...)
    """

    # Convert diagonal to mm
    diagonal_mm = diagonal_inch * 25.4

    # Aspect ratio
    r = width_px / height_px

    # Compute height in mm
    height_mm = diagonal_mm / math.sqrt(r**2 + 1)

    # Compute width in mm
    width_mm = r * height_mm

    return width_mm, height_mm


def correct_to_fixation(
        df,
        screen_mm: list,
        x_col: str="x_coordinate",
        y_col: str="y_coordinate",
        radius_mm: float=30,
        return_shift: bool=True,
    ) -> tuple:
    """
    Robustly recenters gaze coordinates to screen fixation.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing gaze coordinates in mm.
    screen_mm : tuple(float, float)
        Physical screen size in mm (W, H).
    x_col, y_col : str
        Column names for gaze coordinates.
    radius_mm : float
        Radius (in mm) used to select central fixation cluster.
    return_shift : bool
        Whether to return (dx, dy) along with dataframe.

    Returns
    -------
    df_corr : pandas.DataFrame
        Corrected dataframe.
    (dx, dy) : tuple (optional)
        Estimated correction shift in mm.
    """

    W, H = screen_mm
    cx, cy = W / 2.0, H / 2.0

    df_corr = df.copy()

    x = df_corr[x_col].to_numpy(dtype=float)
    y = df_corr[y_col].to_numpy(dtype=float)

    # Robust overall center estimate
    mx = np.nanmedian(x)
    my = np.nanmedian(y)

    # Select points within radius of cluster center
    mask = (
        np.isfinite(x)
        & np.isfinite(y)
        & ((x - mx) ** 2 + (y - my) ** 2 <= radius_mm**2)
    )

    if np.sum(mask) < 10:
        raise ValueError("Not enough fixation samples inside radius.")

    # Compute shift
    dx = np.nanmedian(x[mask]) - cx
    dy = np.nanmedian(y[mask]) - cy

    # Apply correction
    x_corr = x - dx
    y_corr = y - dy

    # Clip to screen bounds
    x_corr = np.clip(x_corr, 0, W)
    y_corr = np.clip(y_corr, 0, H)

    df_corr[x_col] = x_corr
    df_corr[y_col] = y_corr

    if return_shift:
        return df_corr, (dx, dy)
    else:
        return df_corr
    

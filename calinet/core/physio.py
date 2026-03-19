# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import pandas as pd

from calinet.templates.common import (
    SCR_JSON_CONTENT,
    ECG_JSON_CONTENT,
    RESP_JSON_CONTENT,
    PPG_JSON_CONTENT
)

from calinet.core.io import (
    save_json,
    write_physio_tsv_gz_headerless
)

from calinet import utils
from calinet.config import config, available_labs
from calinet.core.metadata import (
    fill_general,
    fill_scr_json,
    fill_ecg_json,
    fill_ppg_json,
    fill_resp_json
)

from calinet.core import units
from calinet.core.events import extract_onsets_from_ttl

import logging
logger = logging.getLogger(__name__)

cs_duration = config.get("cs_duration", 8)
META_DICT = {
    "scr": [fill_scr_json, SCR_JSON_CONTENT],
    "ecg": [fill_ecg_json, ECG_JSON_CONTENT],
    "ppg": [fill_ppg_json, PPG_JSON_CONTENT],
    "resp": [fill_resp_json, RESP_JSON_CONTENT]
}


def split_onsets(
        event_onsets: list,
        sessions: list
    ) -> tuple:
    """
    Split event onsets into session-specific groups and realign timing.

    This function separates a sequence of event onset times into two sessions
    (e.g., acquisition and extinction) based on temporal gaps between events.
    It also computes a new start time for the second session and shifts its
    onsets accordingly.

    Parameters
    ----------
    event_onsets : list of float
        List of event onset times (in seconds), assumed to be sorted in ascending order.
    sessions : list of str
        List containing exactly two session identifiers (e.g., ["acquisition", "extinction"]).
        The first session is assumed to occur earlier in time.

    Returns
    -------
    extinction_ses_new_start : float
        Estimated new start time (in seconds) for the second session.
    session_onsets : dict
        Dictionary mapping session names to lists of onset times:
        - First session: original onset times
        - Second session: onset times shifted relative to the new session start

    Notes
    -----
    The function operates as follows:

    1. Computes the mean difference between consecutive onsets:
       ``mean_diff = mean(diff(event_onsets))``

    2. Iterates through onsets and assigns them to the first session until a
       temporal gap exceeds ``mean_diff + cs_duration`` (global variable).

    3. Remaining onsets are assigned to the second session.

    4. Computes the temporal gap between sessions:
       ``session_time_diff = first_extinction_onset - last_acquisition_onset``

    5. Estimates a new start time for the second session:
       ``extinction_ses_new_start = last_acquisition_onset + mean_diff / 2``

    6. Shifts second-session onsets relative to the new start time.

    Assumptions:
        - `event_onsets` are ordered
        - Exactly two sessions are provided
        - `cs_duration` is defined in the global scope

    Examples
    --------
    >>> split_onsets(
    ...     event_onsets=[0.5, 1.5, 2.5, 10.0, 11.0],
    ...     sessions=["acquisition", "extinction"]
    ... )
    (new_start_time, {
        "acquisition": [...],
        "extinction": [...]
    })
    """

    session_onsets = {}
    for key in sessions:
        session_onsets[key] = []

    # Calculate the mean of the difference between two adjacent onsets
    onset_diffs = [
        round(event_onsets[i] - event_onsets[i - 1], 3)
        for i in range(1, len(event_onsets))
    ]
    mean_diff = sum(onset_diffs) / len(onset_diffs)
    mean_diff = round(mean_diff, 3)

    last_onset = event_onsets[0]
    extinction_events_first_index = 0
    for i, onset in enumerate(event_onsets):
        onset_diff = onset - last_onset
        if onset_diff < mean_diff + cs_duration:
            session_onsets[sessions[0]].append(onset)
        else:
            extinction_events_first_index = i
            break
        last_onset = onset

    session_onsets[sessions[1]] = event_onsets[extinction_events_first_index:]

    num_acquisition_onsets = len(session_onsets[sessions[0]])
    last_acq_onset = session_onsets[sessions[0]][-1]

    num_extinction_onsets = len(session_onsets[sessions[1]])
    first_ext_onset = session_onsets[sessions[1]][0]

    session_time_diff = round(first_ext_onset - last_acq_onset, 3)

    logger.info(f"{len(event_onsets)}({num_acquisition_onsets} + {num_extinction_onsets}) markers found."
                )
    logger.info(f"Session diff: {session_time_diff}")

    extinction_ses_new_start = last_acq_onset + mean_diff / 2
    logger.info(f"Setting new start time for task-{sessions[1]}: {round(extinction_ses_new_start, 3)}s")

    new_extinction_onsets = [
        onset - extinction_ses_new_start for onset in session_onsets[sessions[1]]
    ]

    session_onsets[sessions[1]] = new_extinction_onsets

    return (extinction_ses_new_start, session_onsets)


def split_df_into_sessions(
        physio_df: pd.DataFrame=None,
        sessions: list=["acquisition", "extinction"],
        sr: float=None,
        **kwargs
    ) -> tuple:
    """
    Split a physiological DataFrame into session-specific segments.

    This function detects event onsets from a physiological signal (e.g., TTL
    channel), determines the temporal boundary between sessions, and splits
    the input DataFrame into separate session-specific DataFrames.

    Parameters
    ----------
    physio_df : pandas.DataFrame, optional
        DataFrame containing physiological recordings. Must include a channel
        suitable for onset detection (e.g., TTL).
    sessions : list of str, default=["acquisition", "extinction"]
        List containing exactly two session identifiers. The first session is
        assumed to occur earlier in time.
    sr : float, optional
        Sampling rate of the physiological recording in Hz. Used to convert
        time (seconds) to sample indices.
    **kwargs
        Additional keyword arguments passed to :func:`extract_onsets_from_ttl`.

    Returns
    -------
    session_onsets : dict
        Dictionary mapping session names to lists of onset times (in seconds),
        as returned by :func:`split_onsets`.
    session_physio_dfs : dict
        Dictionary mapping session names to pandas DataFrames containing the
        corresponding segments of `physio_df`.

    Notes
    -----
    The function performs the following steps:

    1. Onset extraction:
       - Calls :func:`extract_onsets_from_ttl` to detect event onsets from the
         physiological signal.

    2. Session splitting:
       - Calls :func:`split_onsets` to determine the temporal boundary between
         sessions and assign onsets accordingly.

    3. Index conversion:
       - Converts the split time (in seconds) to a sample index:
         ``split_index = round(sr * split_time)``

    4. DataFrame segmentation:
       - Splits `physio_df` into two session-specific DataFrames based on
         `split_index`
       - Resets indices for each segment

    Assumptions:
        - `physio_df` is time-ordered
        - `sr` is correctly specified
        - Exactly two sessions are provided

    Examples
    --------
    >>> onsets, dfs = split_df_into_sessions(
    ...     physio_df=data,
    ...     sr=1000
    ... )

    >>> dfs["acquisition"].head()
    >>> dfs["extinction"].head()
    """

    event_onsets = extract_onsets_from_ttl(
        physio_df,
        sr,
        **kwargs
    )

    logger.info(f"Find splitting point between {sessions} sessions")
    split_time, session_onsets = split_onsets(event_onsets, sessions)

    # convert split_time to index
    split_index = round(sr * split_time)
    logger.info(f"Splitting at t={round(split_time,3)}s [index={split_index}]")

    # split physio_df into ses-01 and ses-02 based on index
    session_physio_dfs = {
        sessions[0]: physio_df.iloc[:split_index].reset_index(drop=True),
        sessions[1]: physio_df.iloc[split_index:].reset_index(drop=True),
    }

    return (session_onsets, session_physio_dfs)


def handle_physio(
        subject_name: str=None,
        subject_new_dir: str=None,
        subject_raw_data_path: str=None,
        lab_name: str=None,
    ) -> dict:
    """
    Process raw physiological data and split it into session-specific outputs.

    This function locates a subject's physiological recording, reads the raw
    data using a lab-specific module, splits the data into predefined sessions
    (e.g., acquisition and extinction), and writes the processed outputs to disk.

    Parameters
    ----------
    subject_name : str, optional
        Identifier for the subject. Used for locating files and naming outputs.
    subject_new_dir : str, optional
        Path to the subject's output directory where processed files will be saved.
    subject_raw_data_path : str, optional
        Path to the directory containing the subject's raw physiological data.
    lab_name : str, optional
        Name of the lab. Used to dynamically load lab-specific processing functions.

    Returns
    -------
    session_onsets : dict
        Dictionary mapping session names (e.g., "acquisition", "extinction") to
        onset times (in seconds), as determined during session splitting.

    Raises
    ------
    Exception
        If:
        - The lab-specific file lookup fails
        - The raw physiological file cannot be read
        - Session splitting or file writing fails
        - No physiological file is found for the subject

    Notes
    -----
    The function performs the following steps:

    1. Module resolution:
       - Retrieves a lab-specific processing module via
         `utils.fetch_lab_module`

    2. File discovery:
       - Locates the physiological acquisition file using
         `module.find_physio_acq_file`

    3. Data loading:
       - Reads the raw physiological data using
         `module.read_raw_physio_file`
       - Returns:
         - `physio_df`: DataFrame with physiological signals
         - `sr`: sampling rate (Hz)
         - `chan_info`: channel metadata

    4. Session splitting and output:
       - Calls `split_and_write_output_files` to:
         - Split data into sessions (default: acquisition, extinction)
         - Write session-specific files to ``<subject_new_dir>/physio/``
         - Extract onset information

    Assumptions:
        - Lab-specific modules implement the required interface
        - Physiological data contains sufficient information for session splitting

    Examples
    --------
    >>> onsets = handle_physio(
    ...     subject_name="sub-01",
    ...     subject_new_dir="derivatives/sub-01/",
    ...     subject_raw_data_path="raw/sub-01/",
    ...     lab_name="NeuroLab"
    ... )

    >>> onsets["acquisition"]
    [0.5, 1.2, 2.0]
    """
    
    # lab-specific module
    module = utils.fetch_lab_module(lab_name)
    
    # run
    try:
        physio_file = module.find_physio_acq_file(
            subject_raw_data_path,
            subject_name
        )
    except Exception as e:
        raise Exception(f"find_physio_acq_file failed: {e}") from e

    if physio_file is not None:
        sessions = ["acquisition", "extinction"]
        try:
            physio_df, sr, chan_info = module.read_raw_physio_file(physio_file)
        except Exception as e:
            raise Exception(f"Error processing physio file {physio_file}: {e}") from e
        
        logger.info(f"Splitting dataframe into sessions: {sessions}")
        try:
            output_path = os.path.join(subject_new_dir, "physio")
            session_onsets = split_and_write_output_files(
                physio_df,
                sr=sr,
                chan_info=chan_info,
                sessions=sessions,
                output_path=output_path,
                subject=subject_name,
                lab_name=lab_name
            )

            return session_onsets
        except Exception as e:
            raise Exception(f"Splitting sessions failed: {e}") from e
    else:
        raise Exception(f"No physio files found for {subject_raw_data_path}") from e


def split_and_write_output_files(
        physio_df: pd.DataFrame,
        sr: float=None,
        chan_info: pd.DataFrame=None,
        sessions: list=None,
        output_path: str=None,
        subject: str=None,
        lab_name: str=None
    ) -> dict:
    """
    Split physiological data into sessions and write modality-specific outputs.

    This function separates a physiological recording into session-specific
    segments (e.g., acquisition and extinction), then processes and writes each
    modality (e.g., SCR, ECG) to disk using lab-specific configurations.

    Parameters
    ----------
    physio_df : pandas.DataFrame
        DataFrame containing physiological recordings. Each column typically
        represents a modality (e.g., SCR, ECG).
    sr : float, optional
        Sampling rate of the physiological recording in Hz.
    chan_info : pandas.DataFrame, optional
        DataFrame containing channel metadata (e.g., channel names, types).
    sessions : list of str, optional
        List of session identifiers (e.g., ["acquisition", "extinction"]).
    output_path : str, optional
        Directory where output files will be written.
    subject : str, optional
        Subject identifier used in output file naming.
    lab_name : str, optional
        Name of the lab. Used to retrieve lab-specific configuration such as
        modality definitions and gap factors.

    Returns
    -------
    session_onsets : dict
        Dictionary mapping session names to lists of onset times (in seconds),
        as returned by :func:`split_df_into_sessions`.

    Raises
    ------
    Exception
        If:
        - Session splitting fails
        - Processing or writing of any modality fails

    Notes
    -----
    The function performs the following steps:

    1. Gap factor determination:
       - Retrieves a lab-specific ``gap_factor_between_acq_ext`` from
         `available_labs`
       - Falls back to a default value of 5 if unavailable
       - This factor defines what constitutes a "long gap" between sessions

    2. Session splitting:
       - Calls :func:`split_df_into_sessions` to:
         - Detect onset events
         - Determine session boundaries
         - Split the DataFrame accordingly

    3. Modality processing:
       - Iterates over available modalities defined in
         ``available_labs[lab_name]["Modalities"]``
       - Calls `handle_modality` for each modality and session
       - Writes outputs to ``output_path``

    Assumptions:
        - `physio_df` contains columns corresponding to defined modalities
        - Lab configuration in `available_labs` is correctly specified
        - `handle_modality` implements modality-specific processing and writing

    Examples
    --------
    >>> onsets = split_and_write_output_files(
    ...     physio_df=data,
    ...     sr=1000,
    ...     chan_info=channels,
    ...     sessions=["acquisition", "extinction"],
    ...     output_path="derivatives/sub-01/physio/",
    ...     subject="sub-01",
    ...     lab_name="NeuroLab"
    ... )
    """

    # gap factor is multiplied with min(ITI) to define what a long gap is (e.g., between acquisition and extinction phase).
    # Reading has notably shorter interval, so needs to be lab-specific. Default
    # to 5; which is about 90s
    try:
        gap_factor = available_labs.get(lab_name).get("gap_factor_between_acq_ext")
    except:
        gap_factor = 5

    # split sessions
    try:
        session_onsets, session_physio_dfs = split_df_into_sessions(
            physio_df=physio_df,
            sessions=sessions,
            sr=sr,
            gap_factor=gap_factor
        )
    except Exception as e:
        raise Exception(f"Error while splitting sessions: {e}") from e

    # save to disk
    for task_id, task_df in session_physio_dfs.items():
        
        # Define the output directory for physio data (using a 'physio' folder).
        available_mods = available_labs.get(lab_name).get("Modalities")
        for mod in available_mods:
            logger.info(f"Processing modality='{mod}' | task={task_id}")
            try:
                handle_modality(
                    modality_name=mod,
                    modality_data=task_df[mod],
                    output_path=output_path,
                    subject=subject,
                    lab_name=available_labs.get(lab_name).get("MetaName"),
                    task=task_id,
                    chan_info=chan_info
                )
            except Exception as e:
                raise Exception(f"Failed to handle '{mod}' for task-{task_id}: {e}") from e

    return session_onsets


def handle_modality(
        modality_name: str=None,
        modality_data: pd.DataFrame=None,
        output_path: str=None,
        subject: str=None,
        lab_name: str=None,
        task: str="acquisition",
        chan_info: pd.DataFrame=None
    ) -> None:
    """
    Process and write a single physiological modality to disk.

    This function writes modality-specific physiological data (e.g., SCR, ECG)
    to a compressed TSV file and generates a corresponding JSON metadata file.
    It enriches metadata using both raw channel information and lab-specific
    configuration.

    Parameters
    ----------
    modality_name : str, optional
        Name of the physiological modality (e.g., "scr", "ecg").
    modality_data : pandas.DataFrame, optional
        DataFrame containing the physiological signal for the given modality.
    output_path : str, optional
        Directory where output files will be written.
    subject : str, optional
        Subject identifier used in output file naming.
    lab_name : str, optional
        Name of the lab. Used to retrieve metadata and lab-specific settings.
    task : str, default="acquisition"
        Task or session identifier (e.g., "acquisition", "extinction").
    chan_info : pandas.DataFrame, optional
        DataFrame containing channel metadata (e.g., sampling rate, units).

    Returns
    -------
    None
        This function does not return any value. It writes TSV and JSON files
        to disk.

    Raises
    ------
    Exception
        If:
        - Writing the TSV file fails
        - Extracting modality-specific metadata fails

    Notes
    -----
    The function performs the following steps:

    1. File writing:
       - Writes the physiological data to a compressed TSV file:
         ``<subject>_task-<task>_recording-<modality>_physio.tsv.gz``

    2. Metadata initialization:
       - Retrieves a metadata template from `META_DICT`
       - Extracts modality-specific settings

    3. Channel metadata integration (if available):
       - Sets sampling frequency from `chan_info["samples_per_second"]`
       - Normalizes units using `units.normalize_bids_unit`

    4. Metadata enrichment:
       - Adds general metadata using `fill_general`
       - Adds modality-specific metadata using a function defined in `META_DICT`

    5. JSON writing:
       - Saves metadata to a JSON file corresponding to the TSV file

    Assumptions:
        - `META_DICT` contains valid templates and functions for each modality
        - `chan_info` includes columns "output_name", "samples_per_second", and "units"
        - `modality_data` is properly formatted for writing

    Examples
    --------
    >>> handle_modality(
    ...     modality_name="scr",
    ...     modality_data=data["scr"],
    ...     output_path="derivatives/sub-01/physio/",
    ...     subject="sub-01",
    ...     lab_name="NeuroLab",
    ...     task="acquisition",
    ...     chan_info=channel_info
    ... )
    """

    os.makedirs(output_path, exist_ok=True)
    base_filename = f"{subject}_task-{task}_recording"
    tsv_path = os.path.join(
        output_path,
        f"{base_filename}-{modality_name.lower()}_physio.tsv.gz"
    )

    logger.info(f"Writing {tsv_path}")
    try:
        write_physio_tsv_gz_headerless(
            modality_data,
            tsv_path
        )
    except Exception as e:
        raise Exception(f"write_physio_tsv_gz_headerless failed: {e}") from e

    # update JSON with chan_info
    mod_content = META_DICT.get(modality_name.lower())[-1]
    mod_settings = mod_content.get(modality_name.lower())
    if isinstance(chan_info, pd.DataFrame):
        logger.debug("Reading metadata from raw channel info")
        # set sampling frequency
        mod_content["SamplingFrequency"] = chan_info.set_index("output_name").at[modality_name, "samples_per_second"]

        # set units
        raw_unit = chan_info.set_index("output_name").at[modality_name, "units"]
        mod_settings["Units"] = units.normalize_bids_unit(raw_unit)

    # fill modality-agnostic information
    logger.debug(f"Fetching general information from metadata.csv")
    mod_content = fill_general(
        lab_name,
        modality_name,
        mod_content
    )

    # fill modality-specific metadata
    logger.debug(f"Fetching '{modality_name}'-specific information from metadata.csv")
    
    # first element is function
    f = META_DICT.get(modality_name.lower())[0]

    try:
        mod_content = f(lab_name, mod_content)
    except Exception as e:
        raise Exception(f"Could not extract '{modality_name}' specific metadata from lab='{lab_name}' with function {f}: {e}") from e

    # write
    json_path = tsv_path.replace(".tsv.gz", ".json")
    logger.info(f"Writing {json_path}")
    save_json(json_path, mod_content)

    logger.info(f"Successfully created '{task}'-files")

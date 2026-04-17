# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import re
import copy
import numpy as np
import pandas as pd

from calinet.core.metadata import (
    df_meta,
    stimulus_presentation_from_metadata
)

from typing import Dict, Tuple, Optional
from calinet.utils import filter_non_printable
from calinet.config import eyelink_regex

import logging
logger = logging.getLogger(__name__)


_RE_CAL_TYPE    = eyelink_regex["CAL_TYPE"]
_RE_CAL_VALID   = eyelink_regex["CAL_VALID"]
_RE_ELCL_PROC   = eyelink_regex["ELCL_PROC"]
_RE_GAZE_COORDS = eyelink_regex["GAZE_COORDS"]
_RE_PUPIL       = eyelink_regex["PUPIL"]
_RE_RATE        = eyelink_regex["RATE"]


def _parse_num(value):
    """Convert EyeLink numeric strings safely."""
    value = filter_non_printable(value).strip()

    if value in {".", "...", ""}:
        return np.nan

    return float(value)


def asc_to_df(
        asc_file: str
    ) -> Tuple[Dict[str, pd.DataFrame], Optional[str]]:
    """
    Convert an EyeLink ASC file into left and right eye dataframes.

    This function parses an ASC file exported from an eye tracker, extracts
    gaze coordinates and pupil measurements, and separates data into left and
    right eye streams depending on the recording mode (monocular or binocular).

    Parameters
    ----------
    asc_file : str
        Path to the ASC file.

    Returns
    -------
    data : dict of str to pd.DataFrame
        Dictionary containing:
        - ``"eye1"``: dataframe for left eye data
        - ``"eye2"``: dataframe for right eye data
        Each dataframe contains columns ``["x_coordinate", "y_coordinate", "pupil_size"]``.
    pupil_measurement_type : str or None
        Type of pupil measurement detected in the file (e.g., ``"AREA"`` or
        ``"DIAMETER"``). Returns ``None`` if not detected.

    Raises
    ------
    Exception
        If an error occurs while reading or parsing the ASC file.

    Notes
    -----
    - Recording mode is inferred from ``START`` lines (LEFT, RIGHT, BINOCULAR).
    - Pupil measurement type is inferred from ``SAMPLES`` or ``PUPIL`` lines.
    - Only rows starting with numeric timestamps are treated as valid samples.
    - In monocular mode, data is assigned entirely to one eye.
    - In binocular mode, both left and right eye data are extracted per row.

    Examples
    --------
    >>> data, pupil_type = asc_to_df("subject.asc")
    >>> data["eye1"].head()
    """

    cols = ["x_coordinate", "y_coordinate", "pupil_size"]

    left_rows = []
    right_rows = []

    recording_mode = None
    pupil_measurement_type = None

    logger.debug(f"Converting {asc_file} to DataFrame")
    try:
        with open(asc_file, "r", encoding="utf-8", errors="ignore") as infile:
            for line in infile:

                stripped = line.strip()
                if not stripped:
                    continue

                columns = stripped.split()

                # detect recording eye
                if stripped.startswith("START"):
                    if "BINOCULAR" in columns:
                        recording_mode = "BINOCULAR"
                    elif "RIGHT" in columns:
                        recording_mode = "RIGHT"
                    elif "LEFT" in columns:
                        recording_mode = "LEFT"
                    continue

                # detect pupil type
                if stripped.startswith("SAMPLES") or stripped.startswith("PUPIL"):
                    if "AREA" in columns:
                        pupil_measurement_type = "AREA"
                    elif "DIAMETER" in columns:
                        pupil_measurement_type = "DIAMETER"
                    continue

                # sample rows start with timestamp
                if not stripped[0].isdigit():
                    continue

                # MONOCULAR
                if recording_mode in ("LEFT", "RIGHT"):

                    if len(columns) < 4:
                        continue

                    x = _parse_num(columns[1])
                    y = _parse_num(columns[2])
                    pupil = _parse_num(columns[3])

                    row = [x, y, pupil]

                    if recording_mode == "LEFT":
                        left_rows.append(row)
                    else:
                        right_rows.append(row)

                # BINOCULAR
                elif recording_mode == "BINOCULAR":

                    if len(columns) < 7:
                        continue

                    lx = _parse_num(columns[1])
                    ly = _parse_num(columns[2])
                    lp = _parse_num(columns[3])
                    left_rows.append([lx, ly, lp])

                    rx = _parse_num(columns[4])
                    ry = _parse_num(columns[5])
                    rp = _parse_num(columns[6])
                    right_rows.append([rx, ry, rp])

        df_left = pd.DataFrame(left_rows, columns=cols)
        df_right = pd.DataFrame(right_rows, columns=cols)

    except Exception as e:
        raise Exception(f"Error converting ASC to DataFrames: {e}") from e

    return {
        "eye1": df_left,
        "eye2": df_right
    }, pupil_measurement_type


def get_eyetracker_setup_info(
        asc_file: str,
        eye_name: Optional[str]=None,
        eye_tpl: Optional[dict]=None,
        lab_name: Optional[str]=None
    ) -> Tuple[dict, dict]:
    """
    Extract eye tracker setup information from an ASC file header.

    This function parses the header section of an EyeLink ASC file (up to
    ``'!MODE RECORD'``) to extract calibration details, sampling frequency,
    gaze range, pupil measurement type, and lab-specific metadata. The result
    is merged into a template dictionary and returned along with stimulus
    presentation metadata.

    Parameters
    ----------
    asc_file : str
        Path to the ASC file.
    eye_name : str, optional
        Name of the recorded eye (e.g., ``"LEFT"``, ``"RIGHT"``). Used to filter
        calibration and validation entries.
    eye_tpl : dict, optional
        Template dictionary to populate with extracted metadata. A deep copy is
        used to avoid modifying the original.
    lab_name : str, optional
        Name of the lab used to retrieve additional metadata such as manufacturer,
        model, software version, and stimulus presentation parameters.

    Returns
    -------
    info : dict
        Dictionary containing extracted eye tracker metadata, including fields
        such as ``CalibrationType``, ``SamplingFrequency``, ``GazeRange``,
        and others.
    stim_pres : dict
        Dictionary containing stimulus presentation metadata derived from lab
        configuration.

    Raises
    ------
    Exception
        If an error occurs while reading or parsing the ASC file.

    Notes
    -----
    - Parsing stops once ``'!MODE RECORD'`` is encountered.
    - Calibration and validation values are extracted using regular expressions.
    - Gaze range values may be overwritten by stimulus presentation metadata.
    - If `eye_tpl` is provided, coordinate units are set to millimeters (``mm``).
    - Lab metadata is retrieved via `df_meta` and
      `stimulus_presentation_from_metadata`.

    Examples
    --------
    >>> info, stim = get_eyetracker_setup_info(
    ...     asc_file="subject.asc",
    ...     eye_name="LEFT",
    ...     lab_name="austin"
    ... )
    >>> info["SamplingFrequency"]
    """

    logger.debug(f"Creating StimulusPresentation-dict from {asc_file}")
    # Deep copy to avoid mutating nested dicts in the template
    if isinstance(eye_tpl, dict):
        info = copy.deepcopy(eye_tpl)
    else:
        info = {}

    if eye_name is not None:
        eye = eye_name.strip()
        eye_lower = eye.lower()
        info["RecordedEye"] = eye_lower
    else:
        logger.warning(f"Cannot set 'RecordedEye' without eye_name")

    # Specific information
    if lab_name is not None:
        meta_idx = df_meta.set_index("Parameter")
        manufacturer = meta_idx.at["Eyetracker Manufacturer", lab_name]
        model = meta_idx.at["Eyetracker Manufacturer Model", lab_name]
        software = meta_idx.at["Eyetracker Software Version", lab_name]
        
        info["Manufacturer"] = manufacturer
        info["ManufacturersModelName"] = model
        info["SoftwareVersions"] = software

        # get stimulus presentation from metadata
        stim_pres = stimulus_presentation_from_metadata(df_meta, lab_name)

    else:
        logger.warning(f"Cannot determine general metadata without lab_name")
        stim_pres = {}

    # Track which fields we already found so we can stop early
    found = {
        "CalibrationType": False,
        "AverageCalibrationError": False,
        "PupilFitMethod": False,
        "GazeRange": False,
        "MeasurementType": False,
        "SamplingFrequency": False,
    }

    with open(
            asc_file,
            "r",
            encoding="utf-8",
            errors="ignore"
        ) as f:
        
        for raw in f:
            line = raw.strip()

            # Stop once recording starts
            if re.match(r"^\d+\s", line):
                break

            # Calibration type
            if (not found["CalibrationType"]
                and line.startswith(">>>>>>> CALIBRATION")
                and eye in line):
                m = _RE_CAL_TYPE.search(line)
                if m:
                    info["CalibrationType"] = m.group(1)
                    found["CalibrationType"] = True
                continue

            # Calibration validation errors
            if (not found["AverageCalibrationError"]
                and "!CAL VALIDATION" in line
                and eye in line):
                m = _RE_CAL_VALID.search(line)
                if m:
                    info["AverageCalibrationError"] = float(m.group(1))
                    info["MaximalCalibrationError"] = float(m.group(2))
                    found["AverageCalibrationError"] = True
                continue

            # Pupil fit method
            if not found["PupilFitMethod"] and "ELCL_PROC" in line:
                m = _RE_ELCL_PROC.search(line)
                if m:
                    info["PupilFitMethod"] = m.group(1)
                    found["PupilFitMethod"] = True
                continue

            # Gaze coords
            if not found["GazeRange"] and "GAZE_COORDS" in line:
                m = _RE_GAZE_COORDS.search(line)
                if m:
                    info.setdefault("GazeRange", {})
                    info["GazeRange"]["xmin"] = int(float(m.group(1)))
                    info["GazeRange"]["ymin"] = int(float(m.group(2)))
                    info["GazeRange"]["xmax"] = int(float(m.group(3)))
                    info["GazeRange"]["ymax"] = int(float(m.group(4)))
                    found["GazeRange"] = True
                continue

            # Pupil measurement type
            if not found["MeasurementType"]:
                m = _RE_PUPIL.search(line)
                if m:
                    info["MeasurementType"] = m.group(1)
                    found["MeasurementType"] = True
                continue
            
            # Sampling rate line often looks like: "SAMPLES ... RATE 1000 ..."
            if not found["SamplingFrequency"] and line.startswith("SAMPLES") and "RATE" in line:
                m = _RE_RATE.search(line)
                if m:
                    info["SamplingFrequency"] = int(float(m.group(1)))
                    found["SamplingFrequency"] = True
                continue

            # Optional: stop early if we've found everything we care about
            if all(found.values()):
                break

    # overwrite screen size with millimeters for later conversion
    if len(stim_pres)>0:
        screen_mm = stim_pres.get("StimulusPresentation").get("ScreenSize")
        logger.warning(f"Updating GazeRange with specified screen size: {screen_mm}; was {info['GazeRange']}")
        for ix, i in enumerate(["xmax", "ymax"]):
            info["GazeRange"][i] = screen_mm[ix]

    # set all units to mm
    if isinstance(eye_tpl, dict):
        for col in ["x_coordinate", "y_coordinate", "pupil_size"]:
            info[col]["Units"] = "mm"
    
    logger.info(f"StimulusPresentation: {stim_pres}")
    logger.info(f"Metadata: {info}")

    return info, stim_pres


def _get_first_sample_timestamp(
        raw_file: str
    ) -> int:
    """
    Extract the first valid sample timestamp from an ASC file.

    This function scans an EyeLink ASC file and returns the first timestamp
    corresponding to a valid sample row after recording has started. A valid
    sample row is defined as a line beginning with an integer timestamp followed
    by multiple numeric values.

    Parameters
    ----------
    raw_file : str
        Path to the ASC file.

    Returns
    -------
    timestamp : int
        First detected sample timestamp.

    Raises
    ------
    ValueError
        If no valid sample timestamp is found in the file.

    Notes
    -----
    - Parsing begins only after encountering a line starting with ``START``.
    - A valid sample row must:
        - Start with an integer timestamp
        - Contain at least three additional numeric values
    - Non-numeric or malformed lines are ignored.

    Examples
    --------
    >>> ts = _get_first_sample_timestamp("subject.asc")
    >>> ts
    """

    in_recording = False

    with open(raw_file, "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            s = line.strip()
            if not s:
                continue

            if s.startswith("START"):
                in_recording = True
                continue

            if not in_recording:
                continue

            parts = re.split(r"\s+", s)

            # sample rows must begin with integer timestamp
            if not re.fullmatch(r"\d+", parts[0]):
                continue

            # require multiple following numeric fields to distinguish
            # samples from stray numeric lines
            numeric_after_ts = 0
            for tok in parts[1:8]:
                try:
                    float(tok)
                    numeric_after_ts += 1
                except ValueError:
                    pass

            if numeric_after_ts >= 3:
                return int(parts[0])

    raise ValueError(f"No sample timestamp found in {raw_file}")


def fetch_physioevents(
        raw_file: Optional[str]=None,
        drop_negative_msgs: bool=False
    ) -> Tuple[pd.DataFrame, None]:
    """
    Parse EyeLink events and return a dataframe of event timings.

    This function extracts fixation, saccade, blink, and message events from an
    EyeLink ASC file and converts their timestamps to seconds relative to the
    first recorded sample. Events are returned in a sorted dataframe suitable
    for further processing.

    Parameters
    ----------
    raw_file : str, optional
        Path to the ASC file.
    drop_negative_msgs : bool, default=False
        Whether to exclude message events with negative onset times (relative
        to the first sample timestamp).

    Returns
    -------
    events_df : pd.DataFrame
        Dataframe containing parsed events with columns:
        - ``onset`` (float): event onset in seconds
        - ``duration`` (float or "n/a"): event duration in seconds
        - ``trial_type`` (str): type of event (fixation, saccade, blink, or "n/a")
        - ``blink`` (str): "1" for blink events, "0" otherwise, or "n/a"
        - ``message`` (str): message content for MSG events
    None
        Placeholder value to maintain compatibility with expected return format.

    Raises
    ------
    ValueError
        If no valid sample timestamp can be extracted from the file.

    Notes
    -----
    - Event timestamps are normalized relative to the first sample timestamp.
    - Fixation, saccade, and blink events are constructed from paired start/end
      markers (e.g., ``SFIX``/``EFIX``).
    - Message events (``MSG``) are included with their text content.
    - Lines containing ``!`` or ``VALIDATE`` are ignored.
    - Output is sorted by onset time using a stable sort.

    Examples
    --------
    >>> events_df, _ = fetch_physioevents("subject.asc")
    >>> events_df.head()
    """
    
    rows = []
    ongoing_events = {}

    t0_raw = _get_first_sample_timestamp(raw_file)
    logger.info(f"Index of first sample={t0_raw}")

    with open(raw_file, "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            if "!" in line or "VALIDATE" in line:
                continue

            stripped = line.strip()
            if not stripped:
                continue

            parts = re.split(r"\s+", stripped)
            event_type = parts[0]

            if event_type in ["SFIX", "SSACC", "SBLINK"]:
                ongoing_events[event_type] = parts

            elif event_type in ["EFIX", "ESACC", "EBLINK"]:
                start_event_type = event_type.replace("E", "S")
                start_event = ongoing_events.pop(start_event_type, None)

                if start_event:
                    start_ts = int(start_event[2])
                    onset_sec = (start_ts - t0_raw) / 1000.0
                    duration_sec = float(parts[4]) / 1000.0

                    trial_type = (
                        "fixation" if event_type == "EFIX"
                        else "saccade" if event_type == "ESACC"
                        else "blink"
                    )
                    blink = "1" if event_type == "EBLINK" else "0"

                    rows.append({
                        "onset": onset_sec,
                        "duration": duration_sec,
                        "trial_type": trial_type,
                        "blink": blink,
                        "value": "n/a",
                    })

            elif event_type == "MSG":
                msg_ts = int(parts[1])
                onset_sec = (msg_ts - t0_raw) / 1000.0
                message = " ".join(parts[2:])

                if not (drop_negative_msgs and onset_sec < 0):
                    rows.append({
                        "onset": onset_sec,
                        "duration": "n/a",
                        "trial_type": "n/a",
                        "blink": "n/a",
                        "message": message,
                    })

    # make dataframe
    events_df = pd.DataFrame(
        rows,
        columns=["onset", "duration", "trial_type", "blink", "message"]
    )

    events_df = events_df.sort_values("onset", kind="stable").reset_index(drop=True)

    # return tuple to maintain compatibility
    return events_df, None

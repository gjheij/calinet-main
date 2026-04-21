# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import io
import re
import copy
import numpy as np
import pandas as pd
from copy import deepcopy
from calinet.core.metadata import (
    df_meta,
    stimulus_presentation_from_metadata
)

from calinet.utils import ensure_timestamp
from calinet.core.utils import fetch_physioevents_from_df

import logging
logger = logging.getLogger(__name__)

from typing import Optional, Union, Tuple, Dict, Any, List


_RE_KV = re.compile(r"^##\s*([^:]+):\s*(.*)$")


def parse_smi_header_kv(
        raw_file_path: str
    ) -> dict:
    """
    Parse key-value metadata from an SMI/iView text export header.

    This function reads a text file exported from SMI/iView software and
    extracts metadata from lines starting with ``'##'``. Key-value pairs are
    parsed using a regular expression and returned as a flat dictionary.

    Parameters
    ----------
    raw_file_path : str
        Path to the SMI/iView text file.

    Returns
    -------
    meta : dict
        Dictionary mapping header keys to their corresponding values.

    Notes
    -----
    - Only lines starting with ``'##'`` are considered part of the header.
    - Key-value pairs are extracted using the `_RE_KV` regular expression.
    - If the header is contiguous at the top of the file, parsing could be
      optimized by stopping once non-header lines are encountered.

    Examples
    --------
    >>> meta = parse_smi_header_kv("recording.txt")
    >>> meta.get("Sample Rate")
    """

    meta = {}
    with open(raw_file_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.rstrip("\n")

            if not line.startswith("##"):
                # Many SMI exports put header at top then data.
                # If you know header is contiguous, you can break here once it ends:
                continue

            m = _RE_KV.match(line)
            if m:
                key = m.group(1).strip()
                val = m.group(2).strip()
                meta[key] = val

    return meta


def _candidate_smi_encodings(
        raw_file: str
    ) -> Tuple[str, ...]:
    """
    Infer candidate text encodings for an SMI/iView export file.

    This function inspects the first bytes of a file to detect byte order
    marks (BOM) or patterns indicative of UTF-16 encoding. Based on this
    heuristic, it returns an ordered tuple of likely encodings to try when
    reading the file.

    Parameters
    ----------
    raw_file : str
        Path to the input file.

    Returns
    -------
    encodings : tuple of str
        Tuple of candidate encodings ordered by likelihood.

    Notes
    -----
    - UTF-16 encodings are inferred from BOM markers or null-byte patterns.
    - UTF-8 and common fallback encodings (``cp1252``, ``latin1``) are always
      included as fallbacks.
    - The function reads only the first 4096 bytes for detection.
    - Heuristic detection may not always be correct but provides a robust
      starting point for decoding.

    Examples
    --------
    >>> encodings = _candidate_smi_encodings("recording.txt")
    >>> encodings[0]
    """

    with open(raw_file, "rb") as f:
        head = f.read(4096)

    if head.startswith(b"\xff\xfe"):
        return ("utf-16", "utf-16-le", "utf-8", "cp1252", "latin1")
    if head.startswith(b"\xfe\xff"):
        return ("utf-16", "utf-16-be", "utf-8", "cp1252", "latin1")
    if head.startswith(b"\xef\xbb\xbf"):
        return ("utf-8-sig", "utf-8", "cp1252", "latin1")

    even_nuls = head[0::2].count(b"\x00")
    odd_nuls = head[1::2].count(b"\x00")

    if len(head) >= 8:
        if odd_nuls > len(head[1::2]) * 0.2:
            return ("utf-16-le", "utf-8", "cp1252", "latin1")
        if even_nuls > len(head[0::2]) * 0.2:
            return ("utf-16-be", "utf-8", "cp1252", "latin1")

    return ("utf-8", "utf-8-sig", "cp1252", "latin1", "utf-16-le", "utf-16-be")


def _decode_smi_text(
        raw_file: str
    ) -> Tuple[str, str]:
    """
    Decode an SMI/iView text file using candidate encodings.

    This function attempts to decode a file by trying a sequence of likely
    encodings inferred from `_candidate_smi_encodings`. It selects the first
    encoding that successfully decodes the file without producing excessive
    null characters.

    Parameters
    ----------
    raw_file : str
        Path to the input file.

    Returns
    -------
    text : str
        Decoded text content of the file.
    encoding : str
        Encoding used to successfully decode the file.

    Raises
    ------
    ValueError
        If none of the candidate encodings can decode the file.

    Notes
    -----
    - Candidate encodings are obtained from `_candidate_smi_encodings`.
    - Decodings that produce a high fraction of null characters (``\\x00``)
      are rejected as likely incorrect.
    - The entire file is read as raw bytes before attempting decoding.

    Examples
    --------
    >>> text, enc = _decode_smi_text("recording.txt")
    >>> enc
    """

    encodings = _candidate_smi_encodings(raw_file)
    last_error = None

    with open(raw_file, "rb") as f:
        raw_bytes = f.read()

    for enc in encodings:
        try:
            text = raw_bytes.decode(enc)

            # If decoding with a single-byte codec leaves lots of NULs,
            # it was almost certainly the wrong choice.
            nul_fraction = text.count("\x00") / max(len(text), 1)
            if nul_fraction > 0.01:
                continue

            return text, enc
        except UnicodeError as e:
            last_error = e
            continue

    raise ValueError(
        f"Could not decode SMI file {raw_file!r}. "
        f"Tried encodings={encodings}. Last error={last_error}"
    )


def _parse_meta_and_header_from_text(
        text: str,
        raw_file: str
    ) -> Tuple[Dict[str, str], int, List[str]]:
    """
    Parse metadata and locate the tab-delimited header row in SMI text.

    Scans ``text`` line by line, collecting metadata from lines that begin
    with ``"##"`` and identifying the first header row that begins with
    ``"Time\\tType\\tTrial\\t"``.

    Parameters
    ----------
    text : str
        Full raw text content to parse.
    raw_file : str
        Source file name or path used in the error message when the header
        row cannot be found.

    Returns
    -------
    meta : dict of str to str
        Metadata parsed from ``"##"`` key-value lines.
    header_line_index : int
        Zero-based index of the detected header row.
    lines : list of str
        All lines from ``text`` as returned by ``splitlines()``.

    Raises
    ------
    ValueError
        Raised if no header row starting with
        ``"Time\\tType\\tTrial\\t"`` is found in ``text``. The exception
        message includes ``raw_file`` and a preview of the first 10 lines.

    Notes
    -----
    Metadata is only collected from lines that begin with ``"##"`` and
    match ``_RE_KV``.
    """

    meta = {}
    header_line_index = None
    lines = text.splitlines()

    for i, line in enumerate(lines):
        if line.startswith("##"):
            m = _RE_KV.match(line)
            if m:
                meta[m.group(1).strip()] = m.group(2).strip()
            continue

        if line.startswith("Time\tType\tTrial\t"):
            header_line_index = i
            break

    if header_line_index is None:
        preview = "\n".join(lines[:10])
        raise ValueError(
            f"Could not find SMI header row in {raw_file!r}. "
            f"First lines were:\n{preview}"
        )

    return meta, header_line_index, lines


def _extract_msg_rows_from_decoded_lines(
        lines: List[str],
        header_line_index: int
    ) -> pd.DataFrame:
    """
    Extract ``"MSG"`` rows from decoded SMI text lines.

    Iterates over lines following the header, splits each line using
    ``split("\\t", 3)`` to preserve tabs within the payload, and collects
    message timestamps and text content.

    Parameters
    ----------
    lines : list of str
        Decoded text lines from the SMI file.
    header_line_index : int
        Zero-based index of the header row in ``lines``.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame with columns ``"Time"`` and ``"message"``. The ``"Time"``
        column contains numeric timestamps, and ``"message"`` contains
        cleaned message strings. Returns an empty DataFrame with these
        columns if no valid ``"MSG"`` rows are found.

    Notes
    -----
    - Removes null characters (``"\\x00"``) from lines.
    - Strips ``"# Message:"`` prefixes from payloads.
    - Replaces empty or missing messages with ``"n/a"``.
    - Non-numeric timestamps are skipped.
    """

    msg_times = []
    msg_texts = []

    for raw in lines[header_line_index + 1:]:
        line = raw.rstrip("\n").replace("\x00", "")

        if not line:
            continue

        parts = line.split("\t", 3)
        if len(parts) < 4:
            continue

        time_str, typ, trial, payload = parts

        if typ != "MSG":
            continue

        try:
            msg_time = float(time_str)
        except ValueError:
            continue

        payload = re.sub(r"^# Message:\s*", "", payload.strip())

        if not payload:
            payload = "n/a"

        msg_times.append(msg_time)
        msg_texts.append(payload)

    if not msg_times:
        return pd.DataFrame(columns=["Time", "message"])

    return pd.DataFrame({
        "Time": pd.to_numeric(pd.Series(msg_times), errors="coerce"),
        "message": pd.Series(msg_texts, dtype="string").replace({"": "n/a"}).fillna("n/a"),
    })


def simple_smi_read(
        raw_file: str
    ) -> Tuple[pd.DataFrame, Dict[str, str], str, str, List[str], int]:
    """
    Read and parse an SMI text file into a structured DataFrame.

    Decodes the raw file, extracts metadata and header position, rebuilds
    a clean tab-delimited text buffer starting at the header row, and
    loads the data into a pandas DataFrame.

    Parameters
    ----------
    raw_file : str
        Path to the raw SMI file to read.

    Returns
    -------
    df : pandas.DataFrame
        Parsed tabular data with all columns read as strings.
    meta : dict of str to str
        Metadata extracted from ``"##"`` lines in the file.
    text_for_table : str
        Cleaned text buffer starting from the header row, used for parsing.
    enc : str
        Detected file encoding used during decoding.
    lines : list of str
        All decoded lines from the original file.
    header_line_index : int
        Zero-based index of the detected header row.

    Raises
    ------
    ValueError
        Raised if the required ``"Type"`` column is missing after parsing.

    Notes
    -----
    - Removes null characters (``"\\x00"``) from malformed exports.
    - Column names are stripped and cleaned of BOM characters (``"\\ufeff"``).
    - Reads all columns as ``str`` to preserve raw values.
    """
    
    text, enc = _decode_smi_text(raw_file)
    meta, header_line_index, lines = _parse_meta_and_header_from_text(text, raw_file)

    logger.debug(f"Header starts at index={header_line_index}")

    # Rebuild a clean text buffer starting at the header row.
    text_for_table = "\n".join(lines[header_line_index:])

    # Final safeguard against stray NULs in malformed exports.
    if "\x00" in text_for_table:
        text_for_table = text_for_table.replace("\x00", "")

    text_stream = io.StringIO(text_for_table)
    logger.debug("Reading text with io.StringIO")
    df = pd.read_csv(
        text_stream,
        sep="\t",
        dtype=str
    )

    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]

    if "Type" not in df.columns:
        raise ValueError(
            f"'Type' column missing in {raw_file}. Columns found: {list(df.columns)}"
        )

    return df, meta, text_for_table, enc, lines, header_line_index


def smi_txt_to_df(
        raw_file: str,
        sr: Union[float, int]=None,
        return_full: bool=False
    ) -> Tuple[Dict[str, pd.DataFrame], str]:
    """
    Convert an SMI/iView TXT export into eye-specific DataFrames.

    Reads the raw file, extracts sample rows (``"SMP"``), and constructs
    left and right eye DataFrames containing gaze coordinates and pupil
    size. Optionally adds timestamps based on a provided sampling rate.

    Parameters
    ----------
    raw_file : str
        Path to the SMI/iView TXT file.
    sr : float or int, optional
        Sampling rate used to generate timestamps via ``ensure_timestamp``.
        If ``None``, no timestamps are added.
    return_full : bool
        If ``True``, includes additional outputs such as raw samples,
        full DataFrame, metadata, encoding, and cleaned text.

    Returns
    -------
    out : dict of str to pandas.DataFrame
        Dictionary containing:
        - ``"eye1"``: left eye data
        - ``"eye2"``: right eye data  
        Optionally includes:
        - ``"samples"``: filtered sample rows
        - ``"raw"``: full parsed DataFrame
        - ``"meta"``: metadata dictionary
        - ``"encoding"``: detected file encoding
        - ``"text"``: cleaned text buffer
    data_type : str
        Constant string ``"PUPIL"``.

    Raises
    ------
    KeyError
        Raised if required columns (e.g., ``"Type"`` or eye measurement
        columns) are missing from the input data.

    Notes
    -----
    - Converts relevant columns to numeric using ``errors="coerce"``.
    - Filters rows where ``"Type"`` equals ``"SMP"``.
    - Pupil size is assumed to already be in millimeters.
    - Timestamp generation modifies DataFrames returned by
      ``ensure_timestamp``.
    """

    # read the file | try to be agnostic for encoding
    df, meta, text, enc, lines, header_line_index = simple_smi_read(raw_file)

    logger.debug("Constructing dataframe")

    needed = [
        "L Mapped Diameter [mm]", "R Mapped Diameter [mm]",
        "L POR X [px]", "L POR Y [px]",
        "R POR X [px]", "R POR Y [px]",
    ]

    for c in needed + ["Time"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    samples = df[df["Type"].astype(str).eq("SMP")].copy()

    df_left = pd.DataFrame({
        "x_coordinate": pd.to_numeric(samples["L POR X [px]"], errors="coerce"),
        "y_coordinate": pd.to_numeric(samples["L POR Y [px]"], errors="coerce"),
        "pupil_size": pd.to_numeric(samples["L Mapped Diameter [mm]"], errors="coerce"),
    })

    df_right = pd.DataFrame({
        "x_coordinate": pd.to_numeric(samples["R POR X [px]"], errors="coerce"),
        "y_coordinate": pd.to_numeric(samples["R POR Y [px]"], errors="coerce"),
        "pupil_size": pd.to_numeric(samples["R Mapped Diameter [mm]"], errors="coerce"),
    })

    logger.debug(f"Conversion complete")
    tmp = {
        "eye1": df_left,
        "eye2": df_right,
    }

    out = {}
    if sr is not None:
        # add timestamps if SamplingRate was specified
        logger.debug(f"Adding timestamp based on SamplingFrequency={sr}")
        for key, val in tmp.items():
            out[key], _ = ensure_timestamp(
                val,
                fs=sr,
                force=bool
            )
    else:
        out = deepcopy(tmp)

    if return_full:
        out["samples"] = samples
        out["raw"] = df
        out["meta"] = meta
        out["encoding"] = enc
        out["text"] = text

    # pupil is already mm
    return out, "PUPIL"


def get_eyetracker_setup_info(
        raw_file_path: str,
        eye_name: str,
        eye_json_template: Dict[str, Any],
        lab_name: Optional[str]=None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Fill eyetracker setup metadata from an SMI/iView exported TXT file.

    Creates a deep-copied eyetracker metadata dictionary from
    ``eye_json_template``, populates fields using lab-specific metadata and
    parsed SMI header values, and returns the updated eyetracker info together
    with stimulus presentation metadata.

    Parameters
    ----------
    raw_file_path : str
        Path to the SMI/iView exported TXT file containing ``"##"`` header
        lines.
    eye_name : str
        Eye label to record, such as ``"left"`` or ``"right"``. The stripped,
        lowercased value is written to ``"RecordedEye"``.
    eye_json_template : dict of str to Any
        Template dictionary used to initialize the eyetracker metadata. This
        object is deep-copied before modification.
    lab_name : str, optional
        Lab identifier used to look up lab-specific static metadata in
        ``df_meta``. If ``None``, lookups may fail depending on the contents
        of ``df_meta``.

    Returns
    -------
    info : dict of str to Any
        Updated eyetracker metadata dictionary populated with fields such as
        ``"RecordedEye"``, ``"Manufacturer"``,
        ``"ManufacturersModelName"``, ``"SoftwareVersion"``,
        ``"SamplingFrequency"``, ``"GazeRange"``, ``"CalibrationType"``, and
        ``"MeasurementType"``.
    stim_pres : dict of str to Any
        Stimulus presentation metadata returned by
        ``stimulus_presentation_from_metadata(df_meta, lab_name)`` and updated
        in place with values such as ``"ScreenSize"`` and
        ``"ScreenDistance"``.

    Raises
    ------
    KeyError
        Raised if required rows or keys are missing in ``df_meta``,
        ``eye_json_template``, or ``stim_pres``.
    IndexError
        Raised if a numeric field is expected in parsed SMI metadata but no
        matching value is found before indexing.
    AttributeError
        Raised if ``eye_name`` does not support ``strip()``.
    TypeError
        Raised if input objects do not support the dictionary-style access
        used by the function.

    Notes
    -----
    The function depends on external objects and helpers including ``df_meta``,
    ``parse_smi_header_kv``, ``stimulus_presentation_from_metadata``,
    ``copy``, ``re``, and ``logger``.

    The function mutates the deep-copied ``info`` dictionary and the
    ``stim_pres`` dictionary returned by
    ``stimulus_presentation_from_metadata``.

    All units for ``"x_coordinate"``, ``"y_coordinate"``, and
    ``"pupil_size"`` are set to ``"mm"``.

    If the screen dimensions from ``"Stimulus Dimension [mm]"`` do not match
    the submitted metadata, an error is logged and the screen size from the
    metadata file is retained.
    """

    eye = eye_name.strip()
    eye_lower = eye.lower()

    info = copy.deepcopy(eye_json_template)
    info["RecordedEye"] = eye_lower

    # lab-specific static metadata (same idea as your ASC version)
    meta_idx = df_meta.set_index("Parameter")

    info["Manufacturer"] = meta_idx.at["Eyetracker Manufacturer", lab_name]
    info["ManufacturersModelName"] = meta_idx.at["Eyetracker Manufacturer Model", lab_name]
    info["SoftwareVersion"] = meta_idx.at["Eyetracker Software Version", lab_name]

    stim_pres = stimulus_presentation_from_metadata(df_meta, lab_name)


    # parse SMI header
    smi = parse_smi_header_kv(raw_file_path)

    # SamplingFrequency
    # e.g. "250"
    if "Sample Rate" in smi:
        try:
            info["SamplingFrequency"] = int(re.findall(r"\d+", smi["Sample Rate"])[0])
        except Exception:
            pass

    # GazeRange from "Calibration Area" (pixel dims)
    # e.g. "1920\t1080"

    # Screen / stimulus size in mm
    if "Stimulus Dimension [mm]" in smi:
        nums = re.findall(r"-?\d+\.?\d*", smi["Stimulus Dimension [mm]"])
        if len(nums) >= 2:
            screen_size_txt = [float(nums[0]), float(nums[1])]
            
            # cross-reference with metadata.csv
            screen_size_meta = stim_pres["StimulusPresentation"]["ScreenSize"]

            if screen_size_meta != screen_size_txt:
                logger.error(f"Mismatch in screen dimensions between file {screen_size_txt} and submitted metadata {screen_size_meta}. Using screen size from metadata-file")

    w, h = screen_size_meta
    stim_pres["StimulusPresentation"]["ScreenSize"] = [w, h]  # mm (width, height)

    info.setdefault("GazeRange", {})
    info["GazeRange"]["xmin"] = 0
    info["GazeRange"]["ymin"] = 0
    info["GazeRange"]["xmax"] = w
    info["GazeRange"]["ymax"] = h

    # Eye camera distance (Head Distance [mm]) if your schema has something for it
    # (BIDS eyetrack json often uses "EyeCameraDistance" or similar in custom fields)
    if "Head Distance [mm]" in smi:
        nums = re.findall(r"-?\d+", smi["Head Distance [mm]"])
        if nums:
            try:
                stim_pres["StimulusPresentation"]["ScreenDistance"] = float(nums[0])  # mm
            except Exception:
                pass

    # CalibrationType: infer from number of calibration points if present
    # You have "Calibration Point 0".."8" => 9 points => HV9 (common label)
    cal_pts = [k for k in smi.keys() if k.startswith("Calibration Point ")]
    if cal_pts:
        n = len(cal_pts)
        # Only set if template field exists or you want it always
        if n == 9:
            info["CalibrationType"] = "HV9"
        else:
            info["CalibrationType"] = f"{n}-point"

    # MeasurementType from Format / DIAMETER
    # Your header shows DIAMETER, which implies pupil is already diameter.
    fmt = smi.get("Format", "")
    if fmt:
        # common values: DIAMETER or AREA
        if "DIAMETER" in fmt.upper():
            info["MeasurementType"] = "diameter"
        elif "AREA" in fmt.upper():
            info["MeasurementType"] = "area"

    # Device / software info from header, if you want to add or overwrite
    # (useful if df_meta is missing)
    if "System ID" in smi and not info.get("ManufacturersModelName"):
        info["ManufacturersModelName"] = smi["System ID"]

    if "iView X Version" in smi and not info.get("SoftwareVersion"):
        info["SoftwareVersion"] = smi["iView X Version"]

    # Calibration error fields usually not present in SMI header -> keep empty if in template
    info.setdefault("AverageCalibrationError", None)
    info.setdefault("MaximalCalibrationError", None)

    # set all units to mm
    for col in ["x_coordinate", "y_coordinate", "pupil_size"]:
        info[col]["Units"] = "mm"

    return info, stim_pres


def fetch_msg_events_from_txt(
        raw_file: str,
        timestamps: Optional[pd.Series]=None,
        sr: Optional[Union[float, int]]=None
    ) -> pd.DataFrame:
    """
    Extract message events from a raw SMI TXT file and align them to sample time.

    Reads raw SMI text, extracts ``"MSG"`` rows directly from the decoded
    lines, and interpolates their raw ``"Time"`` values onto the provided
    sample timestamps in seconds using ``"SMP"`` rows.

    Parameters
    ----------
    raw_file : str
        Path to the raw SMI TXT file.
    timestamps : pandas.Series, optional
        Sample timestamps in seconds corresponding to the ``"SMP"`` rows in
        the file. The number of values must match the number of extracted
        ``"SMP"`` rows.
    sr : float or int, optional
        Sampling frequency. This argument is accepted by the function
        signature but is not used by the current implementation.

    Returns
    -------
    events : pandas.DataFrame
        DataFrame with columns ``"onset"``, ``"duration"``, ``"trial_type"``,
        ``"blink"``, and ``"message"``. The ``"onset"`` column contains
        interpolated message times in seconds. The ``"message"`` column
        contains extracted message text, with missing values replaced by
        ``"n/a"``.

    Raises
    ------
    ValueError
        Raised if the number of provided ``timestamps`` does not match the
        number of ``"SMP"`` rows in the file.

    Notes
    -----
    - If no ``"SMP"`` rows are found, an empty events DataFrame is returned.
    - If no ``"MSG"`` rows are found, an empty events DataFrame is returned.
    - If ``"MSG"`` rows exist but none have valid numeric ``"Time"`` values,
      an empty events DataFrame is returned.
    - If no valid sample times are available for interpolation, ``"onset"``
      values are returned as ``NaN``.
    - Message alignment is performed with ``np.interp`` after sorting and
      deduplicating sample raw times.
    """

    df_all, meta, text_for_table, enc, lines, header_line_index = simple_smi_read(raw_file)

    df_samples = df_all[df_all["Type"] == "SMP"].copy()
    if df_samples.empty:
        logger.warning("No SMP rows found in raw file.")
        return pd.DataFrame(columns=["onset", "duration", "trial_type", "blink", "message"])

    df_samples["Time"] = pd.to_numeric(df_samples["Time"], errors="coerce")

    ts = pd.to_numeric(timestamps, errors="coerce").reset_index(drop=True)
    if len(df_samples) != len(ts):
        raise ValueError("Number of timestamps does not match number of SMP rows.")

    smp_raw_time_arr = df_samples["Time"].to_numpy(dtype=float)
    smp_sec_time_arr = ts.to_numpy(dtype=float)

    # read the messages from 'lines'
    df_msg_raw = _extract_msg_rows_from_decoded_lines(lines, header_line_index)

    # safeguard: no messages at all
    if df_msg_raw.empty:
        logger.warning("No MSG rows found in raw file.")
        return pd.DataFrame(columns=["onset", "duration", "trial_type", "blink", "message"])

    # safeguard: no valid message timestamps
    df_msg_raw = df_msg_raw[df_msg_raw["Time"].notna()].copy()
    if df_msg_raw.empty:
        logger.warning("MSG rows found, but none had valid timestamps.")
        return pd.DataFrame(columns=["onset", "duration", "trial_type", "blink", "message"])

    msg_times_arr = df_msg_raw["Time"].to_numpy(dtype=float)
    msg_texts = df_msg_raw["message"].tolist()

    valid = np.isfinite(smp_raw_time_arr) & np.isfinite(smp_sec_time_arr)

    if valid.sum() == 0:
        logger.warning(
            "No valid sample times found for message interpolation; returning NaN onsets."
        )
        msg_onsets = np.full(len(msg_times_arr), np.nan, dtype=float)
    else:
        x = smp_raw_time_arr[valid]
        y = smp_sec_time_arr[valid]

        order = np.argsort(x)
        x = x[order]
        y = y[order]

        x_unique, unique_idx = np.unique(x, return_index=True)
        y_unique = y[unique_idx]

        if len(x_unique) == 0:
            logger.warning(
                "No unique valid sample times found for message interpolation; returning NaN onsets."
            )
            msg_onsets = np.full(len(msg_times_arr), np.nan, dtype=float)
        else:
            msg_onsets = np.interp(
                msg_times_arr,
                x_unique,
                y_unique,
                left=np.nan,
                right=np.nan,
            )

    return pd.DataFrame({
        "onset": msg_onsets,
        "duration": "n/a",
        "trial_type": "n/a",
        "blink": "n/a",
        "message": pd.Series(msg_texts, dtype="string").fillna("n/a"),
    })


def fetch_physioevents(
        df: Optional[pd.DataFrame]=None,
        raw_file: Optional[str]=None,
        **kwargs: Any
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Create physioevents from an eye DataFrame and optional raw SMI message file.

    Builds fixation, saccade, and blink events from ``df`` and, when
    ``raw_file`` is provided, also extracts and aligns ``"MSG"`` rows from
    the raw SMI TXT file. The resulting event tables are merged, filtered,
    and sorted by ``"onset"``.

    Parameters
    ----------
    df : pandas.DataFrame, optional
        Eye-tracking DataFrame in millimeters. Expected to contain
        ``"timestamp"``, ``"x_coordinate"``, ``"y_coordinate"``, and
        ``"pupil_size"`` columns for physioevent generation.
    raw_file : str, optional
        Path to the raw SMI TXT file. Used only to extract and align
        ``"MSG"`` rows into event records.
    **kwargs : Any
        Additional keyword arguments passed to
        ``fetch_physioevents_from_df``.

    Returns
    -------
    events : pandas.DataFrame
        DataFrame with columns ``"onset"``, ``"duration"``, ``"trial_type"``,
        ``"blink"``, and ``"message"``. Returns an empty DataFrame with these
        columns if no valid events are available.
    settings : dict of str to Any
        Settings dictionary returned by ``fetch_physioevents_from_df``, with
        an additional ``"RawFileUsedForMessages"`` flag indicating whether
        ``raw_file`` was provided.

    Raises
    ------
    ValueError
        Propagated from downstream helpers if required columns are missing or
        if message timestamps cannot be aligned as expected.

    Notes
    -----
    - Physioevents are generated from ``df`` via
      ``fetch_physioevents_from_df``.
    - Message events are generated from ``raw_file`` via
      ``fetch_msg_events_from_txt`` using ``df["timestamp"]``.
    - Message rows with missing ``"onset"``, blank ``"message"``, or
      ``"message"`` equal to ``"n/a"`` are removed.
    - The function normalizes missing event fields to ``"n/a"`` for
      ``"duration"``, ``"trial_type"``, ``"blink"``, and ``"message"``.
    - The returned table is sorted by ``"onset"`` and rows with invalid
      ``"onset"`` are dropped.
    """

    df_model, settings = fetch_physioevents_from_df(
        df=df,
        **kwargs
    )

    empty_events = pd.DataFrame(
        columns=["onset", "duration", "trial_type", "blink", "message"]
    )

    if raw_file is not None:
        logger.info(f"Loading MSGs from {raw_file}")
        df_msg = fetch_msg_events_from_txt(
            raw_file=raw_file,
            timestamps=df["timestamp"],
        )

        # safeguard: normalize empty/invalid message tables
        if df_msg is None or df_msg.empty:
            logger.error("No valid MSG events found; this means event-markers are not propagated to eye-tracking files. We cannot know when an event happened now!")
            df_msg = empty_events.copy()
        else:
            df_msg = df_msg.copy()
            if "onset" in df_msg.columns:
                df_msg["onset"] = pd.to_numeric(df_msg["onset"], errors="coerce")

            # drop rows with no usable onset or no usable message
            df_msg = df_msg[
                df_msg["onset"].notna() &
                df_msg["message"].notna() &
                (df_msg["message"].astype(str).str.strip() != "") &
                (df_msg["message"].astype(str).str.strip().str.lower() != "n/a")
            ].copy()

            if df_msg.empty:
                logger.error("MSG table was empty after filtering blank messages; this means event-markers are not propagated to eye-tracking files. We cannot know when an event happened now!")
                df_msg = empty_events.copy()
    else:
        df_msg = empty_events.copy()

    frames = [x for x in (df_msg, df_model) if x is not None and not x.empty]

    if not frames:
        settings["RawFileUsedForMessages"] = raw_file is not None
        return empty_events.copy(), settings

    events = pd.concat(frames, ignore_index=True)
    events = events[["onset", "duration", "trial_type", "blink", "message"]].copy()

    events["onset"] = pd.to_numeric(events["onset"], errors="coerce")
    events = events[events["onset"].notna()].copy()

    if events.empty:
        settings["RawFileUsedForMessages"] = raw_file is not None
        return empty_events.copy(), settings

    events = events.sort_values("onset").reset_index(drop=True)

    for c in ["duration", "trial_type", "blink", "message"]:
        if c not in events.columns:
            events[c] = "n/a"
        else:
            events[c] = events[c].astype("string").fillna("n/a")

    settings["RawFileUsedForMessages"] = raw_file is not None

    return events, settings

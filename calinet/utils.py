# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import re
import os
import glob
import shutil
import importlib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict
from datetime import datetime

from typing import Optional, List, Dict, Any, Union

import logging
logger = logging.getLogger(__name__)


TS_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\]")
SUBJECT_RE = re.compile(
    r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}\]\s+\[(.*?)\]\s+\[[A-Z]+\]\s+"
)


def find_events_file_xlsx(
        raw_path: str,
        file_key: str
    ) -> Optional[str]:
    """
    Recursively search for an events Excel file matching a key and return the first match.

    This function walks the directory tree under ``raw_path`` and identifies the first
    ``.xlsx`` file whose filename contains both the provided ``file_key`` and the
    substring ``"events"`` (case-insensitive). The search stops at the first match.

    Parameters
    ----------
    raw_path : str
        Root directory to recursively search for candidate Excel files.
    file_key : str
        Substring used to match relevant Excel filenames (case-insensitive).

    Returns
    -------
    events_xlsx_file : str or None
        Path to the first matching Excel file. Returns ``None`` if no matching file
        is found.

    Notes
    -----
    This implementation differs from the Bonn pipeline, where the events file is
    typically stored at a predefined location and does not require recursive
    directory traversal or filename-based filtering.

    Only the first matching file is returned, and no validation of file contents
    is performed.
    """
    events_xlsx_file = None
    for root, _, files in os.walk(raw_path):
        for filename in files:
            if (
                filename.lower().endswith(".xlsx")
                and file_key in filename.lower()
                and "events" in filename.lower()
            ):
                events_xlsx_file = os.path.join(root, filename)
                break

    return events_xlsx_file


def find_events_file_csv(
        raw_path: str,
        file_key: str
    ) -> Optional[str]:
    """
    Recursively search for an events CSV file matching a key and return the largest match.

    This function walks the directory tree under ``raw_path`` and identifies all
    ``.csv`` files whose filenames contain the provided ``file_key`` (case-insensitive).
    If multiple matches are found, the largest file (by size on disk) is selected.

    Parameters
    ----------
    raw_path : str
        Root directory to recursively search for candidate CSV files.
    file_key : str
        Substring used to match relevant CSV filenames (case-insensitive).

    Returns
    -------
    events_csv_file : str or None
        Path to the selected CSV file. Returns ``None`` if no matching files are found.

    Notes
    -----
    This implementation differs from the Bonn pipeline, where the events file
    location is typically deterministic and does not require recursive search
    or size-based selection.

    File selection is based on ``os.path.getsize`` to prioritize the most complete
    or populated CSV when multiple candidates exist.

    Side effects include logging via ``logger.warning`` when no matches are found
    and ``logger.info`` when matches are identified and selected.
    """
    matches = []

    for root, _, files in os.walk(raw_path):
        for filename in files:
            if filename.lower().endswith(".csv") and file_key in filename.lower():
                full_path = os.path.join(root, filename)
                matches.append(full_path)

    if not matches:
        logger.warning(f"No CSV files containing '{file_key}' found in {raw_path}")
        return None

    logger.info(f"Found {len(matches)} matching CSV file(s)")

    # if more matches exist, take largest file
    if len(matches)>1:
        events_csv_file = max(matches, key=os.path.getsize)
        size_mb = os.path.getsize(events_csv_file) / (1024 * 1024)
        logger.info(f"Selected largest CSV file of {size_mb:.2f} MB")

    events_csv_file = max(matches, key=os.path.getsize)
    return events_csv_file


def common_write_tsv(
        subset: pd.DataFrame,
        id_key: str,
        language: Optional[str]=None,
        phenotype_dir: Optional[str]=None
    ) -> pd.DataFrame:
    """
    Convert value columns to appropriate numeric types and optionally write a TSV file.

    This function attempts to coerce non-ID columns in a dataframe to numeric
    types where possible. Integer-like columns are converted to pandas nullable
    integer type (``Int64``), while other numeric columns remain as floats.
    Optionally, the processed dataframe is written to a TSV file.

    Parameters
    ----------
    subset : pd.DataFrame
        Dataframe containing participant data, including a ``participant_id`` column
        and one or more value columns.
    id_key : str
        Prefix used for naming the output TSV file.
    language : str, optional
        Language suffix appended to the output filename. If ``None``, no suffix is added.
    phenotype_dir : str, optional
        Directory where the TSV file will be saved. If ``None``, no file is written.

    Returns
    -------
    subset : pd.DataFrame
        Dataframe with value columns converted to appropriate numeric types.

    Notes
    -----
    - Columns are only converted to numeric if at least one valid numeric value exists.
    - Integer-like columns are stored as pandas nullable integers (``Int64``) to
      preserve missing values.
    - Missing values are written as ``"n/a"`` in the TSV file.

    Examples
    --------
    >>> df_clean = common_write_tsv(
    ...     df_subset,
    ...     id_key="bfi_",
    ...     language="en",
    ...     phenotype_dir="/data/phenotypes"
    ... )
    >>> df_clean.dtypes
    """

    subset = subset.copy()

    value_cols = [c for c in subset.columns if c != "participant_id"]

    for col in value_cols:
        numeric = pd.to_numeric(subset[col], errors="coerce")

        # only treat as numeric if at least one real numeric value exists
        if numeric.notna().any():
            non_na = numeric.dropna()

            if (non_na == non_na.round()).all():
                subset[col] = numeric.astype("Int64")
            else:
                subset[col] = numeric
        # else: leave original column untouched

    if phenotype_dir is not None:
        os.makedirs(phenotype_dir, exist_ok=True)
        suffix = "" if language is None else language
        file_path = os.path.join(phenotype_dir, f"{id_key}{suffix}.tsv")
        subset.to_csv(file_path, sep="\t", index=False, na_rep="n/a")
        logger.info(f"Aggregated {id_key.upper()} data saved to: {file_path}")

    return subset


def rename_col(
        col: str,
        old_key: Optional[str]=None,
        new_key: Optional[str]=None,
        strip_leading_zeros: bool=True,
        zero_pattern: str=r'_(0+)(\d+)\b'
    ) -> str:
    """
    Rename a column string by replacing substrings and normalizing numeric suffixes.

    This function optionally replaces a substring (`old_key`) with another
    (`new_key`) and removes leading zeros from numeric suffixes using a regular
    expression pattern. A debug message is logged if the column name changes.

    Parameters
    ----------
    col : str
        Original column name.
    old_key : str, optional
        Substring to be replaced in the column name.
    new_key : str, optional
        Replacement substring for `old_key`.
    strip_leading_zeros : bool, default=True
        Whether to remove leading zeros from numeric suffixes.
    zero_pattern : str, default=r'_(0+)(\d+)\\b'
        Regular expression pattern used to identify leading zeros in suffixes.

    Returns
    -------
    col : str
        Transformed column name.

    Notes
    -----
    - Leading zeros are removed from patterns like ``_01`` → ``_1``.
    - Replacement occurs only if both `old_key` and `new_key` are provided
      and `old_key` is found in the column name.
    - Logging occurs only when the column name is modified.

    Examples
    --------
    >>> rename_col("bfi_01", strip_leading_zeros=True)
    'bfi_1'
    >>> rename_col("old_02", old_key="old", new_key="new")
    'new_2'
    """

    original = col

    # replace key
    if old_key and new_key and old_key in col:
        col = col.replace(old_key, new_key)

    # remove leading zeros
    if strip_leading_zeros:
        col = re.sub(zero_pattern, r'_\2', col)

    # log only if something changed
    if col != original:
        logger.debug(f"Renamed column: '{original}' → '{col}'")

    return col

    
def convert_questionnaire_columns_to_int(
        pheno_df: pd.DataFrame,
        quest_cols: Union[List[str], str],
        id_col: str="participant_id",
    ) -> pd.DataFrame:
    """
    Extract questionnaire columns and convert them to nullable integer dtype.

    This function selects questionnaire-related columns from a dataframe based
    on either an explicit list of column names or a string pattern. The selected
    columns are coerced to numeric values and converted to pandas nullable
    integer type (``Int64``). Non-numeric values are converted to missing
    values (``<NA>``).

    Parameters
    ----------
    pheno_df : pd.DataFrame
        Input dataframe containing participant data and questionnaire columns.
    quest_cols : list of str or str
        Either a list of column names to extract or a string pattern used with
        ``str.contains`` to match column names.
    id_col : str, default="participant_id"
        Column name identifying participants. This column is always retained.

    Returns
    -------
    df : pd.DataFrame
        Dataframe containing the participant ID column and selected questionnaire
        columns converted to ``Int64`` dtype.

    Raises
    ------
    ValueError
        If no columns match the provided `quest_cols` specification.
    ValueError
        If all values in the selected questionnaire columns are missing after
        conversion.

    Notes
    -----
    - When `quest_cols` is a string, column selection is performed using
      ``pheno_df.columns.str.contains``.
    - Non-numeric values are coerced to ``<NA>`` during conversion.
    - All selected columns are cast to pandas nullable integer type (``Int64``).

    Examples
    --------
    >>> df_int = convert_questionnaire_columns_to_int(
    ...     pheno_df,
    ...     quest_cols="bfi_"
    ... )
    >>> df_int.dtypes
    """

    if isinstance(quest_cols, str):
        cols = pheno_df.columns[pheno_df.columns.str.contains(quest_cols)].tolist()
    else:
        cols = [c for c in quest_cols if c in pheno_df.columns]

    if not cols:
        raise ValueError(f"No questionnaire columns matched pattern: {quest_cols}")

    df = pheno_df[[id_col] + cols].copy()

    df[cols] = (
        df[cols]
        .apply(pd.to_numeric, errors="coerce")
        .astype("Int64")
    )

    if df[cols].isna().all().all():
        raise ValueError(f"All questionnaire values missing for columns: {cols}")

    return df


def extract_subject(
        line: str
    ) -> str:
    """
    Extract subject identifier from a log line.

    Parameters
    ----------
    line : str
        Log line containing subject information in bracketed format.

    Returns
    -------
    subject : str
        Extracted subject identifier, or ``"-"`` if no match is found.

    Notes
    -----
    This function uses a regular expression to parse subject identifiers and
    performs no I/O or side effects.
    """
    m = SUBJECT_RE.match(line)
    return m.group(1) if m else "-"


def parse_ts(
        line: str
    ) -> Optional[datetime]:
    """
    Parse timestamp from a log line.

    Parameters
    ----------
    line : str
        Log line containing a timestamp in bracketed format.

    Returns
    -------
    ts : datetime.datetime or None
        Parsed timestamp, or ``None`` if no valid timestamp is found.

    Notes
    -----
    This function uses a regular expression and ``datetime.strptime`` for
    parsing and does not perform I/O.
    """
    m = TS_RE.match(line)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f")


def merge_log_files(
        main_log: Optional[str],
        worker_merged_log: Optional[str],
        output_log: str
    ) -> None:
    """
    Merge and group log files by subject and timestamp.

    Parameters
    ----------
    main_log : str or None
        Path to the main log file.
    worker_merged_log : str or None
        Path to the merged worker log file.
    output_log : str
        Path to the output log file.

    Returns
    -------
    None

    Notes
    -----
    Log lines are parsed, sorted by subject and timestamp, and written to
    ``output_log`` grouped by subject.

    This function performs file I/O and writes a new log file.
    """

    all_lines = []

    for src in [main_log, worker_merged_log]:
        if src is None or not os.path.exists(src):
            continue

        with open(src, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                ts = parse_ts(line)
                if ts is None:
                    continue

                subj = extract_subject(line)
                if subj == "-":
                    subj = "GLOBAL"

                all_lines.append((subj, ts, os.path.basename(src), idx, line))

    all_lines.sort(key=lambda x: (x[0], x[1], x[2], x[3]))

    with open(output_log, "w", encoding="utf-8") as out:
        current_subject = None

        for subj, _, _, _, line in all_lines:
            if subj != current_subject:
                current_subject = subj
                out.write("\n" + "=" * 80 + "\n")
                out.write(f"SUBJECT: {subj}\n")
                out.write("=" * 80 + "\n")

            out.write(line)


def merge_worker_logs(
        output_dir: str,
        main_log_file: str,
        remove_worker_logs: bool=False
    ) -> None:
    """
    Merge worker log files into a main log file.

    Parameters
    ----------
    output_dir : str
        Directory containing worker log files.
    main_log_file : str
        Path to the main log file to append merged content.
    remove_worker_logs : bool, default=False
        Whether to delete worker log files after merging.

    Returns
    -------
    None

    Notes
    -----
    Worker logs matching ``"log.worker.*.log"`` are read, sorted, and appended
    to ``main_log_file``.

    This function performs file I/O and may delete files when
    ``remove_worker_logs=True``.
    """

    worker_logs = glob.glob(os.path.join(output_dir, "log.worker.*.log"))
    all_lines = []

    for wf in worker_logs:
        with open(wf, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                ts = parse_ts(line)
                if ts is None:
                    continue

                subj = extract_subject(line)

                all_lines.append((subj, ts, os.path.basename(wf), line_idx, line))

    # chronological first, subject second
    all_lines.sort(key=lambda x: (x[0], x[1], x[2], x[3]))

    with open(main_log_file, "a", encoding="utf-8") as out:
        out.write("\n" + "=" * 80 + "\n")
        out.write("WORKER LOGS (CHRONOLOGICAL + SUBJECT)\n")
        out.write("=" * 80 + "\n")

        for _, _, _, _, line in all_lines:
            out.write(line)

    if remove_worker_logs:
        for wf in worker_logs:
            os.remove(wf)


def cleanup_logs(
        output_dir: str,
        keep_main: bool=False
    ) -> None:
    """
    Clean up temporary and merged log files.

    Parameters
    ----------
    output_dir : str
        Directory containing log files.
    keep_main : bool, default=False
        Whether to keep the main temporary log file.

    Returns
    -------
    None

    Notes
    -----
    This function closes all active file handlers, removes temporary log files,
    and optionally deletes the main log file.

    This function performs file I/O and modifies global logging handlers.
    """

    main_log = os.path.join(output_dir, "log_tmp.log")
    worker_log = os.path.join(output_dir, "log_merged.log")

    root = logging.getLogger()

    # Close all file handlers first
    for h in root.handlers[:]:
        if isinstance(h, logging.FileHandler):
            h.flush()
            h.close()
            root.removeHandler(h)

    if not keep_main and os.path.exists(main_log):
        os.remove(main_log)

    if os.path.exists(worker_log):
        os.remove(worker_log)

        
def find_available_modalities(
        physio_dir: str,
        subject: Optional[str]=None,
        task_name: Optional[str]=None
    ) -> List[str]:
    """
    Find available recording modalities in a physiology directory.

    Parameters
    ----------
    physio_dir : str
        Directory containing physiology files.
    subject : str or None, optional
        Optional subject filter (e.g., ``"sub-001"``).
    task_name : str or None, optional
        Optional task filter (e.g., ``"acquisition"``).

    Returns
    -------
    modalities : list of str
        Sorted list of detected modalities.

    Notes
    -----
    This function scans filenames using a regular expression and does not
    read file contents.

    This function performs filesystem listing but does not modify files.
    """

    modalities = set()

    pattern = re.compile(
        r'^(?P<subject>sub-[^_]+)_task-(?P<task>[^_]+)_recording-(?P<modality>[^_]+)_physio\.tsv\.gz$'
    )

    for fname in os.listdir(physio_dir):
        match = pattern.match(fname)
        if not match:
            continue

        if subject is not None and match.group("subject") != subject:
            continue

        if task_name is not None and match.group("task") != task_name:
            continue

        modalities.add(match.group("modality"))

    return sorted(modalities)


def _find_events_json(
        in_file: str
    ) -> str:
    """
    Locate the corresponding events JSON file for a physiology file.

    Parameters
    ----------
    in_file : str
        Path to a physiology file.

    Returns
    -------
    events_json : str
        Path to the corresponding events JSON file.

    Raises
    ------
    ValueError
        If the task stem cannot be extracted.
    FileNotFoundError
        If the events JSON file does not exist.

    Notes
    -----
    This function performs filesystem existence checks and string parsing.
    """

    p = Path(in_file)
    m = re.search(r"^(.*?_task-[^_./\\]+)", p.as_posix())
    if not m:
        raise ValueError("Could not find task stem in filename (expected ..._task-<name>...).")
    stem = m.group(1)
    events_json = stem + "_events.json"
    if not os.path.exists(events_json):
        raise FileNotFoundError(f"Event json does not exist: {events_json}")
    return events_json


def clean_output_directory(
        converted_dataset_dir: str,
        log_file: Optional[str]=None
    ) -> None:
    """
    Remove all files and directories in an output directory except a log file.

    Parameters
    ----------
    converted_dataset_dir : str
        Directory to clean.
    log_file : str or None, optional
        Path to a log file to preserve.

    Returns
    -------
    None

    Notes
    -----
    This function deletes files and directories using ``os.unlink`` and
    ``shutil.rmtree``.

    This function performs destructive filesystem operations.
    """
    
    for filename in os.listdir(converted_dataset_dir):
        file_path = os.path.join(converted_dataset_dir, filename)
        
        if filename != os.path.basename(log_file):
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                raise Exception(f"Failed to delete {file_path}. Reason: {e}") from e


def get_session_and_task_name(
        file_name: str
    ) -> Tuple[str, str]:
    """
    Determine session and task name from a filename.

    Parameters
    ----------
    file_name : str
        Input filename.

    Returns
    -------
    result : tuple of (str, str)
        Session name and task name.

    Notes
    -----
    This function performs string matching and has no side effects.
    """

    if "acquisition" in file_name.lower():
        session_name = "ses-01"
        task_name = "acquisition"
    else:
        session_name = "ses-02"
        task_name = "extinction"

    return (session_name, task_name)


def clean_input_subject_names(
        raw_data_dir: str
    ) -> None:
    """
    Normalize subject directory names by removing underscores.

    Parameters
    ----------
    raw_data_dir : str
        Root directory containing subject folders.

    Returns
    -------
    None

    Notes
    -----
    This function renames directories in place and logs operations.

    This function performs filesystem modifications.
    """

    sub_dirs = find_sub_dirs(raw_data_dir)

    # for ever sub_dir in sub_dirs, rename the dir remove any '_' that appears after 'sub-' in its name
    for sub_dir in sub_dirs:
        dir_name = os.path.basename(sub_dir)
        new_dir_name = dir_name[:4] + dir_name[4:].replace("_", "")
        new_dir_path = os.path.join(os.path.dirname(sub_dir), new_dir_name)
        os.rename(sub_dir, new_dir_path)
        logger.info(f"Renamed {sub_dir} to {new_dir_path}")
        # print(f"Renamed {dir_name} to {new_dir_name}")


def find_sub_dirs(
        raw_data_dir: str
    ) -> List[str]:
    """
    Find subject directories within a raw data directory.

    Parameters
    ----------
    raw_data_dir : str
        Root directory to search.

    Returns
    -------
    subject_dirs : list of str
        Filtered list of subject directory paths.

    Notes
    -----
    This function traverses the filesystem and filters directories based on
    naming rules.

    This function performs filesystem access but does not modify files.
    """

    subject_dirs = []

    for root, dirs, _ in os.walk(raw_data_dir):
        root_lower = root.lower()

        # Skip unwanted trees
        if any(x in root_lower for x in ["exclude", "shock", "processed"]):
            continue

        for d in dirs:
            if d.lower().startswith("sub-"):
                subject_dirs.append(Path(root) / d)

    # sort shorter/parent paths first
    subject_dirs = sorted(set(subject_dirs), key=lambda p: (len(p.parts), str(p).lower()))

    filtered = []
    kept = set()

    for p in subject_dirs:
        
        # skip if any ancestor is already kept
        if any(parent in kept for parent in p.parents):
            continue

        filtered.append(str(p))
        kept.add(p)

    return filtered


def extract_subject_id(
        input: str
    ) -> str:
    """
    Extract numeric subject identifier from a string.

    Parameters
    ----------
    input : str
        Input string containing a numeric suffix.

    Returns
    -------
    subject_id : str
        Extracted numeric identifier.

    Notes
    -----
    This function uses regular expressions and has no side effects.
    """

    num = re.search(r'\d+$', input).group()
    return num


def extract_subject_name(
        subject_folder_name: str
    ) -> str:
    """
    Construct standardized subject name from a folder name.

    Parameters
    ----------
    subject_folder_name : str
        Subject folder name.

    Returns
    -------
    subject_name : str
        Standardized subject name (e.g., ``"sub-001"``).

    Notes
    -----
    This function performs string manipulation only.
    """

    num = extract_subject_id(os.path.basename(subject_folder_name))
    subject_name = f"sub-{num}"
    return subject_name


def fetch_lab_module(
        lab_name_or_path: str
    ) -> Any:
    """
    Import a lab-specific module dynamically.

    Parameters
    ----------
    lab_name_or_path : str
        Lab name or path.

    Returns
    -------
    module : Any
        Imported module.

    Notes
    -----
    This function performs dynamic imports and string normalization.
    """

    from pathlib import PureWindowsPath
    from calinet.config import import_modules

    # If a path was passed, reduce to last component using Windows semantics
    lab = PureWindowsPath(lab_name_or_path).name

    # If it contains the backspace char, remove it (symptom of "\b")
    lab = lab.replace("\x08", "")

    return importlib.import_module(import_modules[lab])


def filter_non_printable(
        s: str
    ) -> str:
    """
    Remove non-printable characters from a string.

    Parameters
    ----------
    s : str
        Input string.

    Returns
    -------
    cleaned : str
        String containing only printable characters.

    Notes
    -----
    This function performs string filtering and has no side effects.
    """

    return "".join(c for c in s if c.isprintable())


def update_kwargs(
        kwargs: Dict[str, Any],
        el: str,
        val: Any,
        force: bool=False
    ) -> Dict[str, Any]:
    """
    Update a dictionary with a key-value pair conditionally.

    Parameters
    ----------
    kwargs : dict of str to Any
        Dictionary to update.
    el : str
        Key to update.
    val : Any
        Value to assign.
    force : bool, default=False
        Whether to overwrite existing values.

    Returns
    -------
    kwargs : dict of str to Any
        Updated dictionary.

    Notes
    -----
    This function modifies the input dictionary in place.
    """

    if not force:
        if el not in list(kwargs.keys()):
            kwargs[el] = val
    else:
        kwargs[el] = val

    return kwargs


def _normalize_question_text(
        qtext: str
    ) -> str:
    """
    Normalize question text into a simplified identifier.

    Parameters
    ----------
    qtext : str
        Input question text.

    Returns
    -------
    normalized : str
        Normalized string with lowercase alphanumeric characters and
        underscores.

    Notes
    -----
    This function performs string cleaning using regular expressions.
    """

    q_no_paren = re.sub(r"\s*\(.*?\)", "", qtext).strip()
    cleaned = re.sub(r"[^a-zA-Z0-9 ]+", "", q_no_paren).lower().strip()
    return re.sub(r"\s+", "_", cleaned)


def _read_file_lines(
        filepath: str
    ) -> List[str]:
    """
    Read lines from a file using multiple encodings.

    Parameters
    ----------
    filepath : str
        Path to the file.

    Returns
    -------
    lines : list of str
        File lines without trailing newline characters.

    Raises
    ------
    UnicodeDecodeError
        If the file cannot be decoded with supported encodings.

    Notes
    -----
    This function attempts multiple encodings and performs file I/O.
    """

    for enc in ("utf-8", "utf-8-sig", "utf-16", "latin-1"):
        try:
            with open(filepath, "r", encoding=enc) as f:
                return [line.rstrip("\n") for line in f]
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f"Unable to decode {filepath} with utf-8/utf-8-sig/utf-16/latin-1")


def map_handedness(
        x: Any
    ) -> Optional[str]:
    """
    Map handedness values to standardized labels.

    Parameters
    ----------
    x : Any
        Input value representing handedness.

    Returns
    -------
    handedness : str or None
        ``"left"``, ``"right"``, or ``None`` if not recognized.

    Notes
    -----
    This function performs string matching and has no side effects.
    """

    if pd.isna(x):
        return None
    
    x = str(x).lower()
    
    if "left" in x:
        return "left"
    if "right" in x:
        return "right"
    
    return None


def ensure_timestamp(
        df: pd.DataFrame,
        fs: float,
        force: bool
    ) -> Tuple[pd.DataFrame, bool]:
    """
    Ensure a valid timestamp column exists in a DataFrame.

    This function verifies whether the input DataFrame already contains a
    valid ``"timestamp"`` column as its first column. A timestamp column is
    considered valid if it is numeric, strictly increasing, and has a median
    step size consistent with the expected sampling interval ``1/fs``.

    If a valid timestamp column is present and ``force`` is False, the
    DataFrame is returned unchanged. Otherwise, a new timestamp column is
    generated based on the sampling frequency and inserted as the first
    column.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing time-series data.
    fs : float
        Sampling frequency in Hz used to generate or validate timestamps.
    force : bool
        If True, always overwrite or insert a new timestamp column regardless
        of whether a valid one already exists.

    Returns
    -------
    df_out : pandas.DataFrame
        DataFrame with a valid ``"timestamp"`` column as the first column.
    modified : bool
        Boolean flag indicating whether the DataFrame was modified (True) or
        returned unchanged (False).

    Raises
    ------
    ValueError
        Raised if ``fs`` is not a positive number.
    TypeError
        Raised if ``df`` is not a pandas DataFrame.

    Notes
    -----
    The generated timestamp starts at 0 and increases in steps of ``1/fs``
    for each row.

    Validation of an existing timestamp column checks for:
    - numeric values
    - strictly increasing sequence
    - approximately constant sampling interval

    The tolerance for interval validation is ``1e-6``.
    """
    
    n = len(df)
    ts = np.arange(n, dtype=float) / fs

    if not force and df.columns[0] == "timestamp":
        col0 = pd.to_numeric(df.iloc[:, 0], errors="coerce")
        if col0.notna().all():
            diffs = col0.diff().dropna()
            if (
                len(diffs) > 0
                and (diffs > 0).all()
                and abs(float(diffs.median()) - (1.0 / fs)) < 1e-6
            ):
                return df, False

    df2 = df.copy()
    df2.insert(0, "timestamp", ts)
    df2 = df2[["timestamp"] + [c for c in df2.columns if c != "timestamp"]]

    return df2, True


def creation_date(
        path: str,
        regex=r"\d{4}-\d{2}-\d{2}"
    ) -> pd.Timestamp:
    """
    Return the file date as a normalized pandas Timestamp.

    The timestamp is derived from the file's modification time using
    ``os.path.getmtime`` and converted to a pandas ``Timestamp``. The
    resulting datetime is normalized to midnight (00:00:00), keeping only
    the date component.

    Parameters
    ----------
    path : str
        Path to the file for which the timestamp should be retrieved.
    regex: byte-string
        Regex to use for extracting the date from file
    
    Returns
    -------
    creation_date : pandas.Timestamp
        Normalized timestamp representing the file's modification date.

    Raises
    ------
    FileNotFoundError
        Raised if the file at ``path`` does not exist.
    OSError
        Raised if the file metadata cannot be accessed.
    """

    match = re.search(regex, path)

    if match:
        return pd.to_datetime(match.group(0)).normalize()
    else:
        return pd.NaT


def fetch_creation_dates(
        raw_path: str,
    ) -> Dict[str, pd.Timestamp]:
    """
    Build a mapping from participant identifiers to physiology acquisition dates.

    For each participant listed in the input DataFrame, this function locates
    the corresponding raw physiology ``.acq`` file using a lab-specific
    discovery function and extracts its modification date. The resulting
    mapping links each ``participant_id`` to a normalized acquisition date.

    Parameters
    ----------
    raw_path : str
        Path to the root directory containing participant-specific raw data
        subdirectories.

    Returns
    -------
    creation_dates : dict of str to pandas.Timestamp
        Dictionary mapping each ``participant_id`` (e.g. ``"sub-001"``) to a
        normalized acquisition date derived from the corresponding ``.acq``
        file.
    """

    subject_dirs = find_sub_dirs(raw_path)

    # find lab-specific way to find physio-file

    creation_dates = {}
    for subject_dir in subject_dirs:
        subject_name = os.path.basename(subject_dir)          # e.g. 'sub-001'
        participant_id = f"sub-{extract_subject_id(subject_name)}"

        # find acq file
        ev_file = find_events_file_csv(
            raw_path=subject_dir,
            file_key="acquisition"
        )

        # find creation date
        date = creation_date(ev_file)
        logger.debug(f"Raw ID: {subject_name} | Converted ID: {participant_id} | File: {ev_file} | Timestamp: {date}")
        creation_dates[participant_id] = date

    return creation_dates


def append_acq_date_to_df(
        pheno_df: pd.DataFrame,
        raw_path: str,
    ) -> pd.DataFrame:
    """
    Append acquisition dates to a phenotype DataFrame.

    This function retrieves physiology acquisition dates for each participant
    using lab-specific file discovery logic and adds them as a new column
    ``"acq_date"`` to the input DataFrame. Acquisition dates are derived from
    the modification time of the corresponding ``.acq`` files.

    Parameters
    ----------
    pheno_df : pandas.DataFrame
        Phenotype DataFrame containing a ``"participant_id"`` column.
    raw_path : str
        Path to the root directory containing participant-specific raw data
        subdirectories.

    Returns
    -------
    pheno_df : pandas.DataFrame
        Input DataFrame with an added ``"acq_date"`` column containing
        normalized acquisition dates.

    Raises
    ------
    KeyError
        Raised if the ``"participant_id"`` column is missing in ``pheno_df``.
    FileNotFoundError
        Raised if a physiology ``.csv`` file cannot be located for a subject.
    OSError
        Raised if file metadata cannot be accessed.

    Notes
    -----
    Acquisition dates are obtained via ``fetch_creation_dates`` and mapped to
    participants using ``participant_id``.

    If acquisition dates are missing for any participants, a warning is
    logged and a fallback date of ``1900-01-01`` is assigned.

    This function modifies ``pheno_df`` in place and also returns it for
    convenience.
    """

    # fetch creation dates
    logger.info(f"Fetching acquisition date from PsychoPy-files for 'recorded_at'")
    creation_dates = fetch_creation_dates(raw_path)
    
    # add to dataframe
    pheno_df["acq_date"] = pheno_df["participant_id"].map(creation_dates)
    missing_mask = pheno_df["acq_date"].isna()
    if missing_mask.any():
        missing_ids = pheno_df.loc[missing_mask, "participant_id"].tolist()
        logger.warning(
            "Missing acquisition dates for participants: "
            + ", ".join(sorted(missing_ids))
        )

        # fallback date
        pheno_df.loc[missing_mask, "acq_date"] = pd.Timestamp("1900-01-01")

    return pheno_df

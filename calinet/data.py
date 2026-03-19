# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

from __future__ import annotations

import numbers
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Sequence, Union, Optional, Tuple, Dict, List

import numpy as np
import pandas as pd

from calinet.core import io as cio
import logging
logger = logging.getLogger(__name__)


@dataclass
class TrimResult:
    """
    Container for trimmed physiology data and associated metadata.

    Parameters
    ----------
    data : pandas.DataFrame
        Trimmed physiology data.
    events : pandas.DataFrame
        Event data aligned to the trimmed physiology recording.
    trimpoints : tuple of float
        Start and end trim points in seconds as ``(start, end)``.
    duration : float
        Duration of the trimmed segment in seconds.

    Notes
    -----
    This dataclass stores trimmed data in memory and does not read from or
    write to files.
    """

    data: pd.DataFrame
    events: pd.DataFrame
    trimpoints: Tuple[float, float]
    duration: float


def _is_none_token(
        x: Any
    ) -> bool:
    """
    Check whether input represents a string ``"none"`` token.

    Parameters
    ----------
    x : Any
        Input value to evaluate.

    Returns
    -------
    is_none : bool
        ``True`` if ``x`` is a string equal to ``"none"`` (case-insensitive),
        otherwise ``False``.

    Notes
    -----
    This function performs a type check and string comparison. It does not
    modify input or produce side effects.
    """
    return isinstance(x, str) and x.lower() == "none"


def _resolve_reference_event_indices(
        events_df: pd.DataFrame,
        reference: Any,
        event_time_col: str,
        event_name_col: Optional[str]=None,
        event_value_col: Optional[str]=None
    ) -> Tuple[bool, Optional[int], Optional[int]]:
    """
    Resolve reference specification into event index bounds.

    Parameters
    ----------
    events_df : pandas.DataFrame
        DataFrame containing event information. Must be sorted by
        ``event_time_col`` prior to calling this function.
    reference : Any
        Reference specification. Can be:

        * ``"file"``: use full file range (no event-based trimming)
        * ``"marker"``: use first and last event
        * sequence of two integers: direct event indices
        * sequence of two values: resolved via ``event_value_col`` or
          ``event_name_col``
    event_time_col : str
        Column name representing event timestamps.
    event_name_col : str or None, optional
        Column name containing event names used for string-based resolution.
    event_value_col : str or None, optional
        Column name containing numeric or categorical event values used for
        resolution.

    Returns
    -------
    getmarker : bool
        Whether trimming should be based on event markers.
    start_idx : int or None
        Zero-based start index into ``events_df`` after sorting.
    end_idx : int or None
        Zero-based end index into ``events_df`` after sorting.

    Raises
    ------
    ValueError
        Raised if reference specification is invalid or cannot be uniquely
        resolved.

    Notes
    -----
    This function performs validation and index resolution in memory and does
    not modify ``events_df`` or produce side effects.
    """

    if isinstance(reference, str):
        ref = reference.lower()
        if ref == "file":
            return False, None, None
        if ref == "marker":
            if events_df.empty:
                raise ValueError("Event reference requested but events dataframe is empty.")
            return True, 0, len(events_df) - 1
        raise ValueError("reference must be 'file', 'marker', a 2-element sequence of indices, or a 2-element sequence of names/values.")

    if isinstance(reference, Sequence) and not isinstance(reference, (str, bytes)):
        if len(reference) != 2:
            raise ValueError("reference sequence must have exactly two elements.")

        a, b = reference

        # numeric indices
        if isinstance(a, numbers.Number) and isinstance(b, numbers.Number):
            start_idx = int(a)
            end_idx = int(b)
            if start_idx < 0 or end_idx < start_idx or end_idx >= len(events_df):
                raise ValueError("Invalid event indices in reference.")
            return True, start_idx, end_idx

        # resolve by unique event value or unique event name
        def resolve_one(x: Any) -> int:
            # try numeric-like against value column first
            if event_value_col is not None:
                try:
                    x_num = float(x)
                    matches = events_df.index[events_df[event_value_col] == x_num].tolist()
                    if len(matches) == 1:
                        return matches[0]
                except (TypeError, ValueError):
                    pass

                matches = events_df.index[events_df[event_value_col] == x].tolist()
                if len(matches) == 1:
                    return matches[0]

            if event_name_col is not None and isinstance(x, str):
                matches = events_df.index[
                    events_df[event_name_col].astype(str).str.lower() == x.lower()
                ].tolist()
                if len(matches) == 1:
                    return matches[0]

            raise ValueError(f"Could not uniquely resolve reference event {x!r}.")

        start_label = resolve_one(a)
        end_label = resolve_one(b)

        # convert labels to positional indices after sort/reset_index below
        pos_lookup = {label: pos for pos, label in enumerate(events_df.index)}
        start_idx = pos_lookup[start_label]
        end_idx = pos_lookup[end_label]

        if end_idx < start_idx:
            raise ValueError("End reference event occurs before start reference event.")

        return True, start_idx, end_idx

    raise ValueError("Invalid reference specification.")


def _read_tsv_any(
        path: Path
    ) -> pd.DataFrame:
    """
    Read a tab-separated values file with optional gzip compression.

    Parameters
    ----------
    path : pathlib.Path
        Path to the input ``.tsv`` or ``.tsv.gz`` file.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame containing the parsed tabular data.

    Notes
    -----
    If ``path`` has suffix ``".gz"``, the file is read using
    ``cio.read_physio_tsv_headerless``. Otherwise, it is read using
    ``pandas.read_csv`` with tab separation.

    This function performs file I/O and logs the read operation via
    ``logger.debug``.
    """
    logger.debug(f"Reading table: {path}")
    if path.suffix == ".gz":
        return cio.read_physio_tsv_headerless(path)
    return pd.read_csv(path, sep="\t")


def _write_tsv_any(
        df: pd.DataFrame,
        path: Path
    ) -> None:
    """
    Write a DataFrame to a tab-separated values file with optional gzip compression.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to write.
    path : pathlib.Path
        Output file path. Supports ``.tsv`` and ``.tsv.gz``.

    Returns
    -------
    None

    Notes
    -----
    If ``path`` has suffix ``".gz"``, the DataFrame is written using
    ``cio.write_physio_tsv_gz_headerless``. Otherwise, it is written using
    ``pandas.DataFrame.to_csv`` with tab separation and ``index=False``.

    This function performs file I/O and logs the write operation via
    ``logger.debug``.
    """
    logger.debug(f"Writing table: {path}")
    if path.suffix == ".gz":
        cio.write_physio_tsv_gz_headerless(df, path)
    else:
        df.to_csv(path, sep="\t", index=False)


def _strip_double_suffix(
        path: Path
    ) -> str:
    """
    Remove ``.tsv.gz`` or ``.tsv`` suffixes from a filename.

    Parameters
    ----------
    path : pathlib.Path
        Input file path.

    Returns
    -------
    stem : str
        Filename with trailing ``.tsv.gz`` or ``.tsv`` removed. If neither
        suffix is present, ``path.stem`` is returned.

    Notes
    -----
    This function operates on the path name only and does not access the
    filesystem or produce side effects.
    """
    name = path.name
    if name.endswith(".tsv.gz"):
        return name[:-7]
    if name.endswith(".tsv"):
        return name[:-4]
    return path.stem


def _build_output_path(
        path: Path,
        prefix: str
    ) -> Path:
    """
    Build an output path by prefixing the filename.

    Parameters
    ----------
    path : pathlib.Path
        Original file path.
    prefix : str
        Prefix to prepend to ``path.name``.

    Returns
    -------
    output_path : pathlib.Path
        New path in the same directory with ``prefix`` prepended to the
        original filename.

    Notes
    -----
    This function constructs a new path object in memory and does not create
    directories or write files.
    """
    return path.with_name(prefix + path.name)


def _parse_bids_entities(
        name: str
    ) -> Dict[str, Any]:
    """
    Extract simple BIDS-like entities from a filename stem.

    Parameters
    ----------
    name : str
        Filename stem or basename split by underscores, such as
        ``"sub-001_task-rest_recording-scr_physio"``.

    Returns
    -------
    entities : dict of str to Any
        Mapping of parsed entity keys to values. Parts containing ``"-"``
        are split once into key and value. Parts without ``"-"`` are stored
        with value ``True``.

    Notes
    -----
    This function performs string parsing only and does not access the
    filesystem or produce side effects.
    """
    """
    Extract simple BIDS-like entities from a filename stem.
    """
    
    entities = {}
    for part in name.split("_"):
        if "-" in part:
            key, value = part.split("-", 1)
            entities[key] = value
        else:
            entities[part] = True
    return entities


def _find_task_events_file(
        physio_file: Path
    ) -> Optional[Path]:
    """
    Find the matching task-level events file for a physiology file.

    Parameters
    ----------
    physio_file : pathlib.Path
        Path to a physiology file such as
        ``sub-001_task-acquisition_recording-scr_physio.tsv.gz``.

    Returns
    -------
    task_events_file : pathlib.Path or None
        Path to the matching ``sub-<id>_task-<task>_events.tsv`` file in the
        same directory if it exists, otherwise ``None``.

    Notes
    -----
    This function parses ``sub`` and ``task`` entities from
    ``physio_file.name`` and checks for the existence of the candidate file in
    the same directory.

    This function performs filesystem existence checks and logs warnings via
    ``logger.warning`` when parsing fails or no matching file is found.
    """
    
    stem = _strip_double_suffix(physio_file)
    entities = _parse_bids_entities(stem)

    sub = entities.get("sub")
    task = entities.get("task")

    if sub is None or task is None:
        logger.warning(f"Could not parse sub/task from {physio_file.name}")
        return None

    candidate = physio_file.parent / f"sub-{sub}_task-{task}_events.tsv"
    if candidate.exists():
        return candidate

    logger.warning(f"No task events file found for {physio_file.name}")
    return None


def _find_physioevents_pair(
        physio_file: Path
    ) -> Optional[Path]:
    """
    Find the paired ``*_physioevents`` file for a physiology file.

    Parameters
    ----------
    physio_file : pathlib.Path
        Path to a physiology file such as
        ``sub-001_task-acquisition_recording-eye2_physio.tsv.gz`` or
        ``sub-001_task-acquisition_recording-eye2_physio.tsv``.

    Returns
    -------
    physioevents_file : pathlib.Path or None
        Path to the matching ``*_physioevents.tsv.gz`` or
        ``*_physioevents.tsv`` file in the same directory if it exists,
        otherwise ``None``.

    Notes
    -----
    This function performs filename substitution and filesystem existence
    checks. It does not read or write file contents.
    """

    name = physio_file.name
    candidate_name = name.replace("_physio.tsv.gz", "_physioevents.tsv.gz")
    candidate = physio_file.parent / candidate_name
    if candidate.exists():
        return candidate

    candidate_name = name.replace("_physio.tsv", "_physioevents.tsv")
    candidate = physio_file.parent / candidate_name
    if candidate.exists():
        return candidate

    return None


def _find_json_sidecar(
        tsv_file: Path
    ) -> Optional[Path]:
    """
    Find the JSON sidecar corresponding to a tabular file.

    Parameters
    ----------
    tsv_file : pathlib.Path
        Path to a ``.tsv`` or ``.tsv.gz`` file.

    Returns
    -------
    json_file : pathlib.Path or None
        Path to the matching ``.json`` sidecar if it exists, otherwise
        ``None``.

    Notes
    -----
    For files ending in ``.tsv.gz``, the suffix pair is replaced with
    ``.json``. For files ending in ``.tsv``, the suffix is replaced with
    ``.json``.

    This function performs a filesystem existence check and does not read or
    write file contents.
    """

    name = tsv_file.name
    if name.endswith(".tsv.gz"):
        candidate = tsv_file.with_name(name[:-7] + ".json")
    elif name.endswith(".tsv"):
        candidate = tsv_file.with_suffix(".json")
    else:
        return None
    return candidate if candidate.exists() else None


def _apply_trimpoints_to_events(
        events_df: pd.DataFrame,
        trimpoints: Tuple[float, float],
        *,
        event_time_col: str="onset"
    ) -> pd.DataFrame:
    """
    Apply an existing trim window to an event dataframe.

    Parameters
    ----------
    events_df : pandas.DataFrame
        DataFrame containing event information.
    trimpoints : tuple of float
        Start and end trim points in seconds as ``(start, end)`` in the
        original file coordinate system.
    event_time_col : str, default="onset"
        Column in ``events_df`` containing event timestamps in seconds.

    Returns
    -------
    trimmed_events : pandas.DataFrame
        Event dataframe filtered to rows whose ``event_time_col`` falls within
        ``trimpoints``. Returned timestamps are shifted so the trimmed segment
        starts at ``0``.

    Notes
    -----
    Non-numeric values in ``event_time_col`` are coerced to ``NaN`` and
    removed before trimming.

    If ``events_df`` is empty, an empty dataframe with column
    ``event_time_col`` is returned.

    This function creates a modified copy of ``events_df`` in memory and does
    not write files or modify the input dataframe in place.
    """

    sta_time, sto_time = trimpoints

    if events_df is None or events_df.empty:
        return pd.DataFrame(columns=[event_time_col])

    out = events_df.copy()
    out[event_time_col] = pd.to_numeric(out[event_time_col], errors="coerce")
    out = out.dropna(subset=[event_time_col])

    mask = (out[event_time_col] >= sta_time) & (out[event_time_col] <= sto_time)
    out = out.loc[mask].copy()
    out[event_time_col] = out[event_time_col] - sta_time
    out = out.reset_index(drop=True)

    return out


def run_pspm_trim_directory(
        root_dir: Union[str, Path],
        from_: Union[float, str],
        to: Union[float, str],
        reference: Any="marker",
        *,
        fs_fallback: Optional[float]=None,
        timestamp_col: str="timestamp",
        event_time_col: str="onset",
        event_name_col: Optional[str]="name",
        event_value_col: Optional[str]="value",
        drop_offset_markers: bool=False,
        prefix: str="t",
        overwrite: bool=False
    ) -> List[Dict[str, Any]]:
    """
    Run ``pspm_trim`` across a directory tree of physiology files and trim
    paired event files to matching windows.

    Parameters
    ----------
    root_dir : str or pathlib.Path
        Root directory to scan recursively for physiology files.
    from_ : float or str
        Start trim point passed to ``pspm_trim``. May be numeric or the string
        ``"none"``.
    to : float or str
        End trim point passed to ``pspm_trim``. May be numeric or the string
        ``"none"``.
    reference : Any, default="marker"
        Reference specification passed to ``pspm_trim``.
    fs_fallback : float or None, optional
        Sampling frequency in Hz to use when a physiology JSON sidecar does
        not define ``"SamplingFrequency"``.
    timestamp_col : str, default="timestamp"
        Timestamp column name for physiology waveform files.
    event_time_col : str, default="onset"
        Timestamp column name for task and auxiliary event files.
    event_name_col : str or None, default="name"
        Event name column used for marker resolution inside ``pspm_trim``.
    event_value_col : str or None, default="value"
        Event value column used for marker resolution inside ``pspm_trim``.
    drop_offset_markers : bool, default=False
        Whether to drop markers in offset regions outside the chosen reference
        markers when trimming relative to markers.
    prefix : str, default="t"
        Prefix added to output filenames when ``overwrite`` is ``False``.
    overwrite : bool, default=False
        Whether to overwrite input files in place. If ``False``, trimmed
        outputs are written to new files with prefixed names.

    Returns
    -------
    results : list of dict of str to Any
        Summary records for each discovered physiology file. Each record
        includes file paths, trim metadata, status, and error information when
        applicable.

    Notes
    -----
    Waveform inputs are files matching ``*_physio.tsv.gz`` and
    ``*_physio.tsv``. Files matching ``*_physioevents.*`` are excluded from
    waveform discovery and are only processed as paired auxiliary event files.

    Matching task events files are expected to follow the pattern
    ``sub-<id>_task-<task>_events.tsv`` in the same directory as each
    physiology file.

    When multiple modalities share the same task events file and
    ``overwrite=True``, the original task events content is cached in memory
    to prevent repeated subtraction of ``"StartTime"``.

    This function performs recursive filesystem scans, reads and writes TSV and
    JSON files, logs progress and errors, and may overwrite existing files
    when ``overwrite`` is ``True``.
    """

    root_dir = Path(root_dir)
    logger.info(f"Scanning directory for physio files: {root_dir}")

    physio_files = sorted(root_dir.rglob("*_physio.tsv.gz")) + sorted(root_dir.rglob("*_physio.tsv"))

    # exclude physioevents masquerading as physio by substring overlap
    physio_files = [p for p in physio_files if "_physioevents." not in p.name]

    logger.info(f"Found {len(physio_files)} physio file(s)")

    results = []

    # Cache original task-events content so each modality uses the same
    # unmodified source, even when overwrite=True.
    task_events_cache: dict[Path, pd.DataFrame] = {}
    task_events_json_cache: dict[Path, dict] = {}

    for physio_file in physio_files:
        logger.info(f"Processing physio file: {physio_file}")

        try:
            physio_json = _find_json_sidecar(physio_file)
            physio_meta = cio.load_json(physio_json) if physio_json else {}

            fs = physio_meta.get("SamplingFrequency", fs_fallback)
            if fs is None:
                logger.warning(
                    f"No SamplingFrequency found for {physio_file.name}; "
                    f"timestamp creation will fail if the file has no timestamp column."
                )

            task_events_file = _find_task_events_file(physio_file)
            physioevents_file = _find_physioevents_pair(physio_file)

            physio_df = _read_tsv_any(physio_file)

            if task_events_file is None:
                logger.warning(
                    f"Skipping {physio_file.name}: no matching task events file found."
                )
                results.append(
                    {
                        "physio_file": str(physio_file),
                        "status": "skipped",
                        "reason": "missing_task_events",
                    }
                )
                continue

            # Always use a cached copy of the ORIGINAL task events file.
            # This prevents double-subtracting StartTime when overwrite=True
            # and multiple modalities share the same *_events.tsv.
            if task_events_file not in task_events_cache:
                logger.debug(f"Caching original task events: {task_events_file}")
                task_events_cache[task_events_file] = _read_tsv_any(task_events_file).copy()

                task_events_json = _find_json_sidecar(task_events_file)
                if task_events_json:
                    task_events_json_cache[task_events_file] = dict(cio.load_json(task_events_json))
                else:
                    task_events_json_cache[task_events_file] = None
            else:
                logger.debug(f"Reusing cached original task events: {task_events_file}")

            task_events_df = task_events_cache[task_events_file].copy()

            logger.debug(
                f"Physio shape={physio_df.shape}, task events shape={task_events_df.shape}"
            )

            if event_time_col in task_events_df.columns:
                logger.debug(
                    "Task events before trim: onset min=%.3f, max=%.3f",
                    task_events_df[event_time_col].min(),
                    task_events_df[event_time_col].max(),
                )

            trim_result = pspm_trim(
                data_df=physio_df,
                events_df=task_events_df,
                from_=from_,
                to=to,
                reference=reference,
                start_time=physio_meta.get("StartTime", 0.0),
                fs=fs,
                timestamp_col=timestamp_col,
                event_time_col=event_time_col,
                event_name_col=event_name_col,
                event_value_col=event_value_col,
                drop_offset_markers=drop_offset_markers,
            )

            logger.info(
                f"Trimmed {physio_file.name}: "
                f"start={trim_result.trimpoints[0]:.3f}s, "
                f"end={trim_result.trimpoints[1]:.3f}s, "
                f"duration={trim_result.duration:.3f}s"
            )

            out_physio_file = physio_file if overwrite else _build_output_path(physio_file, prefix)
            out_task_events_file = (
                task_events_file if overwrite else _build_output_path(task_events_file, prefix)
            )

            _write_tsv_any(trim_result.data, out_physio_file)
            _write_tsv_any(trim_result.events, out_task_events_file)

            if physio_json:
                out_physio_json = physio_json if overwrite else _build_output_path(physio_json, prefix)
                new_meta = dict(physio_meta)
                new_meta["StartTime"] = 0.0
                new_meta["TrimPoints"] = list(trim_result.trimpoints)
                new_meta["Duration"] = trim_result.duration
                cio.save_json(out_physio_json, new_meta)

            cached_task_events_meta = task_events_json_cache.get(task_events_file)
            if cached_task_events_meta is not None:
                out_task_events_json = (
                    _find_json_sidecar(task_events_file)
                    if overwrite
                    else _build_output_path(_find_json_sidecar(task_events_file), prefix)
                )
                task_events_meta = dict(cached_task_events_meta)
                task_events_meta["TrimPoints"] = list(trim_result.trimpoints)
                task_events_meta["Duration"] = trim_result.duration
                cio.save_json(out_task_events_json, task_events_meta)

            out_physioevents_file = None

            if physioevents_file is not None:
                logger.info(f"Found paired physioevents file: {physioevents_file.name}")

                physioevents_df = _read_tsv_any(physioevents_file)

                trimmed_physioevents = _apply_trimpoints_to_events(
                    physioevents_df,
                    trim_result.trimpoints,
                    event_time_col=event_time_col,
                )

                out_physioevents_file = (
                    physioevents_file if overwrite else _build_output_path(physioevents_file, prefix)
                )
                _write_tsv_any(trimmed_physioevents, out_physioevents_file)

                physioevents_json = _find_json_sidecar(physioevents_file)
                if physioevents_json:
                    physioevents_meta = cio.load_json(physioevents_json)
                    out_physioevents_json = (
                        physioevents_json if overwrite else _build_output_path(physioevents_json, prefix)
                    )
                    physioevents_meta = dict(physioevents_meta)
                    physioevents_meta["TrimPoints"] = list(trim_result.trimpoints)
                    physioevents_meta["Duration"] = trim_result.duration
                    physioevents_meta["StartTime"] = 0.0
                    cio.save_json(out_physioevents_json, physioevents_meta)

                logger.info(
                    f"Trimmed paired physioevents file {physioevents_file.name} "
                    f"to same window as {physio_file.name}"
                )

            results.append(
                {
                    "physio_file": str(physio_file),
                    "task_events_file": str(task_events_file),
                    "physioevents_file": str(physioevents_file) if physioevents_file else None,
                    "output_physio_file": str(out_physio_file),
                    "output_task_events_file": str(out_task_events_file),
                    "output_physioevents_file": str(out_physioevents_file) if out_physioevents_file else None,
                    "trimpoints": trim_result.trimpoints,
                    "duration": trim_result.duration,
                    "status": "ok",
                }
            )

        except Exception as e:
            logger.exception(f"Failed processing {physio_file}: {e}")
            results.append(
                {
                    "physio_file": str(physio_file),
                    "status": "error",
                    "error": str(e),
                }
            )

    logger.info("Directory trimming complete")
    return results


def pspm_trim(
        data_df: Union[pd.DataFrame, np.ndarray, List[Any]],
        events_df: Union[pd.DataFrame, pd.Series, np.ndarray, List[Any], Tuple[Any, ...], None],
        from_: Union[float, str],
        to: Union[float, str],
        reference: Any="marker",
        start_time: Optional[float]=None,
        *,
        fs: Optional[Union[float, int]]=None,
        timestamp_col: str="timestamp",
        event_time_col: str="onset",
        event_name_col: Optional[str]="name",
        event_value_col: Optional[str]="value",
        drop_offset_markers: bool=False
    ) -> TrimResult:
    """
    Trim waveform data and events to a time interval and shift timestamps so
    the trimmed segment starts at ``0`` seconds.

    Parameters
    ----------
    data_df : pandas.DataFrame, numpy.ndarray, or list
        Waveform data. Supported inputs are:

        - a DataFrame containing waveform samples
        - a NumPy array of waveform values
        - a list of waveform values

        If an array-like is provided, it is converted to a DataFrame. A
        missing timestamp column is generated using ``fs``.
    events_df : pandas.DataFrame, pandas.Series, numpy.ndarray, list, tuple, or None
        Event information. Supported inputs are:

        - a DataFrame containing an event time column
        - a Series of event onsets
        - a 1-dimensional NumPy array, list, or tuple of event onsets
        - ``None``, meaning no events are available

        If an array-like is provided, it is converted to a DataFrame with one
        column named according to ``event_time_col``.
    from_ : float or str
        Start trim point relative to the chosen reference. May be numeric or
        the string ``"none"``.
    to : float or str
        End trim point relative to the chosen reference. May be numeric or the
        string ``"none"``.
    reference : Any, default="marker"
        Defines how trim points are interpreted. Supported values are:

        - ``"file"``: absolute seconds from the start of the file
        - ``"marker"``: relative to the first and last event
        - a 2-element sequence of integer event indices
        - a 2-element sequence of unique event names or event values

        Name-based and value-based references require ``events_df`` to provide
        the corresponding columns.
    start_time : float or None, optional
        Optional start-time offset in seconds used to align event onsets to the
        waveform time base.
    fs : float, int, or None, optional
        Sampling frequency in Hz used to generate ``timestamp_col`` when it is
        missing from ``data_df``.
    timestamp_col : str, default="timestamp"
        Column in ``data_df`` containing timestamps in seconds.
    event_time_col : str, default="onset"
        Column in ``events_df`` containing event timestamps in seconds.
    event_name_col : str or None, default="name"
        Column containing event names used for reference lookup.
    event_value_col : str or None, default="value"
        Column containing event values used for reference lookup.
    drop_offset_markers : bool, default=False
        If trimming relative to markers, whether to drop markers in the offset
        region outside the reference markers.

    Returns
    -------
    result : TrimResult
        Dataclass containing:

        - ``data``: trimmed waveform dataframe
        - ``events``: trimmed event dataframe with shifted timestamps
        - ``trimpoints``: start and end trim times in original file
          coordinates
        - ``duration``: duration of the trimmed interval in seconds

    Raises
    ------
    TypeError
        Raised if ``data_df`` or ``events_df`` has an unsupported type.
    ValueError
        Raised if trim arguments are invalid, if event arrays are not
        1-dimensional, if marker-based trimming is requested without valid
        events, or if the resulting interval is invalid.
    KeyError
        Raised if ``event_time_col`` is missing from ``events_df`` after
        normalization.
    RuntimeError
        Raised if timestamp generation fails.

    Notes
    -----
    This function mimics the behaviour of PsPM ``pspm_trim`` while operating
    on pandas-compatible structures.

    Input waveform and event data are copied or normalized into DataFrames in
    memory. The returned timestamps are shifted so the trimmed waveform starts
    at ``0``.

    If ``timestamp_col`` is absent from ``data_df``, this function calls
    ``ensure_timestamp`` to generate it using ``fs``.

    This function does not write files, but it does log progress, validation
    details, and warnings via the module logger.

    Examples
    --------
    Trim using absolute time relative to the file start.

    >>> physio = pd.DataFrame({
    ...     "timestamp": [0, 1, 2, 3, 4, 5, 6],
    ...     "scr": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    ... })
    >>> events = pd.DataFrame({
    ...     "onset": [1.5, 4.5],
    ...     "name": ["stim_on", "stim_off"]
    ... })
    >>> res = trim_physio_and_events(physio, events, 1, 5, reference="file")
    >>> res.data["timestamp"].tolist()
    [0.0, 1.0, 2.0, 3.0, 4.0]

    Trim relative to first and last event.

    >>> res = trim_physio_and_events(
    ...     physio,
    ...     events,
    ...     from_=-1,
    ...     to=1,
    ...     reference="marker"
    ... )
    >>> res.trimpoints
    (0.5, 5.5)

    Trim relative to specific event indices.

    >>> res = trim_physio_and_events(
    ...     physio,
    ...     events,
    ...     from_=0,
    ...     to=0,
    ...     reference=(0, 1)
    ... )
    >>> res.duration
    3.0

    Trim relative to event names.

    >>> events = pd.DataFrame({
    ...     "onset": [1, 3, 6],
    ...     "name": ["start", "stim", "end"]
    ... })
    >>> res = trim_physio_and_events(
    ...     physio,
    ...     events,
    ...     from_=0,
    ...     to=0,
    ...     reference=("start", "end"),
    ...     event_name_col="name"
    ... )
    >>> res.data["timestamp"].tolist()
    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

    Provide events as a NumPy array of onsets only.

    >>> events = np.array([1.5, 4.5])
    >>> res = trim_physio_and_events(
    ...     physio,
    ...     events,
    ...     from_=-1,
    ...     to=1,
    ...     reference="marker"
    ... )
    >>> res.events["onset"].tolist()
    [1.0, 4.0]
    """

    logger.info("Starting PsPM-style trimming")

    if isinstance(data_df, pd.DataFrame):
        logger.debug("Input data_df is a DataFrame")
        data_df = data_df.copy()

    elif isinstance(data_df, (np.ndarray, list, tuple)):
        logger.debug("Input data_df is array-like, converting to DataFrame")

        arr = np.asarray(data_df)

        if arr.ndim == 1:
            data_df = pd.DataFrame({"signal": arr})
        else:
            data_df = pd.DataFrame(arr)

        logger.debug(f"Converted data_df shape: {data_df.shape}")

    else:
        raise TypeError(
            "data_df must be a pandas DataFrame, numpy array, list, or tuple"
        )    

    # ensure timestamp column
    if timestamp_col not in data_df.columns:

        if fs is None:
            raise ValueError(
                "fs must be provided when data_df does not contain a timestamp column"
            )

        logger.info(f"Timestamp column not found, generating using sampling rate [fs={fs}]")
        from calinet.utils import ensure_timestamp

        data_df, inserted = ensure_timestamp(data_df, fs, force=False)

        if inserted:
            logger.debug("Timestamp column inserted using sampling rate")
        else:
            logger.debug("Existing timestamp column already valid")

    if timestamp_col not in data_df.columns:
        raise RuntimeError("Failed to create timestamp column")

    # validate trimming inputs
    logger.info(f"Trimming parameters: from={from_}, to={to}, reference={reference}")

    if not (_is_none_token(from_) or isinstance(from_, numbers.Number)):
        raise ValueError("from_ must be numeric or 'none'")

    if not (_is_none_token(to) or isinstance(to, numbers.Number)):
        raise ValueError("to must be numeric or 'none'")

    # validate events
    if events_df is None:
        logger.info("No events provided")
        events_df = pd.DataFrame(columns=[event_time_col])

    elif isinstance(events_df, pd.DataFrame):
        logger.debug("events_df provided as DataFrame")
        events_df = events_df.copy()

    elif isinstance(events_df, pd.Series):
        logger.debug("events_df provided as Series, converting to DataFrame")
        events_df = pd.DataFrame({event_time_col: events_df})

    elif isinstance(events_df, (np.ndarray, list, tuple)):
        logger.debug("events_df provided as array-like, converting to DataFrame")
        arr = np.asarray(events_df)

        if arr.ndim == 0:
            arr = arr.reshape(1)

        if arr.ndim != 1:
            raise ValueError(
                "Array-like events must be 1D and contain event onsets"
            )

        events_df = pd.DataFrame({event_time_col: arr})

    else:
        raise TypeError(
            "events_df must be DataFrame, Series, ndarray, list, tuple or None"
        )

    if event_time_col not in events_df.columns:
        raise KeyError(f"{event_time_col} not found in events_df")

    if not events_df.empty:

        logger.debug(f"Initial events count: {len(events_df)}")

        events_df[event_time_col] = pd.to_numeric(
            events_df[event_time_col], errors="coerce"
        )

        events_df = events_df.dropna(subset=[event_time_col])

        logger.debug(f"Events after NaN removal: {len(events_df)}")

        events_df = events_df.sort_values(event_time_col).reset_index(drop=False)
    else:
        logger.debug("Events dataframe empty")

        events_df = events_df.reset_index(drop=False)

    # compute file time range
    file_start = float(data_df[timestamp_col].iloc[0])   # likely 0.0
    file_end = float(data_df[timestamp_col].iloc[-1])
    duration = file_end - file_start

    logger.debug(f"File start: {file_start:.6f}s, end: {file_end:.6f}s, duration: {duration:.6f}s")

    data_df["_time0"] = data_df[timestamp_col] - file_start

    if not events_df.empty:
        if start_time is None:
            events_df["_time0"] = events_df[event_time_col] - file_start
        else:
            logger.info(f"StartTime={round(start_time)}s, subtracting this from onset times to align channels")
            events_df["_time0"] = events_df[event_time_col] - float(start_time)
    else:
        events_df["_time0"] = pd.Series(dtype=float)

    # resolve reference markers
    logger.debug("Resolving reference markers")

    getmarker, start_idx, end_idx = _resolve_reference_event_indices(
        events_df,
        reference,
        "_time0",
        event_name_col,
        event_value_col,
    )

    logger.debug(
        f"Reference resolution -> use_markers={getmarker}, start_idx={start_idx}, end_idx={end_idx}"
    )    

    # Determine sta_p / sto_p and offsets
    if _is_none_token(from_):
        sta_p = 0.0
        sta_offset = 0.0
    else:
        if getmarker:
            if events_df.empty:
                raise ValueError("Cannot use marker-based start trimming without events.")
            sta_p = float(events_df.loc[start_idx, "_time0"])
            sta_offset = float(from_)
        else:
            sta_p = float(from_)
            sta_offset = 0.0
    
    if _is_none_token(to):
        sto_p = duration
        sto_offset = 0.0
    else:
        if getmarker:
            if events_df.empty:
                raise ValueError("Cannot use marker-based end trimming without events.")
            sto_p = float(events_df.loc[end_idx, "_time0"])
            sto_offset = float(to)
        else:
            sto_p = float(to)
            sto_offset = 0.0

    # Clip to file boundaries
    if (sta_p + sta_offset) < 0:
        logger.warning("Start time outside file bounds, clipping to start")
        if sta_p > 0:
            sta_offset = -sta_p
        else:
            sta_p = 0.0
            sta_offset = 0.0

    sta_time = sta_p + sta_offset

    if (sto_p + sto_offset) > duration:
        logger.warning("End time outside file bounds, clipping to file end")
        sto_time = duration
    else:
        sto_time = sto_p + sto_offset

    if sto_time < sta_time:
        raise ValueError(f"Trim interval is invalid: start={sta_time}, end={sto_time}")

    # Trim waveform data and shift timestamps to new origin
    logger.info(f"Start={round(sta_time, 3)}s, end={round(sto_time, 3)}s")
    data_mask = (
        (data_df["_time0"] >= sta_time)
        & (data_df["_time0"] <= sto_time)
    )

    trimmed_data = data_df.loc[data_mask].copy()

    logger.debug(f"Trimmed waveform samples: {len(trimmed_data)}")

    trimmed_data[timestamp_col] = trimmed_data["_time0"] - sta_time
    trimmed_data = trimmed_data.drop(columns="_time0").reset_index(drop=True)

    # Trim events and shift to new origin
    if not events_df.empty:

        if getmarker and drop_offset_markers:
            newstartpoint = sta_p
            newendpoint = min(sto_p, sto_time)
        else:
            newstartpoint = sta_time
            newendpoint = sto_time

        ev_mask = (
            (events_df["_time0"] >= newstartpoint)
            & (events_df["_time0"] <= newendpoint)
        )

        trimmed_events = events_df.loc[ev_mask].copy()

        logger.debug(f"Trimmed events: {len(trimmed_events)}")

        trimmed_events[event_time_col] = trimmed_events["_time0"] - sta_time
        trimmed_events = trimmed_events.drop(
            columns=["_time0", "index"],
            errors="ignore",
        ).reset_index(drop=True)

    else:
        logger.debug("No events to trim")

        trimmed_events = events_df.drop(
            columns=["_time0", "index"],
            errors="ignore",
        )

    # Return result
    logger.info("Trimming complete")

    return TrimResult(
        data=trimmed_data,
        events=trimmed_events,
        trimpoints=(sta_time, sto_time),
        duration=sto_time - sta_time,
    )

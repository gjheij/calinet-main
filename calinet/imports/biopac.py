# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, List, Union, Pattern

import re
import bioread
import numpy as np
import pandas as pd

import logging
logger = logging.getLogger(__name__)


ChannelSelector = Union[int, str, Pattern[str]]  # index, exact name, or regex pattern


def resolve_channel_regexes(
        acq_channels: list,
        channel_regex_config: dict
    ) -> dict:
    """
    Resolve channel regex config against actual channel names in one .acq file.

    Parameters
    ----------
    acq_channels : list[str]
        Channel names present in the file.
    channel_regex_config : dict[str, Pattern | list[Pattern]]
        Mapping of logical channel names (e.g. SCR, TTL) to one or more regexes.

    Returns
    -------
    dict[str, Pattern]
        One resolved regex per logical channel, chosen from the candidates.

    Raises
    ------
    ValueError
        If no candidate matches for a required logical channel.
    """
    resolved = {}

    for logical_name, candidates in channel_regex_config.items():
        if not isinstance(candidates, (list, tuple)):
            candidates = [candidates]

        matched = None
        for rx in candidates:
            for ch_name in acq_channels:
                if rx.search(ch_name):
                    matched = rx
                    break
            if matched is not None:
                break

        if matched is None:
            raise ValueError(
                f"No matching channel found for '{logical_name}'. "
                f"Candidates: {[getattr(c, 'pattern', str(c)) for c in candidates]}. "
                f"Available channels: {acq_channels}"
            )

        resolved[logical_name] = matched

    return resolved


@dataclass(frozen=True)
class BiopacReadResult:
    """
    Container for Biopac signal data and associated metadata.

    Attributes
    ----------
    df : pd.DataFrame
        Dataframe containing the selected or full signal data.
    sampling_rate_hz : float
        Sampling rate of the signals in Hertz.
    channel_info : pd.DataFrame
        Dataframe describing all available channels (index, name, units,
        sampling rate, etc.).
    selected_channel_info : pd.DataFrame
        Dataframe describing only the selected/output channels.
    selection_map : dict of str to int, optional
        Mapping from output channel names to original column indices.
    """
    df: pd.DataFrame
    sampling_rate_hz: float
    channel_info: pd.DataFrame              
    selected_channel_info: pd.DataFrame
    selection_map: Optional[Dict[str, int]]


def _to_native_endian(arr):
    """Ensure NumPy array is native-endian."""
    arr = np.asarray(arr)

    if arr.dtype.byteorder not in ("=", "|"):
        arr = arr.byteswap().view(arr.dtype.newbyteorder("="))

    return arr


def _build_channel_table(acq) -> pd.DataFrame:
    rows = []
    for i, ch in enumerate(acq.channels):
        rows.append(
            {
                "index": i,
                "name": getattr(ch, "name", None),
                "units": getattr(ch, "units", None),
                "samples_per_second": getattr(ch, "samples_per_second", None),
                "length": len(getattr(ch, "data", [])),
            }
        )
    return pd.DataFrame(rows)


def _select_channel_indices(acq, selectors: Dict[str, ChannelSelector]) -> Dict[str, int]:
    """
    Map output column names -> channel index in file, supporting:
      - int: direct channel index
      - str: exact channel name match (case-insensitive)
      - regex pattern: first match on channel name (case-insensitive)
      - list: int/str/regex candidates 
    """
    names = [getattr(ch, "name", "") or "" for ch in acq.channels]
    names_lower = [n.lower() for n in names]

    out: Dict[str, int] = {}

    for out_name, sel in selectors.items():
        if isinstance(sel, int):
            # selector is int
            idx = sel
            if idx >= len(acq.channels):
                raise IndexError(f"Channel index {idx} out of range (0..{len(acq.channels)-1})")
            elif idx < 0:
                # select last channel
                logger.info(f"Idx={idx}, assuming last channel (={len(acq.channels)-1})")
                idx = len(acq.channels)-1

            out[out_name] = idx
            continue

        if isinstance(sel, re.Pattern):
            # selector is regex
            pat = sel
            idx = next((i for i, n in enumerate(names) if pat.search(n)), None)
            if idx is None:
                raise KeyError(f"No channel name matches regex for '{out_name}': {pat.pattern}")
            out[out_name] = idx
            continue

        if isinstance(sel, str):
            # selector is string
            target = sel.strip().lower()
            idx = next((i for i, n in enumerate(names_lower) if n == target), None)
            if idx is None:
                raise KeyError(f"No channel named '{sel}' found in file. Available: {names}")
            out[out_name] = idx
            continue

        if isinstance(sel, (list, tuple)):
            # list of candidates
            idx = None
            tried = []

            for candidate in sel:
                if isinstance(candidate, int):
                    tried.append(str(candidate))
                    if 0 <= candidate < len(acq.channels):
                        idx = candidate
                        break

                elif isinstance(candidate, re.Pattern):
                    tried.append(candidate.pattern)
                    idx = next((i for i, n in enumerate(names) if candidate.search(n)), None)
                    if idx is not None:
                        break

                elif isinstance(candidate, str):
                    tried.append(candidate)
                    target = candidate.strip().lower()
                    idx = next((i for i, n in enumerate(names_lower) if n == target), None)
                    if idx is not None:
                        break

                else:
                    raise TypeError(
                        f"Unsupported selector type inside sequence for '{out_name}': {type(candidate)}"
                    )

            if idx is None:
                raise KeyError(
                    f"No channel matched any selector for '{out_name}': {tried}. Available: {names}"
                )

            out[out_name] = idx
            continue

        raise TypeError(f"Unsupported selector type for '{out_name}': {type(sel)}")

    return out


import logging

logger = logging.getLogger(__name__)

def read_acq_file(
        path: str,
        channels: Optional[Dict[str, ChannelSelector]] = None,
        *,
        rename: Optional[Dict[str, str]] = None,
        keep_unmapped: bool = False,
    ) -> BiopacReadResult:
    """
    Read a BIOPAC .acq file into a DataFrame.

    Parameters
    ----------
    path:
        Path to .acq file.
    channels:
        Mapping of output column name -> channel selector.
        Selector can be:
          - int: channel index
          - str: exact channel name in file (case-insensitive)
          - compiled regex: match channel name
        If None:
          - keep_unmapped=False (default): returns *all* channels using their file names
          - keep_unmapped=True: same behavior (kept for API compatibility)
    rename:
        Optional mapping to rename columns after loading (e.g., {"HR": "cardiac"}).
        Applied after channel selection.
    drop:
        Optional iterable of column names to drop after rename.
    keep_unmapped:
        If channels is provided and keep_unmapped=True, include all other channels too.

    Returns
    -------
    BiopacReadResult(df, sampling_rate_hz, channel_info)
    """

    logger.info(f"Reading BIOPAC file from: {path}")
    acq = bioread.read_file(path)

    logger.debug(f"Loaded file with {len(acq.channels)} channels")

    channel_info = _build_channel_table(acq)
    channel_srs = channel_info['samples_per_second'].tolist()

    # Choose which channels to load
    if channels is None:
        logger.info("No channel selection provided → loading all channels")

        data = {}
        for i, ch in enumerate(acq.channels):
            name = (getattr(ch, "name", None) or f"ch{i}").strip()
            data[name] = _to_native_endian(ch.data)

        if np.unique(channel_srs).shape[0]>1:
            raise ValueError(f"Found different sampling rates in channels: {np.unique(channel_srs)}. Cannot combine in single dataframe")
        
        df = pd.DataFrame(data)
        selected_idx_map = {col: i for i, col in enumerate(df.columns)}

        logger.debug(f"Loaded channels: {list(df.columns)}")

    else:
        logger.info(f"Selecting channels using mapping: {channels}")

        idx_map = _select_channel_indices(acq, channels)
        idx_list = list(idx_map.values())
        selected_idx_map = idx_map

        logger.debug(f"Resolved channel indices: {idx_map}")

        channel_srs = [channel_srs[i] for i in idx_list]
        logger.debug(f"Available sampling rates: {channel_srs}")
        if np.unique(channel_srs).shape[0]>1:
            raise ValueError(f"Found different sampling rates in channels: {np.unique(channel_srs)}. Cannot combine in single dataframe")
        
        data = {
            out_name: _to_native_endian(acq.channels[idx].data)
            for out_name, idx in idx_map.items()
        }

        if keep_unmapped:
            logger.info("keep_unmapped=True → including remaining channels")

            used = set(idx_map.values())
            for i, ch in enumerate(acq.channels):
                if i in used:
                    continue

                name = (getattr(ch, "name", None) or f"ch{i}").strip()

                if name in data:
                    name = f"{name}__ch{i}"

                data[name] = _to_native_endian(ch.data)
        
        df = pd.DataFrame(data)

        logger.debug(f"Final columns after selection: {list(df.columns)}")

    # Sampling rate
    sr = channel_srs[0]
    logger.info(f"Sampling rate resolved: {sr} Hz")

    # Optional renames
    if rename:
        logger.info(f"Renaming columns: {rename}")
        df = df.rename(columns=rename)

    selected_channel_info = channel_info.copy()

    if selected_idx_map is not None and channels is not None:
        wanted = set(selected_idx_map.values())
        selected_channel_info = selected_channel_info[
            selected_channel_info["index"].isin(wanted)
        ].copy()

        inv = {}
        for out_name, idx in selected_idx_map.items():
            inv.setdefault(idx, []).append(out_name)

        selected_channel_info["output_name"] = selected_channel_info["index"].map(
            lambda i: ",".join(inv.get(i, []))
        )

        selected_channel_info = selected_channel_info.sort_values(
            ["output_name", "index"]
        )

        logger.debug("Constructed selected_channel_info table")

    logger.info("Finished reading BIOPAC file")

    return BiopacReadResult(
        df=df,
        sampling_rate_hz=float(sr) if sr is not None else np.nan,
        channel_info=channel_info,
        selected_channel_info=selected_channel_info,
        selection_map=selected_idx_map,
    )


def _sample_indices_to_ttl(
    onset_idx,
    n_samples: int,
    pulse_width_samples: int = 1,
    pulse_mag: int = 5,
):
    logger.debug(f"n_samples={n_samples}, pulse_width (samples)={pulse_width_samples}, pulse magnitude={pulse_mag}")
    ttl = np.zeros(n_samples, dtype=int)

    onset_idx = np.asarray(onset_idx, dtype=int).ravel()
    onset_idx = onset_idx[(onset_idx >= 0) & (onset_idx < n_samples)]

    pulse_width_samples = max(1, int(pulse_width_samples))

    for idx in onset_idx:
        ttl[idx:min(idx + pulse_width_samples, n_samples)] = pulse_mag

    return ttl


def _deduplicate_sample_onsets(onset_idx, min_gap_samples: int = 1):
    onset_idx = np.asarray(onset_idx, dtype=int).ravel()
    if onset_idx.size == 0:
        return onset_idx

    onset_idx = np.sort(onset_idx)
    kept = [int(onset_idx[0])]

    for idx in onset_idx[1:]:
        if int(idx) - kept[-1] >= min_gap_samples:
            kept.append(int(idx))

    return np.asarray(kept, dtype=int)


def _split_metadata_values(line: str) -> list[str]:
    """
    Split metadata line after '=' on tabs first, then fallback to whitespace.
    """
    rhs = line.split("=", 1)[1].strip()

    if "\t" in rhs:
        vals = [x.strip() for x in rhs.split("\t")]
    else:
        vals = [x.strip() for x in rhs.split()]

    return [v for v in vals if v != ""]


def _clean_unit(unit: Optional[str]) -> Optional[str]:
    if unit is None:
        return None
    unit = unit.strip()
    if unit in {"*", ""}:
        return None

    # BIOPAC txt exported with latin1 often mangles µ as �
    if unit == "�S":
        return "µS"

    return unit


def _select_column_indices_from_names(
    names: list[str],
    selectors: Dict[str, ChannelSelector],
) -> Dict[str, int]:
    names_lower = [n.lower() for n in names]
    out: Dict[str, int] = {}

    for out_name, sel in selectors.items():
        if isinstance(sel, int):
            if sel < 0 or sel >= len(names):
                raise IndexError(f"Column index {sel} out of range (0..{len(names)-1})")
            out[out_name] = sel
            continue

        if isinstance(sel, re.Pattern):
            idx = next((i for i, n in enumerate(names) if sel.search(n)), None)
            if idx is None:
                raise KeyError(f"No column matches regex for '{out_name}': {sel.pattern}")
            out[out_name] = idx
            continue

        if isinstance(sel, str):
            target = sel.strip().lower()
            idx = next((i for i, n in enumerate(names_lower) if n == target), None)
            if idx is None:
                raise KeyError(f"No column named '{sel}' found. Available: {names}")
            out[out_name] = idx
            continue

        if isinstance(sel, (list, tuple)):
            idx = None
            tried = []

            for candidate in sel:
                if isinstance(candidate, int):
                    tried.append(str(candidate))
                    if 0 <= candidate < len(names):
                        idx = candidate
                        break

                elif isinstance(candidate, re.Pattern):
                    tried.append(candidate.pattern)
                    idx = next((i for i, n in enumerate(names) if candidate.search(n)), None)
                    if idx is not None:
                        break

                elif isinstance(candidate, str):
                    tried.append(candidate)
                    target = candidate.strip().lower()
                    idx = next((i for i, n in enumerate(names_lower) if n == target), None)
                    if idx is not None:
                        break

                else:
                    raise TypeError(
                        f"Unsupported selector type inside sequence for '{out_name}': {type(candidate)}"
                    )

            if idx is None:
                raise KeyError(
                    f"No column matched any selector for '{out_name}': {tried}. Available: {names}"
                )

            out[out_name] = idx
            continue

        raise TypeError(f"Unsupported selector type for '{out_name}': {type(sel)}")

    return out


def read_biopac_txt_noheader(
        filepath: str,
        sampling_rate_hz: Union[float, int],
        column_names: Optional[Union[List[str], str]]=None,
    ) -> BiopacReadResult:
    """
    Read a Biopac TXT file without header and return structured data.

    This function loads a tab-separated TXT file without a header row,
    enforces numeric values, assigns column names, and constructs channel
    metadata. It returns a `BiopacReadResult` object containing the data
    and associated channel information.

    Parameters
    ----------
    filepath : str
        Path to the input TXT file.
    sampling_rate_hz : float or int
        Sampling rate of the recorded signals in Hertz.
    column_names : list of str or str, optional
        Column names to assign to the data. If ``None``, default names
        (``ch0``, ``ch1``, ...) are used. If provided, the number of names
        must match the number of columns in the file.

    Returns
    -------
    result : BiopacReadResult
        Object containing:
        - ``df``: the loaded signal data
        - ``sampling_rate_hz``: sampling rate as float
        - ``channel_info``: metadata for all channels
        - ``selected_channel_info``: copy of channel metadata
        - ``selection_map``: None

    Raises
    ------
    ValueError
        If `column_names` is provided and its length does not match the
        number of columns in the file.
    ValueError
        If non-numeric values are encountered in the data.

    Notes
    -----
    - The file is read using tab separation and ASCII encoding.
    - Trailing empty columns (e.g., from trailing tabs) are removed.
    - All values are strictly converted to numeric types.
    - Channel metadata includes index, name, sampling rate, and signal length.

    Examples
    --------
    >>> result = read_biopac_txt_noheader(
    ...     filepath="data.txt",
    ...     sampling_rate_hz=1000,
    ...     column_names=["eda", "ecg"]
    ... )
    >>> result.df.head()
    """
    
    logger.debug("Input file is .txt file without header; reading with 'ascii'-encoding")
    df = pd.read_csv(
        filepath,
        sep="\t",
        header=None,
        encoding="ascii",   # utf-8 also fine here
    )

    # trailing tab creates an empty last column
    df = df.dropna(axis=1, how="all")

    # make sure everything is numeric
    df = df.apply(pd.to_numeric, errors="raise")

    if column_names is None:
        df.columns = [f"ch{i}" for i in range(df.shape[1])]
    else:
        if len(column_names) != df.shape[1]:
            raise ValueError(
                f"column_names has length {len(column_names)}, "
                f"but file has {df.shape[1]} columns"
            )
        
        logger.debug(f"Columns names: {column_names}")
        df.columns = column_names

    channel_info = pd.DataFrame(
        {
            "index": range(df.shape[1]),
            "name": list(df.columns),
            "units": [None] * df.shape[1],
            "samples_per_second": [float(sampling_rate_hz)] * df.shape[1],
            "length": [len(df)] * df.shape[1],
            "output_name": list(df.columns),
        }
    )

    return BiopacReadResult(
        df=df,
        sampling_rate_hz=float(sampling_rate_hz),
        channel_info=channel_info,
        selected_channel_info=channel_info.copy(),
        selection_map=None,
    )


def read_txt_file(
        filepath: str,
        channels: Optional[Dict[str, ChannelSelector]]=None,
        *,
        rename: Optional[Dict[str, str]]=None,
        drop: Optional[Iterable[str]]=None,
        keep_unmapped: bool=False,
        stimulus_label: str="Stimulus",
        create_synthetic_ttl: bool=True,
        ttl_pulse_width_s: float=0.01,
        ttl_pulse_mag: int=5,
        ttl_column_name: str="TTL",
        stimulus_min_gap_s: float=0.5,
    ) -> BiopacReadResult:
    """
    Read a Biopac TXT file with header metadata and return structured signal data.

    This function parses a TXT file containing header-style metadata and numeric
    signal data, extracts the sampling rate and channel information, optionally
    detects stimulus events, creates a synthetic TTL channel, selects and renames
    channels, and returns the result as a `BiopacReadResult`.

    Parameters
    ----------
    filepath : str
        Path to the input TXT file.
    channels : dict of str to ChannelSelector, optional
        Mapping that defines which input columns should be selected and how they
        should be exposed in the output dataframe.
    rename : dict of str to str, optional
        Mapping used to rename selected output columns.
    drop : iterable of str, optional
        Column names to drop from the selected output dataframe.
    keep_unmapped : bool, default=False
        Whether to retain columns that were not explicitly selected by `channels`.
    stimulus_label : str, default="Stimulus"
        Label used to detect stimulus event markers in the text file.
    create_synthetic_ttl : bool, default=True
        Whether to generate a synthetic TTL channel from detected stimulus events.
    ttl_pulse_width_s : float, default=0.01
        Width of each synthetic TTL pulse in seconds.
    ttl_pulse_mag : int, default=5
        Amplitude assigned to the synthetic TTL pulse.
    ttl_column_name : str, default="TTL"
        Name of the synthetic TTL output column.
    stimulus_min_gap_s : float, default=0.5
        Minimum separation in seconds required to treat two detected stimulus
        events as distinct onsets.

    Returns
    -------
    result : BiopacReadResult
        Object containing:
        - ``df``: selected signal dataframe
        - ``sampling_rate_hz``: sampling rate as float
        - ``channel_info``: metadata for all parsed channels
        - ``selected_channel_info``: metadata for selected output channels
        - ``selection_map``: mapping from output names to source column indices

    Raises
    ------
    ValueError
        If the sampling rate cannot be determined from the header metadata.
    ValueError
        If no numeric data rows are found in the file.

    Notes
    -----
    - Metadata is extracted from header lines containing ``key=value`` pairs.
    - The sampling rate is inferred from the ``Interval`` header field.
    - Channel titles are normalized so that ``gsr`` becomes ``SCR`` and
      ``pulse`` becomes ``Pulse``.
    - If enabled, the synthetic TTL channel is created from detected stimulus
      events after deduplication in sample space.
    - If `channels` is not provided, all parsed columns are returned.

    Examples
    --------
    >>> result = read_txt_file(
    ...     filepath="physio.txt",
    ...     create_synthetic_ttl=True,
    ...     ttl_column_name="TTL"
    ... )
    >>> result.df.head()
    """
    
    metadata: Dict[str, Union[List[str], str]] = {}
    times: List[float] = []
    data: List[List[float]] = []
    stimulus_events: List[Dict[str, Union[int, float]]] = []

    logger.debug("Input file is .txt file with header")
    sr = None

    def _is_number_token(tok: str) -> bool:
        try:
            float(tok)
            return True
        except ValueError:
            return False

    stimulus_pattern = re.compile(
        r"\*\s*{0}\b".format(re.escape(stimulus_label)),
        re.IGNORECASE,
    )

    logger.debug("Reading file with 'latin1' encoding")
    with open(filepath, "r", encoding="latin1", errors="replace") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            # header / metadata
            if "=" in line and not _is_number_token(line.split()[0]):
                key = line.split("=", 1)[0].strip()

                if key == "Interval":
                    vals = _split_metadata_values(line)
                    if vals:
                        try:
                            interval = float(vals[0])
                            if interval > 0:
                                sr = 1.0 / interval
                        except ValueError:
                            pass
                elif key in {"ChannelTitle", "UnitName", "Range", "TopValue", "BottomValue"}:
                    metadata[key] = _split_metadata_values(line)
                else:
                    metadata[key] = line.split("=", 1)[1].strip()
                continue

            parts = re.split(r"\s+", line)

            # Original-style event detection:
            # event belongs at the current sample index (= number of samples already read)
            if stimulus_pattern.search(line):
                numeric_parts = [p for p in parts if _is_number_token(p)]
                if numeric_parts:
                    stimulus_events.append(
                        {
                            "sample_index": len(times),
                            "time": float(numeric_parts[0]),
                        }
                    )
                parts = [p for p in parts if _is_number_token(p)]

            try:
                nums = list(map(float, parts))
            except ValueError:
                continue

            if len(nums) < 2:
                continue

            times.append(nums[0])
            data.append(nums[1:])

    if sr is None:
        raise ValueError("Could not determine sampling rate from header 'Interval'.")

    logger.debug(f"Extracted SamplingFrequency: {sr}")
    if not data:
        raise ValueError("No numeric data rows found in {!r}".format(filepath))

    n_signal_cols = len(data[0])

    logger.debug("Building channel information dataframe from metadata")
    raw_titles = metadata.get("ChannelTitle", [])
    if not isinstance(raw_titles, list):
        raw_titles = []

    if len(raw_titles) == n_signal_cols:
        signal_names = raw_titles[:]
    else:
        signal_names = ["ch{0}".format(i) for i in range(n_signal_cols)]

    normalized_signal_names = []
    for name in signal_names:
        low = name.strip().lower()
        if low == "gsr":
            normalized_signal_names.append("SCR")
        elif low == "pulse":
            normalized_signal_names.append("Pulse")
        else:
            normalized_signal_names.append(name.strip())

    df = pd.DataFrame(data, columns=normalized_signal_names)
    df.insert(0, "Time", times)

    # Build TTL from event sample indices, deduplicated in sample space
    if create_synthetic_ttl:
        logger.debug("Create synthetic TTL channel")
        if stimulus_events:
            onset_idx = np.asarray(
                [int(ev["sample_index"]) for ev in stimulus_events],
                dtype=int,
            )
        else:
            onset_idx = np.asarray([], dtype=int)

        min_gap_samples = max(1, int(round(stimulus_min_gap_s * sr)))
        onset_idx = _deduplicate_sample_onsets(onset_idx, min_gap_samples=min_gap_samples)

        pulse_width_samples = max(1, int(round(ttl_pulse_width_s * sr)))

        df[ttl_column_name] = _sample_indices_to_ttl(
            onset_idx=onset_idx,
            n_samples=len(df),
            pulse_width_samples=pulse_width_samples,
            pulse_mag=ttl_pulse_mag,
        )

    units = metadata.get("UnitName", [])
    if not isinstance(units, list):
        units = []

    ranges = metadata.get("Range", [])
    if not isinstance(ranges, list):
        ranges = []

    tops = metadata.get("TopValue", [])
    if not isinstance(tops, list):
        tops = []

    bottoms = metadata.get("BottomValue", [])
    if not isinstance(bottoms, list):
        bottoms = []

    channel_rows = [
        {
            "index": 0,
            "name": "Time",
            "units": "s",
            "samples_per_second": float(sr),
            "length": len(df),
            "range": None,
            "top_value": None,
            "bottom_value": None,
        }
    ]

    for i, name in enumerate(normalized_signal_names, start=1):
        raw_idx = i - 1
        channel_rows.append(
            {
                "index": i,
                "name": name,
                "units": _clean_unit(units[raw_idx]) if raw_idx < len(units) else None,
                "samples_per_second": float(sr),
                "length": len(df),
                "range": ranges[raw_idx] if raw_idx < len(ranges) and ranges[raw_idx] != "*" else None,
                "top_value": tops[raw_idx] if raw_idx < len(tops) and tops[raw_idx] != "*" else None,
                "bottom_value": bottoms[raw_idx] if raw_idx < len(bottoms) and bottoms[raw_idx] != "*" else None,
            }
        )

    if create_synthetic_ttl:
        channel_rows.append(
            {
                "index": len(channel_rows),
                "name": ttl_column_name,
                "units": "a.u.",
                "samples_per_second": float(sr),
                "length": len(df),
                "range": None,
                "top_value": ttl_pulse_mag,
                "bottom_value": 0.0,
            }
        )

    channel_info = pd.DataFrame(channel_rows)

    if channels is None:
        selected_df = df.copy()
        selected_idx_map = {col: i for i, col in enumerate(df.columns)}
    else:
        logger.debug(f"Select column indices from column names: {list(df.columns)}")
        idx_map = _select_column_indices_from_names(list(df.columns), channels)
        selected_idx_map = idx_map

        selected_data = {
            out_name: df.iloc[:, idx].to_numpy()
            for out_name, idx in idx_map.items()
        }

        if keep_unmapped:
            used = set(idx_map.values())
            for i, col in enumerate(df.columns):
                if i in used:
                    continue
                name = col if col not in selected_data else "{0}__col{1}".format(col, i)
                selected_data[name] = df.iloc[:, i].to_numpy()

        selected_df = pd.DataFrame(selected_data, index=df.index)

    if rename:
        selected_df = selected_df.rename(columns=rename)

    if drop:
        selected_df = selected_df.drop(columns=list(drop), errors="ignore")

    selected_channel_info = channel_info.copy()

    if selected_idx_map is not None and channels is not None:
        wanted = set(selected_idx_map.values())
        selected_channel_info = selected_channel_info[
            selected_channel_info["index"].isin(wanted)
        ].copy()

        inv = {}
        for out_name, idx in selected_idx_map.items():
            inv.setdefault(idx, []).append(out_name)

        selected_channel_info["output_name"] = selected_channel_info["index"].map(
            lambda i: ",".join(inv.get(i, []))
        )
        selected_channel_info = selected_channel_info.sort_values(["output_name", "index"])
    else:
        selected_channel_info["output_name"] = selected_channel_info["name"]

    return BiopacReadResult(
        df=selected_df,
        sampling_rate_hz=float(sr),
        channel_info=channel_info,
        selected_channel_info=selected_channel_info,
        selection_map=selected_idx_map,
    )

# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Union

from calinet.core.metadata import build_hed_map
from calinet.core.shock import (
    get_blank_task_ratings,
    extract_task_ratings_from_events_df
)

from calinet import utils
from calinet.core.io import save_json

from calinet.config import config
ENIGMA_EVENT_NAMES = config.get("event_names")

import logging
logger = logging.getLogger(__name__)


def parse_events_csv(
        events_file: str,
        task_name: str,
        event_onsets: list
    ):

    logger.info(f"Reading events from {events_file}")
    events_df = pd.read_csv(events_file)

    this_task_ratings = extract_task_ratings_from_events_df(
        task_name,
        events_df
    )

    column_list = [
        "trial_type",
        "us",
        "cs_img",
        "cs_duration",
        "task"
    ]

    if "cs_onset" in list(events_df.columns):
        ref_event = "cs_onset"
        column_list += ["cs_onset"]
    else:
        # Ams doesn't have 'cs_onset' -> use duration to filter ev's
        ref_event = "cs_duration"

    if task_name == "acquisition":
        column_list += ["shock_begin", "shock_end"]
    
    # filter events
    events_df = (
        events_df[column_list]
        .dropna(subset=[ref_event])
        .loc[lambda df: df["trial_type"].str.contains("CS", na=False)]
    )

    # add onsets
    if len(events_df) != len(event_onsets):
        raise ValueError(f"Psychopy file contains {len(events_df)} trials for '{task_name}', but {len(event_onsets)} were extracted from physio-file")
    
    events_df["cs_onset"] = event_onsets
    events_df.rename(
        columns={
            "trial_type": "event_type",
            "cs_duration": "duration",
            "task": "task_name",
            "cs_onset": "onset",
            "cs_img": "stimulus_name"
        },
        inplace=True
    )

    # enfore float
    events_df["duration"] = events_df["duration"].astype(float)

    # add stimulus names
    events_df["stimulus_name"] = events_df["stimulus_name"].apply(
        lambda x: "square" if "aligned" in x else "diamond"
    )

    events_df = events_df.reset_index(drop=True)

    if task_name == "acquisition":

        us_delay = config.get("SOA", 7.5)

        # rows where a US should occur
        us_rows = events_df.loc[events_df["us"] == 1.0].copy()

        if not us_rows.empty:

            logger.info(f"Adding {len(us_rows)} US trials to events")
            # convert original rows to CS-only
            events_df.loc[events_df["us"] == 1.0, "us"] = 0

            # create US rows
            us_rows["event_type"] = "US"
            us_rows["stimulus_name"] = "shock"
            us_rows["onset"] = us_rows["onset"] + us_delay
            us_rows["duration"] = us_rows["shock_end"]-us_rows["shock_begin"]
            us_rows["us"] = 1

            # combine and sort by onset
            events_df = (
                pd.concat([events_df, us_rows], ignore_index=True)
                .sort_values("onset")
                .reset_index(drop=True)
            )

        # drop helper column if present
        events_df.drop(
            columns=["shock_begin", "shock_end", "shock_duration", "us"], 
            errors="ignore",
            inplace=True
        )

    # add additional events
    events_df = process_trial_data(
        events_df,
        CS_type="visual",
        US_type="electrical",
        soa=config.get("SOA", 7.5),
        insert_usp=False # already added above
    )

    # events_df = add_start_end_block_rows(events_df, task_name)

    return (this_task_ratings, events_df)


def add_event(
        df,
        ref_event=None,
        new_event=None,
        soa=None,
        duration=0.0,
        time_column="onset",
        duration_column="duration",
        event_column="event_type",
        hed_map=None,
    ):

    out = df.copy()
    if new_event not in set(out[event_column].astype(str).unique()):
        csm_rows = out.loc[out[event_column].astype(str) == ref_event].copy()
        if not csm_rows.empty:
            usm_rows = csm_rows.copy()
            usm_rows[time_column] = pd.to_numeric(usm_rows[time_column], errors="coerce") + float(soa)
            usm_rows[event_column] = new_event
            usm_rows["HED"] = hed_map.get(new_event, "n/a")

            if duration_column is not None and duration_column in usm_rows.columns:
                usm_rows[duration_column] = duration

            out = pd.concat([out, usm_rows], ignore_index=True)
            out = out.sort_values(by=[time_column], kind="mergesort").reset_index(drop=True)

    return out


def process_trial_data(
    df: pd.DataFrame,
    CS_type: str = "auditory",
    US_type: str = "electrical",
    event_column: str = "event_type",
    soa: Optional[Union[float, int]] = None,
    time_column: str = "onset",
    duration_column: Optional[str] = "duration",
    insert_usm: bool = True,
    insert_uso: bool = True,
    insert_usp: bool = True,
    us_duration: float= 0.5
):
    if event_column not in df.columns:
        raise ValueError(f"Input df must contain '{event_column}'.")

    if insert_usm or insert_uso:
        if soa is None:
            raise ValueError("soa must be provided when insert_usm=True or insert_uso=True.")
        if time_column not in df.columns:
            raise ValueError(f"'{time_column}' is required to insert USm/USo events.")

    hed_map = build_hed_map(CS_type, US_type)

    out = df.copy()
    existing_labels = set(out[event_column].astype(str).unique())

    # Remap ONLY if data is not already in processed label space
    already_remapped = existing_labels.issubset(ENIGMA_EVENT_NAMES)
    if not already_remapped:
        ev = out[event_column].astype(str).tolist()
        remapped = ["n/a"] * len(ev)

        for i, event in enumerate(ev):
            e = event.strip()

            # US events (raw or already suffixed)
            if e == "US" or e.startswith("US"):
                remapped[i] = "USp"
                continue

            # CS events like CS1m, CS2p, CSm, CSp, etc.
            if e.startswith("CS"):
                # "m" anywhere at end => CS-
                if e.endswith(("m", "-")):
                    remapped[i] = "CSm"
                    continue

                # "p" at end => CS+
                if e.endswith(("p", "+")):
                    # reinforced if next event is US or USp
                    if i < len(ev) - 1 and str(ev[i + 1]).strip().startswith("US"):
                        remapped[i] = "CSpr"
                    else:
                        remapped[i] = "CSpu"
                    continue

            # otherwise leave as-is
            remapped[i] = e

        out[event_column] = remapped

    # Ensure HED is present/updated (works for both raw-remapped and already-remapped)
    out["HED"] = [hed_map.get(str(lbl), "n/a") for lbl in out[event_column].astype(str)]

    # Insert USm rows only if requested and not already present
    if insert_usm:
        out = add_event(
            out,
            ref_event="CSm",
            new_event="USm",
            soa=soa,
            time_column=time_column,
            duration_column=duration_column,
            hed_map=hed_map,
            duration=0.0
        )

    if insert_uso:
        out = add_event(
            out,
            ref_event="CSpu",
            new_event="USo",
            soa=soa,
            time_column=time_column,
            duration_column=duration_column,
            hed_map=hed_map,
            duration=0.0
        )

    if insert_usp:
        out = add_event(
            out,
            ref_event="CSpr",
            new_event="USp",
            soa=soa,
            time_column=time_column,
            duration_column=duration_column,
            hed_map=hed_map,
            duration=us_duration
        )

    # Levels dict for events.json
    levels_dict = {
        "CSpr": {
            "Description": "Conditioned stimulus (CS+) followed by US (reinforced)",
            "HED": hed_map.get("CSpr", "n/a"),
        },
        "CSpu": {
            "Description": "Conditioned stimulus (CS+) not followed by US (unreinforced)",
            "HED": hed_map.get("CSpu", "n/a"),
        },
        "CSm": {
            "Description": "Conditioned stimulus (CS-) never paired with US",
            "HED": hed_map.get("CSm", "n/a"),
        },
        "USp": {
            "Description": "Unconditioned aversive stimulus",
            "HED": hed_map.get("USp", "n/a"),
        },
        "USm": {
            "Description": "Timepoint where US would have occurred during CS- trials (no US delivered)",
            "HED": hed_map.get("USm", "n/a"),
        },
        "USo": {
            "Description": "Timepoint where US would have occurred during CS+ trials (no US delivered)",
            "HED": hed_map.get("USo", "n/a"),
        },        
    }
    
    # reorder columns
    cols =  [
        "onset",
        "duration",
        "event_type",
        "stimulus_name",
        "HED",
        "task_name"
    ]
    
    out = out[[c for c in cols if c in out.columns]]

    return out, levels_dict


def infer_modalities_from_hed(levels_dict: dict):
    """
    Infer CS and US modalities from an existing levels_dict
    (like the one you printed).

    Returns
   ----
    cs_modality : str
    us_modality : str
    """

    # Reverse maps
    cs_reverse_map = {
        "Visual-presentation": "visual",
        "Auditory-presentation": "auditory",
        "Somatosensory-stimulation": "somatosensory",
    }

    us_reverse_map = {
        "Auditory-presentation": "auditory",
        "Somatosensory-stimulation": "electrical", 
    }

    # Infer CS modality
    cs_hed = levels_dict.get("CSm", {}).get("HED", "")
    cs_modality = None
    for hed_tag, mod_name in cs_reverse_map.items():
        if hed_tag in cs_hed:
            cs_modality = mod_name
            break

    if cs_modality is None:
        raise ValueError("Could not infer CS modality from HED.")

    # Infer US modality
    us_hed = (
        levels_dict.get("USp", {}).get("HED", "") or
        levels_dict.get("USm", {}).get("HED", "")
    )

    us_modality = None
    for hed_tag, mod_name in us_reverse_map.items():
        if hed_tag in us_hed:
            us_modality = mod_name
            break

    if us_modality is None:
        raise ValueError(f"Could not infer US modality from HED levels: {levels_dict.keys()}.")

    return cs_modality, us_modality


def parse_events_xlsx(event_file, task_name, event_onsets):
    
    # read file
    events_df = pd.read_excel(event_file)

    # create placeholders
    this_task_ratings = get_blank_task_ratings(task_name)

    # extract columns
    events_df = events_df[["cs_duration", "trial_type", "cs_img", "task"]]

    # rename
    events_df.rename(
        columns={
            "trial_type": "event_type",
            "cs_duration": "duration",
            "task": "task_name",
            "cs_img": "stimulus_name"
        },
        inplace=True
    )

    # Ensure the length of event_onsets matches the number of rows in the dataframe
    if len(event_onsets) > len(events_df):
        logger.warning("Extra markers found in physio data. Trimming")
        event_onsets = event_onsets[: len(events_df)]

    # Add 'onsets' column at the beginning of the dataframe
    events_df.insert(0, "onset", event_onsets)

    # Replace 'cs_img' values
    events_df["stimulus_name"] = events_df["stimulus_name"].replace(
        {
            "stimuli/aligned.png": "square",
            "stimuli/rotated.png": "diamond"
        }
    )

    # events_df = add_start_end_block_rows(events_df, task_name)
    return (this_task_ratings, events_df)


def write_events_to_file(
        events_df: pd.DataFrame,
        events_tpl: dict=None,
        subject_name: str=None,
        beh_folder: str=None,
        task_name:str=None,
        levels_dict: dict=None
    ) -> str:

    # Events JSON
    events_json_output_path = os.path.join(
        beh_folder,
        f"{subject_name}_task-{task_name}_events.json",
    )

    # set updated levels (ENIGMA naming)
    if isinstance(levels_dict, dict):
        events_tpl["event_type"] = levels_dict

    # save
    save_json(events_json_output_path, events_tpl)

    # Events TSV
    events_tsv_output_path = events_json_output_path.replace(".json", ".tsv")
    events_df.fillna("n/a").to_csv(
        events_tsv_output_path,
        sep="\t",
        index=False
    )

    logger.info(f"Saved {task_name} events to {events_tsv_output_path}")
    return events_tsv_output_path


def handle_events(
        raw_path: str=None,
        conv_path: str=None,
        subject_name: str=None,
        events_dict: dict=None,
        events_tpl: dict=None,
        write_files: bool=True
    ) -> Tuple[dict, dict]:
    
    logger.info("Processing events data")

    sessions = {
        "acquisition": "ses-01",
        "extinction": "ses-02"
    }

    task_ratings = {}
    task_events = {}
    def _merge_ratings(dest: dict, src: dict):
        for k, v in src.items():
            dest.setdefault(k, []).extend(v)

    def _try_parse(task_name: str):
        """Return (ratings_dict, events_df) or (None, None) if nothing parsed."""
        csv_event_file = utils.find_events_file_csv(raw_path, task_name)
        if csv_event_file:
            return parse_events_csv(
                csv_event_file,
                task_name,
                events_dict.get(task_name)
            )

        logger.warning(
            f"No Events CSV found for {subject_name}, task-{task_name}. Trying XLSX..."
        )

        xlsx_event_file = utils.find_events_file_xlsx(
            raw_path,
            task_name
        )

        if not xlsx_event_file:
            logger.warning(
                f"No Events XLSX found for {subject_name}, task-{task_name}. No events files found."
            )
            return None, None

        try:
            return parse_events_xlsx(
                xlsx_event_file,
                task_name,
                events_dict.get(task_name)
            )
        except Exception:
            logger.exception(
                f"Failed to parse XLSX events for {subject_name}, task-{task_name} (file: {xlsx_event_file})"
            )
            return None, None


    def _write_events(events_df, **kwargs):
        beh_folder = os.path.join(conv_path, "physio")
        os.makedirs(beh_folder, exist_ok=True)

        return write_events_to_file(
            events_df,
            events_tpl=events_tpl,
            subject_name=subject_name,
            beh_folder=beh_folder,
            **kwargs
        )

    for task_name in sessions.keys():
        
        # parse events
        logger.info(f"Parsing events for task='{task_name}'")
        ratings, parsed_events = _try_parse(task_name)

        # events consists of onset dataframe and levels-dict for events.json
        if isinstance(parsed_events, tuple):
            df_events, levels_dict = parsed_events
        else:
            df_events = parsed_events.copy()
            levels_dict = None
        
        # check
        if ratings is None and df_events is None:
            return None

        _merge_ratings(task_ratings, ratings)
        if df_events is None or getattr(df_events, "empty", True):
            logger.warning(
                f"Parsed events but dataframe is empty for {subject_name}, task-{task_name}"
            )
            continue
        
        if write_files:
            try:
                ev_file = _write_events(
                    df_events,
                    task_name=task_name,
                    levels_dict=levels_dict
                )
                task_events[task_name] = ev_file
                logger.info(f"Created events data for {subject_name}, task-{task_name}")
            except Exception:
                logger.exception(
                    f"Error creating events for {subject_name}, task-{task_name}"
                )
        else:
            task_events[task_name] = df_events.copy()

    return task_ratings, task_events


def extract_onsets_from_ttl(
        physio_df: pd.DataFrame,
        sr: int,
        min_cluster_size: int =20,
        gap_factor: float=5.0,
        min_gap_sec: float=30.0,
        event_col: str="TTL",
        thr_on: float=2.5,
        thr_off: float=1.0,
        min_rise_interval_sec: Optional[float]=None,
        min_rise_interval_floor_sec: float=0.05,
    ) -> list:
    """
    Extract stimulus onset times from an analog TTL/marker channel.

    This function converts an analog TTL channel (e.g., BIOPAC stimulation
    output) into a sequence of clean stimulus onset timestamps. It is designed
    to be robust to noise, analog jitter, and marker artifacts commonly present
    in physiological recordings.

    The extraction proceeds in several stages:

    1. **Analog → digital conversion using hysteresis**
       The TTL signal is binarized using two thresholds (`thr_on`, `thr_off`)
       to prevent rapid state oscillations due to noise.

    2. **Rising edge detection**
       Rising edges in the digital signal correspond to potential stimulus
       onset events.

    3. **Refractory filtering**
       A minimum interval between successive rising edges is enforced to
       suppress duplicate detections within the same stimulus pulse.

    4. **Cluster filtering**
       Stimulus events are grouped based on long temporal gaps. Clusters
       containing fewer than `min_cluster_size` events are considered noise
       and removed.

    Parameters
    ----------
    physio_df : pandas.DataFrame
        DataFrame containing physiological recordings, including a TTL/event
        channel.

    sr : int
        Sampling rate of the physiological recording (Hz).

    min_cluster_size : int, default=20
        Minimum number of events required for a cluster to be considered valid.
        Clusters with fewer events are discarded.

    gap_factor : float, default=5.0
        Multiplier applied to the estimated inter-onset interval (IOI) to
        determine the threshold for identifying large gaps between clusters.

    min_gap_sec : float, default=30.0
        Absolute minimum gap duration (seconds) used when identifying cluster
        boundaries.

    event_col : str, default="TTL"
        Name of the column containing the TTL or stimulation signal.

    thr_on : float, default=2.5
        Threshold above which the signal is considered ON.

    thr_off : float, default=1.0
        Threshold below which the signal is considered OFF.

        Using two thresholds creates hysteresis, which prevents rapid
        toggling due to noise (i.e., `thr_on > thr_off`).

    min_rise_interval_sec : float or None, default=None
        Minimum allowed interval between consecutive rising edges (seconds).
        If `None`, the interval is estimated automatically as half the typical
        inter-onset interval.

    min_rise_interval_floor_sec : float, default=0.05
        Lower bound for the refractory interval when estimating it
        automatically. Prevents unrealistically small intervals.

    Returns
    -------
    list of float
        List of stimulus onset times in seconds.

    Raises
    ------
    AssertionError
        If the specified event column does not exist or if invalid threshold
        settings are provided.

    Exception
        If no TTL markers can be detected in the signal.

    Notes
    -----
    This function is particularly useful when TTL markers are stored as
    **analog signals** rather than digital pulses, which is common in
    BIOPAC exports. The hysteresis + refractory filtering combination makes
    the method robust to noisy marker signals.

    The clustering step additionally removes small marker bursts that may
    originate from recording artifacts or device glitches.

    Typical workflow:

    >>> onsets = extract_onsets_from_ttl(df, sr=1000)
    >>> print(onsets[:5])

    The returned onsets can then be used for stimulus alignment,
    event-related analyses, or constructing synthetic TTL channels.
    """

    logger.info(f"Extracting marker data [sr={sr}]")
    assert event_col in physio_df.columns, (
        f"Physio-dataframe does not contain event channel named '{event_col}'"
    )
    assert thr_on > thr_off, "thr_on must be > thr_off for hysteresis"

    x = physio_df[event_col].to_numpy()
        
    # --- 1) Hysteresis to digital state ---
    state = np.zeros_like(x, dtype=np.int8)
    is_on = False
    for i, v in enumerate(x):
        if (not is_on) and (v >= thr_on):
            is_on = True
        elif is_on and (v <= thr_off):
            is_on = False
        state[i] = 1 if is_on else 0

    # --- 2) Raw rising edges ---
    rising_raw = np.where(np.diff(state) == 1)[0] + 1
    if len(rising_raw) == 0:
        raise Exception(
            f"No markers found using column={event_col} with thr_on={thr_on}, thr_off={thr_off}."
        )

    times_raw = rising_raw / sr

    if len(times_raw) == 1:
        logger.info("1 marker found.")
        return times_raw.tolist()

    # --- 3) Estimate typical IOI from raw rises (before filtering) ---
    diffs_raw = np.diff(times_raw)
    small = diffs_raw[diffs_raw < np.percentile(diffs_raw, 90)]
    typical_ioi = float(np.median(small)) if len(small) else float(np.median(diffs_raw))

    # If user didn't provide a refractory interval, pick something conservative.
    # For TTL pulse trains, a good start is 50% of typical IOI (prevents double-counting within an IOI).
    if min_rise_interval_sec is None:
        min_rise_interval_sec = max(0.5 * typical_ioi, min_rise_interval_floor_sec)
    else:
        min_rise_interval_sec = max(min_rise_interval_sec, min_rise_interval_floor_sec)

    logger.info(
        f"typical_ioi≈{typical_ioi:.3f}s | min_rise_interval_sec={min_rise_interval_sec:.3f}s | "
        f"min_gap_sec={min_gap_sec} | min_cluster_size={min_cluster_size}"
    )

    # --- 4) Refractory filter on rising edges ---
    min_rise_samples = int(round(min_rise_interval_sec * sr))
    keep_idx = [0]
    last = rising_raw[0]
    for i in range(1, len(rising_raw)):
        if (rising_raw[i] - last) >= min_rise_samples:
            keep_idx.append(i)
            last = rising_raw[i]

    rising_edges = rising_raw[np.array(keep_idx, dtype=int)]
    times = rising_edges / sr

    if len(times) == 1:
        logger.info("1 marker found after refractory filtering.")
        return times.tolist()

    # --- 5) Long-gap clustering (your original logic) ---
    diffs = np.diff(times)

    if len(diffs) > 0:
        largest_gap = float(np.max(diffs))
        largest_gap_idx = int(np.argmax(diffs))
        logger.info(
            f"Largest gap between markers: {largest_gap:.3f}s "
            f"(between markers {largest_gap_idx} and {largest_gap_idx+1})"
        )
        
    small_diffs = diffs[diffs < np.percentile(diffs, 90)]
    typical_ioi2 = np.median(small_diffs) if len(small_diffs) > 0 else np.median(diffs)

    long_gap_thresh = max(gap_factor * typical_ioi2, min_gap_sec)
    gap_idx = np.where(diffs > long_gap_thresh)[0]

    logger.info(f"Long gap threshold={round(long_gap_thresh, 3)}s ({round(typical_ioi2, 3)}*{gap_factor})")
    if len(gap_idx) == 0:
        logger.info(f"{len(times)} markers found in a single cluster. No long gaps.")
        return times.tolist()

    cluster_bounds = []
    start = 0
    for gi in gap_idx:
        end = gi + 1
        cluster_bounds.append((start, end))
        start = end
    cluster_bounds.append((start, len(times)))

    clusters, kept_sizes, dropped_sizes = [], [], []
    for (s, e) in cluster_bounds:
        c = times[s:e]
        if len(c) >= min_cluster_size:
            clusters.append(c)
            kept_sizes.append(len(c))
        else:
            dropped_sizes.append(len(c))

    total = len(times)

    if clusters:
        logger.info(
            f"{total} markers found ({'+'.join(map(str, [len(times[s:e]) for s,e in cluster_bounds]))})."
        )
        logger.info(f"Keeping clusters with sizes: {kept_sizes}, dropping: {dropped_sizes}")
        cleaned_times = np.concatenate(clusters)
    else:
        logger.info(f"{total} markers found, but no large cluster; keeping all.")
        cleaned_times = times

    logger.info("Marker data successfully extracted")
    return cleaned_times.tolist()

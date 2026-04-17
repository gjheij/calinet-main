# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import re
import ast
import json
import logging
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Union

logger = logging.getLogger(__name__)

# Regexes tuned to PsychoPy .log style
# Example line: "308.2216    EXP    New trial (rep=0, index=0): {'task': ...}"
TRIAL_RE = re.compile(r"^(\d+\.\d+)\s+\t?EXP\s+\t?New trial.*?:\s+(\{.*\})")

# Example line: "325.2343    EXP    cs: autoDraw = True"
AUTODRAW_RE = re.compile(r"^(\d+\.\d+)\s+\t?EXP\s+\t?([^:]+):\s+autoDraw\s*=\s*(True|False)")


# PsychoPy .log patterns
TRIAL_RE = re.compile(
    r"^(\d+\.\d+)\s+\t?EXP\s+\t?New trial.*?:\s+(\{.*\})"
)
AUTODRAW_RE = re.compile(
    r"^(\d+\.\d+)\s+\t?EXP\s+\t?([^:]+):\s+autoDraw\s*=\s*(True|False)"
)


@dataclass
class Trial:
    trial_n: int
    trial_start: float
    info: Dict[str, Any]
    fixation_on: Optional[float] = None
    fixation_off: Optional[float] = None
    pretrial_on: Optional[float] = None
    pretrial_off: Optional[float] = None
    stim_on: Optional[float] = None
    stim_off: Optional[float] = None


def extract_psychopy_trial_timing(
        path: str,
        *,
        fixation_component: str="fixation_image_hab",
        pretrial_component: Optional[str]="fixation_image_pre_trial",
        stim_component: str="cs",
        filter_info: Optional[List[str]]=None,
        stim_duration_key: str="cs_duration",
        iti_duration_key: str="iti_duration",
    ) -> pd.DataFrame:
    """
    Extract trial timing information from a PsychoPy log file.

    This function parses a PsychoPy ``.log`` file and reconstructs trial-level
    timing for selected visual components by detecting trial boundaries and
    ``autoDraw`` state changes. For each retained trial, it captures onset and
    offset times for a fixation component, an optional pre-trial component,
    and a stimulus component, then derives summary timing metrics.

    Parameters
    ----------
    path : str
        Path to the PsychoPy ``.log`` file to parse.
    fixation_component : str, default="fixation_image_hab"
        Component name used to detect fixation onset and offset from
        ``autoDraw`` log entries.
    pretrial_component : str or None, default="fixation_image_pre_trial"
        Optional component name used to detect a pre-trial interval. If
        ``None``, no pre-trial component is tracked.
    stim_component : str, default="cs"
        Component name used to detect stimulus onset and offset from
        ``autoDraw`` log entries.
    filter_info : list of str or None, optional
        Optional list of keys used to filter which trial-boundary entries are
        retained. If provided, a trial is kept only if at least one of these
        keys is present in the trial's ``info`` dictionary.
    stim_duration_key : str, default="cs_duration"
        Key in the trial ``info`` dictionary containing the scheduled stimulus
        duration.
    iti_duration_key : str, default="iti_duration"
        Key in the trial ``info`` dictionary containing the scheduled
        inter-trial interval duration.

    Returns
    -------
    df : pandas.DataFrame
        Dataframe with one row per retained trial. Columns include:

        - ``trial_n``: zero-based retained trial index
        - ``trial_start``: timestamp of the parsed trial boundary
        - ``task``: value of ``info["task"]`` if present
        - ``trial_type``: value of ``info["trial_type"]`` or ``info["type"]``
        - ``info``: original parsed trial info dictionary
        - ``fixation_on``, ``fixation_off``, ``fixation_dur``
        - ``pretrial_on``, ``pretrial_off``, ``pretrial_dur``
        - ``stim_on``, ``stim_off``, ``stim_dur_logged``
        - ``fixation_to_stim``: ``stim_on - fixation_on``
        - ``scheduled_stim_duration``: value from ``stim_duration_key``
        - ``scheduled_iti_duration``: value from ``iti_duration_key``

        If the dataframe is not empty, additional derived columns are added:

        - ``next_stim_on``: next trial's ``stim_on``
        - ``stim_to_next_stim``: time from current ``stim_on`` to next
          ``stim_on``
        - ``stim_dur_error``: logged minus scheduled stimulus duration
        - ``fixation_to_stim_minus_scheduled_iti``: observed minus scheduled
          interval from fixation onset to stimulus onset
        - ``fixation_dur_minus_scheduled_iti``: fixation duration minus
          scheduled ITI

    Raises
    ------
    FileNotFoundError
        Raised if ``path`` does not exist.
    OSError
        Raised if the log file cannot be opened.
    SyntaxError
        Raised if a parsed trial-info string cannot be interpreted by
        ``ast.literal_eval``.
    ValueError
        Raised if a matched timestamp cannot be converted to ``float``.

    Notes
    -----
    Trial boundaries are detected using ``TRIAL_RE`` and component state
    changes are detected using ``AUTODRAW_RE``. This function assumes those
    regular expressions are defined elsewhere in the module and match the
    relevant PsychoPy log lines.

    Only the first detected onset and first detected offset for each tracked
    component are stored per trial.

    The scheduled ITI stored in ``info[iti_duration_key]`` often does not
    equal ``stim_on - fixation_on`` because a trial may include additional
    fixed periods between fixation onset and stimulus onset.

    The ``trial_n`` counter increments only for retained trials. If
    ``filter_info`` excludes some trial boundaries, the numbering in the
    output will reflect the filtered subset rather than the original raw trial
    numbering in the log.

    This function does not modify the source log file.

    Examples
    --------
    Parse a log file using the default component names.

    >>> df = extract_psychopy_trial_timing("experiment.log")
    >>> isinstance(df, pd.DataFrame)
    True

    Disable pre-trial tracking.

    >>> df = extract_psychopy_trial_timing(
    ...     "experiment.log",
    ...     pretrial_component=None
    ... )
    >>> "pretrial_on" in df.columns
    True

    Keep only trials whose info dict contains selected keys.

    >>> df = extract_psychopy_trial_timing(
    ...     "experiment.log",
    ...     filter_info=["trial_type", "task"]
    ... )

    Use custom component names and metadata keys.

    >>> df = extract_psychopy_trial_timing(
    ...     "experiment.log",
    ...     fixation_component="fixation",
    ...     pretrial_component="pretrial_fix",
    ...     stim_component="image_stim",
    ...     stim_duration_key="stim_dur",
    ...     iti_duration_key="iti_dur"
    ... )
    >>> {"stim_on", "stim_off"}.issubset(df.columns)
    True
    """
    trials: List[Trial] = []
    current: Optional[Trial] = None
    trial_counter = 0

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            # 1) detect trial boundary
            m_trial = TRIAL_RE.match(line)
            if m_trial:
                t = float(m_trial.group(1))
                info = ast.literal_eval(m_trial.group(2))
                keys = set(info.keys())

                keep = (
                    filter_info is None
                    or any(k in keys for k in filter_info)
                )

                current = None
                if keep:
                    current = Trial(
                        trial_n=trial_counter,
                        trial_start=t,
                        info=info,
                    )
                    trials.append(current)
                    trial_counter += 1
                continue

            if current is None:
                continue

            # 2) detect autoDraw toggle
            m_ad = AUTODRAW_RE.match(line)
            if not m_ad:
                continue

            t = float(m_ad.group(1))
            comp = m_ad.group(2).strip()
            state = (m_ad.group(3) == "True")

            # fixation component
            if comp == fixation_component:
                if state and current.fixation_on is None:
                    current.fixation_on = t
                elif (not state) and current.fixation_on is not None and current.fixation_off is None:
                    current.fixation_off = t
                continue

            # pre-trial component
            if pretrial_component is not None and comp == pretrial_component:
                if state and current.pretrial_on is None:
                    current.pretrial_on = t
                elif (not state) and current.pretrial_on is not None and current.pretrial_off is None:
                    current.pretrial_off = t
                continue

            # stimulus component
            if comp == stim_component:
                if state and current.stim_on is None:
                    current.stim_on = t
                elif (not state) and current.stim_on is not None and current.stim_off is None:
                    current.stim_off = t
                continue

    rows = []
    for tr in trials:
        fixation_to_stim = None
        if tr.fixation_on is not None and tr.stim_on is not None:
            fixation_to_stim = tr.stim_on - tr.fixation_on

        fixation_dur = None
        if tr.fixation_on is not None and tr.fixation_off is not None:
            fixation_dur = tr.fixation_off - tr.fixation_on

        pretrial_dur = None
        if tr.pretrial_on is not None and tr.pretrial_off is not None:
            pretrial_dur = tr.pretrial_off - tr.pretrial_on

        stim_dur_logged = None
        if tr.stim_on is not None and tr.stim_off is not None:
            stim_dur_logged = tr.stim_off - tr.stim_on

        rows.append(
            {
                "trial_n": tr.trial_n,
                "trial_start": tr.trial_start,
                "task": tr.info.get("task"),
                "trial_type": tr.info.get("trial_type") or tr.info.get("type"),
                "info": tr.info,

                "fixation_on": tr.fixation_on,
                "fixation_off": tr.fixation_off,
                "fixation_dur": fixation_dur,

                "pretrial_on": tr.pretrial_on,
                "pretrial_off": tr.pretrial_off,
                "pretrial_dur": pretrial_dur,

                "stim_on": tr.stim_on,
                "stim_off": tr.stim_off,
                "stim_dur_logged": stim_dur_logged,

                "fixation_to_stim": fixation_to_stim,

                "scheduled_stim_duration": tr.info.get(stim_duration_key),
                "scheduled_iti_duration": tr.info.get(iti_duration_key),
            }
        )

    df = pd.DataFrame(rows)

    if not df.empty:
        df["next_stim_on"] = df["stim_on"].shift(-1)
        df["stim_to_next_stim"] = df["next_stim_on"] - df["stim_on"]

        df["stim_dur_error"] = (
            df["stim_dur_logged"] - df["scheduled_stim_duration"]
        )

        df["fixation_to_stim_minus_scheduled_iti"] = (
            df["fixation_to_stim"] - df["scheduled_iti_duration"]
        )

        if "fixation_dur" in df.columns and "scheduled_iti_duration" in df.columns:
            df["fixation_dur_minus_scheduled_iti"] = (
                df["fixation_dur"] - df["scheduled_iti_duration"]
            )

    return df




def extract_cs_and_isi(
        event_file: Union[str, Path]
    ) -> pd.DataFrame:
    """
    Extract CS events and compute inter-stimulus intervals (ISI) from an events file.

    This function reads a tab-separated events file, filters for conditioned
    stimulus (CS) events (e.g., ``CS+`` and ``CS−``), and computes the
    inter-stimulus interval (ISI) between consecutive CS events based on their
    onset and duration.

    Parameters
    ----------
    event_file : str or pathlib.Path
        Path to a tab-separated events file (e.g., BIDS ``events.tsv``) that
        contains at least the columns ``event_type``, ``onset``, and
        ``duration``.

    Returns
    -------
    df_cs : pandas.DataFrame
        Filtered dataframe containing only CS events, with additional columns:

        - ``ISI_marker``: inter-stimulus interval defined as:

          ``onset(current) - (onset(previous) + duration(previous))``

        The first CS event is assigned ``ISI_marker = 0``.

    Raises
    ------
    FileNotFoundError
        Raised if ``event_file`` does not exist.
    KeyError
        Raised if required columns (``event_type``, ``onset``, ``duration``)
        are missing from the file.
    pandas.errors.ParserError
        Raised if the file cannot be parsed as a tab-separated table.

    Notes
    -----
    The ISI is computed relative to the *offset* of the previous CS event,
    where:

    - offset(previous) = onset(previous) + duration(previous)

    This ensures that ISI reflects the gap between the end of one stimulus and
    the start of the next.

    The first CS event has no previous event, so its ISI is explicitly set to
    ``0``.

    The function assumes that events are already ordered chronologically in
    the input file. If this is not guaranteed, consider sorting by ``onset``
    before computing ISI.

    Examples
    --------
    Extract CS events and compute ISI.

    >>> df_cs = extract_cs_and_isi("sub-01_task-acquisition_events.tsv")
    >>> "ISI_marker" in df_cs.columns
    True

    Example input structure.

    >>> df = pd.DataFrame({
    ...     "event_type": ["CS+", "CS-", "US"],
    ...     "onset": [1.0, 5.0, 8.0],
    ...     "duration": [2.0, 2.0, 1.0]
    ... })
    >>> df.to_csv("events.tsv", sep="\\t", index=False)
    >>> out = extract_cs_and_isi("events.tsv")
    >>> out["ISI_marker"].tolist()
    [0.0, 2.0]
    """
    # 1) Keep only CS+ and CS− events
    df = pd.read_csv(event_file, delimiter="\t")
    df_cs = df[df["event_type"].str.contains("CS")].copy()

    # 2) Compute ISI based on offset of previous CS event
    #    ISI = onset(current) − (onset(previous) + duration(previous))
    df_cs["previous_offset"] = df_cs["onset"].shift(1) + df_cs["duration"].shift(1)
    df_cs["ISI_marker"] = df_cs["onset"] - df_cs["previous_offset"]

    # 3) First ISI = 0
    df_cs.loc[df_cs.index[0], "ISI_marker"] = 0

    # Optional: drop helper column
    df_cs = df_cs.drop(columns=["previous_offset"])

    return df_cs


def compare_psychopy_and_biopac_ISIs(
    dataset: str,
    raw_subject: str,
    task_name: str,
    project_root: str,
    mapper_file: str = "mapper.json",
    verbose: bool = False
):
    """
    dataset:       e.g. "bonn", "reading"
    raw_subject:   e.g. "001"
    task_name:     e.g. "acquisition" or "extinction"
    project_root:  root directory of project (where mapper.json lives)

    Returns merged dataframe: tsv_df (with ISI_marker, ISI_psychopy, ISI_diff)
    """

    # 1) Load subject ID mapping
    mapper_path = os.path.join(
        project_root,
        "converted",
        dataset,
        mapper_file
    )
    
    assert os.path.exists(mapper_path), FileExistsError(f"Mapper file {mapper_path} does not exist")

    logger.info(f"Subject mapper: {mapper_path}")
    with open(mapper_path, "r") as f:
        mapper = json.load(f)
    
    mapper_subj = f"sub-{raw_subject}"
    if mapper_subj not in mapper:
        raise KeyError(f"Subject {mapper_subj} not found in mapper.json")

    bids_subject = mapper[mapper_subj]   # e.g. "sub-CalinetReading24"

    logger.info(f"{raw_subject} >> {bids_subject}")

    # 3) Build input file paths
    # PsychoPy log file (raw sourcedata)
    base_dir = Path(project_root) / "sourcedata" / dataset
    pattern = f"*{raw_subject}*_task-{task_name}_*.log"

    log_matches = list(base_dir.rglob(pattern))
    if not log_matches:
        raise FileNotFoundError(
            f"No PsychoPy log file found under {base_dir} matching pattern: {pattern}"
        )

    if len(log_matches) > 1:
        log_event_file = max(log_matches, key=lambda p: p.stat().st_size)
        print(f"⚠️ Multiple log matches found, using the largest one: {log_event_file}")
    else:
        log_event_file = log_matches[0]

    tsv_event_file = (
        Path(project_root)
        / "converted"
        / dataset
        / bids_subject
        / "physio"
        / f"{bids_subject}_task-{task_name}_events.tsv"
    )

    if not log_event_file.exists():
        raise FileNotFoundError(f"File {log_event_file} does not exist")

    if not tsv_event_file.exists():
        raise FileNotFoundError(f"File {tsv_event_file} does not exist")

    logger.info(f"Log: '{log_event_file}'")
    logger.info(f"TSV: '{tsv_event_file}'")

    # 4) Run your processing functions
    pspy_df = extract_psychopy_trial_timing(log_event_file)
    tsv_df = extract_cs_and_isi(tsv_event_file)

    # 5) Merge ISIs (must have same number of CS entries!)
    if len(pspy_df) != len(tsv_df):
        logger.error("⚠️ WARNING: Length mismatch between PsychoPy CS events and TSV CS events!")
        logger.error(f"PsychoPy CS count = {len(pspy_df)}, TSV CS count = {len(tsv_df)}")
        raise Exception

    # Add columns
    tsv_df["ISI_psychopy"] = pspy_df["iti"].values
    tsv_df["ISI_diff"] = tsv_df["ISI_psychopy"] - tsv_df["ISI_marker"]
    
    mmin, mmax, mmean = tsv_df['ISI_diff'].min(), tsv_df['ISI_diff'].max(), tsv_df['ISI_diff'].mean()
    logger.debug(f"Minimum discrepancy: {round(mmin, 3)}s")
    logger.debug(f"Maximum discrepancy: {round(mmax, 3)}s")
    logger.debug(f"Average discrepancy: {round(mmean, 3)}s")
    logger.info("Done")

    # 6) Print + return
    return tsv_df, mmin, mmax, mmean

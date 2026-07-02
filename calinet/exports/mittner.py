# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import shutil
import numpy as np
import pandas as pd
from pathlib import Path

import pypillometry as pp
import calinet.core.io as cio

import logging
logger = logging.getLogger(__name__)

from typing import Any, Tuple

EVENT_REMAP = {
    "CSm": "CSmUSm",
    "CSpu": "CSpUSm",
    "CSpr": "CSpUSp",
}

def read_eye_physio(path: Path) -> pd.DataFrame:
    df = cio.read_physio_tsv_headerless(path)
    return df


def load_pupil_signal(
    subject_dir: Path,
    sub_id: str,
    ses_id: str,
    task_id: str = "fearconditioning",
    eye_strategy: str = "mean_available",
    input_type: str = "raw",  # "raw" or "derivative"
):
    beh_dir = subject_dir

    if input_type == "raw":
        pattern = (
            f"sub-{sub_id}_ses-{ses_id}_task-{task_id}_"
            f"recording-eye*_physio.tsv.gz"
        )
    elif input_type == "derivative":
        pattern = (
            f"sub-{sub_id}_ses-{ses_id}_task-fearconditioning_"
            f"desc-preproc_physio.tsv.gz"
        )
    else:
        raise ValueError("input_type must be 'raw' or 'derivative'")

    logger.info(
        f"Loading pupil signal for sub-{sub_id}, ses-{ses_id}, "
        f"task-{task_id}, input_type={input_type}, eye_strategy={eye_strategy}"
    )
    logger.debug(f"Searching physio files in {beh_dir} with pattern: {pattern}")

    physio_files = sorted(beh_dir.glob(pattern))
    logger.info(f"Found {len(physio_files)} physio file(s)")

    if len(physio_files) == 0:
        raise FileNotFoundError(f"No physio files found in {beh_dir} using {pattern}")

    if input_type == "derivative":
        eye_file = physio_files[0]
        
        logger.info(f"Reading derivative physio file: {eye_file}")
        df = cio.read_physio_tsv_headerless(eye_file)
        meta = df.attrs.get("metadata", {})

        logger.debug(f"Derivative metadata: {meta}")

        signal = df["pupil_size"].interpolate(limit_direction="both").to_numpy()
        sr = meta["SamplingFrequency"]

        logger.info(
            f"Loaded derivative signal with {len(signal)} samples, "
            f"SamplingFrequency={sr}"
        )

        return signal, sr, {
            "input_type": "derivative",
            "source_file": str(eye_file),
            "eye_strategy": "preprocessed",
            "available_eyes": ["preprocessed"],
            "missing_fraction": {
                "pupil_size": float(df["pupil_size"].isna().mean())
            },
            "metadata": meta,
        }

    eye_data = {}
    sampling_rates = []
    metadata = {}

    for eye_file in physio_files:
        eye_name = "eye1" if "recording-eye1" in eye_file.name else "eye2"
        logger.info(f"Reading {eye_name} physio file: {eye_file}")
        df = cio.read_physio_tsv_headerless(eye_file)

        eye_data[eye_name] = df["pupil_size"]
        metadata[eye_name] = df.attrs.get("metadata", {})
        
        logger.debug(f"{eye_name} metadata: {metadata[eye_name]}")
        logger.info(
            f"{eye_name}: {len(df)} samples, "
            f"missing pupil_size={df['pupil_size'].isna().mean():.3f}"
        )        

        sampling_rates.append(metadata[eye_name]["SamplingFrequency"])

    if len(set(sampling_rates)) != 1:
        raise ValueError(f"Sampling rates differ across eyes: {sampling_rates}")

    sr = sampling_rates[0]
    pupil_df = pd.DataFrame(eye_data)

    if eye_strategy in ["eye1", "eye2"]:
        if eye_strategy not in pupil_df:
            raise ValueError(f"{eye_strategy} not found. Available: {list(pupil_df.columns)}")
        signal = pupil_df[eye_strategy]

    elif eye_strategy == "best_metadata":
        best_eye = None

        for eye_name, meta in metadata.items():
            recorded_eye = str(meta.get("RecordedEye", "")).lower()
            best = str(meta.get("BestEye", "")).lower()

            if best in ["l", "left"] and recorded_eye == "left":
                best_eye = eye_name
            elif best in ["r", "right"] and recorded_eye == "right":
                best_eye = eye_name

        if best_eye is None:
            best_eye = pupil_df.isna().mean().idxmin()

        signal = pupil_df[best_eye]

    elif eye_strategy == "mean_available":
        signal = pupil_df.mean(axis=1, skipna=True)

    elif eye_strategy == "mean_complete":
        signal = pupil_df.where(pupil_df.notna().all(axis=1)).mean(axis=1)

    else:
        raise ValueError(
            "Unknown eye_strategy. Use one of: "
            "'eye1', 'eye2', 'best_metadata', 'mean_available', 'mean_complete'"
        )

    signal = signal.interpolate(limit_direction="both").to_numpy()

    return signal, sr, {
        "input_type": "raw",
        "eye_strategy": eye_strategy,
        "available_eyes": list(pupil_df.columns),
        "missing_fraction": pupil_df.isna().mean().to_dict(),
    }


def process_session(
    subject_dir: Path,
    sub_id: str,
    ses_id: str,
    task_id: str = "fearconditioning",
    phys_dir: str = "beh",
    event_col: str = "event_type",
    lowpass_cutoff: float = 5,
    fsd: int = 50,
    baseline_lp: float = 0.25,
    lam_min: float = 1.0,
    lam_max: float = 100.0,
    lam_sig: float = 1.0,
    verbose: int = 100,
    eye_strategy: str = "mean_available",
    input_type: str = "raw",
    output_dir: Any = None 
) -> Tuple[pd.DataFrame, Any]:

    subject_dir = Path(subject_dir)
    beh_dir = subject_dir / f"ses-{ses_id}" / phys_dir
    if output_dir is None:
        stan_output_dir = beh_dir / "cmdstan"
    else:
        stan_output_dir = Path(output_dir) / subject_dir.stem / f"ses-{ses_id}"

    if input_type == "raw":
        event_name = f"sub-{sub_id}_ses-{ses_id}_task-{task_id}_events.tsv"
    elif input_type == "derivative":
        event_name = f"sub-{sub_id}_ses-{ses_id}_task-{task_id}_desc-preproc_events.tsv"
    else:
        raise ValueError("input_type must be 'raw' or 'derivative'")

    event_file = beh_dir / event_name
    if not event_file.exists():
        raise FileNotFoundError(event_file)

    logger.info(
        f"Processing session sub-{sub_id}, ses-{ses_id}, "
        f"input_type={input_type}, phys_dir={phys_dir}"
    )
    logger.info(f"Reading events file: {event_file}")

    df_event = pd.read_csv(event_file, sep="\t")
    logger.debug(f"Loaded {len(df_event)} events")

    event_col = event_col if event_col in df_event.columns else "trial_type"
    logger.debug(f"Using event column: {event_col}")

    df_event = df_event[~df_event[event_col].str.contains("USp", na=False)]
    logger.debug(f"Events after removing USp: {len(df_event)}")

    relevant_events = df_event.loc[
        df_event[event_col].isin(["CSpu", "CSpr", "CSm"])
    ].copy()
    logger.info(f"Relevant CS events: {len(relevant_events)}")

    signal, sr, _ = load_pupil_signal(
        beh_dir,
        sub_id,
        ses_id,
        eye_strategy=eye_strategy,
        input_type=input_type,
    )

    pup = pp.PupilData(
        signal,
        sampling_rate=sr,
        event_onsets=relevant_events["onset"].to_numpy() * 1000,
        event_labels=relevant_events[event_col].to_numpy(),
    )

    # save stan output in unique folder
    if stan_output_dir.exists():
        shutil.rmtree(stan_output_dir)

    stan_output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Stan output path: {str(stan_output_dir)}")
    logger.info(
        f"Running pypillometry: lowpass={lowpass_cutoff}, fsd={fsd}, "
        f"baseline_lp={baseline_lp}, lam_min={lam_min}, "
        f"lam_max={lam_max}, lam_sig={lam_sig}"
    )

    d = (
        pup.lowpass_filter(cutoff=lowpass_cutoff)
           .downsample(fsd=fsd)
           .estimate_baseline(
                lp=baseline_lp,
                lam_min=lam_min,
                lam_max=lam_max,
                lam_sig=lam_sig,
                verbose=verbose,
                stan_kwargs={
                    "output_dir": str(stan_output_dir),
                    "show_console": False,
                },
           )
           .estimate_response()
    )

    # check variance of baseline and signal
    baseline_var = np.var(d.baseline)
    signal_var = np.var(d.sy)

    logger.info(f"Baseline variance / Signal variance: {baseline_var / signal_var:.4f}; baseline = {(baseline_var/signal_var)*100:.2f}% of signal variance")

    traditional_baseline = d.stat_per_event((-200, 0))
    traditional_response = d.stat_per_event((800, 1200)) - traditional_baseline
    novel_response = d.response_pars["coef"]

    if not (
        len(traditional_response)
        == len(novel_response)
        == len(relevant_events)
    ):
        raise ValueError(
            "Mismatch between events and response estimates: "
            f"{len(relevant_events)=}, "
            f"{len(traditional_response)=}, "
            f"{len(novel_response)=}"
        )
    
    # print result
    tmp = pd.DataFrame({
        "condition": relevant_events[event_col].to_numpy(),
        "traditional": traditional_response,
        "mittner": novel_response,
    })

    logger.info("Condition-wise response estimates:")

    for cond, g in tmp.groupby("condition"):
        logger.info(
            f" {cond:4s} "
            f"(n={len(g):2d}) | "
            f"traditional ={g['traditional'].mean():8.4f} ± {g['traditional'].std():7.4f} | "
            f"mittner ={g['mittner'].mean():8.4f} ± {g['mittner'].std():7.4f}"
        )

    trial_df = pd.DataFrame({
        "subject": f"sub-{sub_id}",
        "session": f"ses-{ses_id}",
        "event_type": relevant_events[event_col].to_numpy(),
        "condition": relevant_events[event_col].map(EVENT_REMAP).to_numpy(),
        "traditional": traditional_response,
        "mittner": novel_response,
    })

    logger.info(f"Finished session sub-{sub_id}, ses-{ses_id}")

    return trial_df, d


def process_subject(
    input_dir: Path | str,
    output_dir: Path | str | None = None,
    **kwargs
) -> tuple[pd.DataFrame, pd.DataFrame]:

    input_dir = Path(input_dir)
    sub_id = input_dir.name.replace("sub-", "")

    logger.info(f"Processing subject directory: {input_dir}")
    logger.info(f"Writing outputs to: {output_dir}")

    if output_dir is None:
        output_dir = input_dir.parent / "derivatives"
        stan_dir = output_dir / "cmdstan"
    else:
        output_dir = Path(output_dir)
        stan_dir = output_dir.parent.parent / "cmdstan"

    output_dir.mkdir(parents=True, exist_ok=True)

    session_ids = sorted(
        p.name.replace("ses-", "")
        for p in input_dir.glob("ses-*")
        if p.is_dir()
    )

    logger.info(f"Found sessions: {session_ids}")
    all_trial_dfs = []

    for ses_id in session_ids:
        ses_df, _ = process_session(
            input_dir,
            sub_id,
            ses_id,
            output_dir=stan_dir,
            **kwargs
        )
        all_trial_dfs.append(ses_df)

    trial_df = pd.concat(all_trial_dfs, ignore_index=True)

    long_df = trial_df.melt(
        id_vars=["subject", "session", "event_type", "condition"],
        value_vars=["traditional", "mittner"],
        var_name="method_name",
        value_name="response",
    )
    
    # mean/median aggregation for each condition and method
    agg_dfs = []

    for agg_name, agg_fun in [
        ("mean", "mean"),
        ("median", "median"),
    ]:

        tmp = (
            long_df
            .groupby(
                ["subject", "method_name", "condition"],
                as_index=False,
            )["response"]
            .agg(agg_fun)
            .pivot(
                index=["subject", "method_name"],
                columns="condition",
                values="response",
            )
            .reset_index()
        )

        tmp = tmp.rename_axis(None, axis=1)

        for col in ["CSmUSm", "CSpUSm", "CSpUSp"]:
            if col not in tmp.columns:
                tmp[col] = np.nan

        tmp["aggregation"] = agg_name

        tmp = tmp[
            [
                "subject",
                "method_name",
                "aggregation",
                "CSmUSm",
                "CSpUSm",
                "CSpUSp",
            ]
        ]

        agg_dfs.append(tmp)

    short_df = pd.concat(agg_dfs, ignore_index=True)

    trial_file = output_dir / f"sub-{sub_id}_desc-pypillometry_trialwise.csv"
    short_file = output_dir / f"sub-{sub_id}_desc-pypillometry.csv"

    logger.info(f"Writing trialwise output: {trial_file}")
    trial_df.to_csv(trial_file, index=False)

    logger.info(f"Writing summary output: {short_file}")
    short_df.to_csv(short_file, index=False)

    return trial_df, short_df


def load_dataset_overview(path: str | Path) -> pd.DataFrame:
    overview = pd.read_csv(path)
    required = {"Corpus_ID", "Dataset"}
    missing = required - set(overview.columns)
    if missing:
        raise ValueError(f"Overview file missing columns: {missing}")
    return overview


def select_tuning_subjects(
    overview: pd.DataFrame,
    root_dir: str | Path,
    n_per_dataset: int = 8,
    random_state: int = 42,
):
    root_dir = Path(root_dir)

    selected = {}

    for dataset, df in overview.groupby("Dataset"):
        sample = df.sample(
            n=min(n_per_dataset, len(df)),
            random_state=random_state,
        )

        subject_dirs = [
            root_dir / f"sub-{sub.replace('sub-', '')}"
            if not str(sub).startswith("sub-")
            else root_dir / sub
            for sub in sample["Corpus_ID"]
        ]

        selected[dataset] = [p for p in subject_dirs if p.exists()]

    return selected

def tune_dataset(
    dataset_name: str,
    subject_dirs: list[Path],
    args,
    lam_sig_grid=[0.1, 1.0, 10.0],
    lp_grid=[0.10, 0.25, 0.50],
):
    rows = []

    for lp in lp_grid:
        for lam_sig in lam_sig_grid:
            for subject_dir in subject_dirs:
                try:
                    trial_df, short_df = process_subject(
                        subject_dir,
                        output_dir=Path(args.output_dir) / "tuning" / dataset_name / subject_dir.name,
                        eye_strategy=args.eye_strategy,
                        input_type=args.input_type,
                        event_col=args.event_col,
                        lowpass_cutoff=args.lowpass_cutoff,
                        fsd=args.fsd,
                        baseline_lp=lp,
                        lam_sig=lam_sig,
                        verbose=args.verbose,
                    )

                    rows.append({
                        "dataset": dataset_name,
                        "subject": subject_dir.name,
                        "lp": lp,
                        "lam_sig": lam_sig,
                        "status": "ok",
                        "n_trials": len(trial_df),
                        "mittner_sd": trial_df["mittner"].std(),
                        "traditional_sd": trial_df["traditional"].std(),
                        "mittner_mean": trial_df["mittner"].mean(),
                    })

                except Exception as e:
                    rows.append({
                        "dataset": dataset_name,
                        "subject": subject_dir.name,
                        "lp": lp,
                        "lam_sig": lam_sig,
                        "status": "failed",
                        "error": str(e),
                    })

    return pd.DataFrame(rows)

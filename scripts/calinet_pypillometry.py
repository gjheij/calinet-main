# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import time
import logging
import argparse
import pandas as pd

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from calinet.logger import init_logging
from calinet.exports.mittner import (
    process_subject,
    tune_dataset,
    load_dataset_overview,
    select_tuning_subjects
)



def find_subject_dirs(root_dir: str | Path):
    root_dir = Path(root_dir)
    return sorted([p for p in root_dir.glob("sub-*") if p.is_dir()])


def run_one_subject(subject_dir: Path, args):
    subject_out = Path(args.output_dir) / subject_dir.name
    subject_out.mkdir(parents=True, exist_ok=True)

    trial_df, short_df = process_subject(
        subject_dir,
        output_dir=subject_out,
        eye_strategy=args.eye_strategy,
        input_type=args.input_type,
        event_col=args.event_col,
        lowpass_cutoff=args.lowpass_cutoff,
        fsd=args.fsd,
        baseline_lp=args.baseline_lp,
        lam_min=args.lam_min,
        lam_max=args.lam_max,
        lam_sig=args.lam_sig,
        verbose=args.verbose
    )

    return subject_dir.name, trial_df, short_df


def read_failed_subjects_file(path: str | Path):
    path = Path(path)
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip()
    ]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Run the pypillometry preprocessing pipeline on a single subject "
            "directory and save the resulting trial-level and summary outputs."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        type=str,
        help=(
            "Subject directory.\n\n"
            "Examples:\n"
            "  Z:\\PupilFear\\Corpus\\sub-001\n"
            "  Z:\\PupilFear\\Corpus\\derivatives\\pspm\\sub-001"
        ),
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=str,
        help="Directory where pypillometry outputs will be written.",
    )

    parser.add_argument(
        "--eye-strategy",
        default="mean_available",
        choices=[
            "eye1",
            "eye2",
            "best_metadata",
            "mean_available",
            "mean_complete",
        ],
        help="Eye selection strategy.",
    )

    parser.add_argument(
        "--input-type",
        default="raw",
        choices=["raw", "derivative"],
        help="Input dataset type.",
    )

    parser.add_argument(
        "--event-col",
        default="event_type",
        choices=["event_type", "trial_type"],
        help="Column containing event labels.",
    )

    parser.add_argument(
        "--lowpass-cutoff",
        type=float,
        default=5.0,
        help="Low-pass filter cutoff (Hz).",
    )

    parser.add_argument(
        "--fsd",
        type=int,
        default=50,
        help="Downsampled sampling frequency.",
    )

    parser.add_argument(
        "--baseline-lp",
        type=float,
        default=0.25,
        help="Baseline low-pass cutoff.",
    )

    parser.add_argument(
        "--lam-min",
        type=float,
        default=1.0,
        help="Minimum smoothing lambda.",
    )

    parser.add_argument(
        "--lam-max",
        type=float,
        default=100.0,
        help="Maximum smoothing lambda.",
    )

    parser.add_argument(
        "--lam-sig",
        type=float,
        default=1.0,
        help="Signal smoothing lambda.",
    )

    parser.add_argument(
        "--verbose",
        type=int,
        default=100,
        help="Verbosity passed to process_subject.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )

    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of subjects to process in parallel.",
    )

    parser.add_argument(
        "--failed-subjects-file",
        type=str,
        default=None,
        help="Optional txt file with one subject ID per line, e.g. sub-001.",
    )

    parser.add_argument(
        "--append-group-outputs",
        action="store_true",
        help="Append current outputs to existing group-level pypillometry CSVs.",
    )

    parser.add_argument(
        "--overview-file",
        type=str,
        default=None,
        help="CSV with Corpus_ID and Dataset columns.",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    log_file = os.path.join(args.output_dir, "log.log")
    log_level = logging.DEBUG if args.debug else logging.INFO
    init_logging(level=log_level, logfile=log_file)

    cmdstan_logger = logging.getLogger("cmdstanpy")
    cmdstan_logger.handlers.clear()
    cmdstan_logger.propagate = True
    cmdstan_logger.setLevel(logging.WARNING)

    logger = logging.getLogger("calinet.pypillometry")

    input_dir = Path(args.input_dir)
    logger.info(f"Input: {args.input_dir}")
    logger.info(f"Output: {args.output_dir}")

    n_jobs = args.n_jobs
    input_dir = Path(args.input_dir)

    if args.failed_subjects_file is not None:
        subject_ids = read_failed_subjects_file(args.failed_subjects_file)

        if input_dir.name.startswith("sub-"):
            root_dir = input_dir.parent
        else:
            root_dir = input_dir

        subject_dirs = [root_dir / sub for sub in subject_ids]
        subject_dirs = [p for p in subject_dirs if p.exists()]

    elif input_dir.name.startswith("sub-"):
        subject_dirs = [input_dir]
        n_jobs = 1
    else:
        subject_dirs = find_subject_dirs(input_dir)

    logger.info(f"Using {n_jobs} parallel jobs")
    logger.info(f"Found {len(subject_dirs)} subjects")

    start_time = time.perf_counter()

    all_trial_dfs = []
    all_short_dfs = []
    failed_subjects = []

    if n_jobs == 1:
        for subject_dir in subject_dirs:
            logger.info(f"Processing {subject_dir.name}")

            try:
                sub, trial_df, short_df = run_one_subject(subject_dir, args)

                all_trial_dfs.append(trial_df)
                all_short_dfs.append(short_df)

                logger.info(
                    f"{sub}: trial_df={trial_df.shape}, short_df={short_df.shape}"
                )

            except Exception:
                failed_subjects.append(subject_dir.name)
                logger.exception(f"Failed processing {subject_dir.name}")

    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futures = {
                ex.submit(run_one_subject, subject_dir, args): subject_dir
                for subject_dir in subject_dirs
            }

            for fut in as_completed(futures):
                subject_dir = futures[fut]

                try:
                    sub, trial_df, short_df = fut.result()

                    all_trial_dfs.append(trial_df)
                    all_short_dfs.append(short_df)

                    logger.info(
                        f"{sub}: trial_df={trial_df.shape}, short_df={short_df.shape}"
                    )

                except Exception:
                    failed_subjects.append(subject_dir.name)
                    logger.exception(f"Failed processing {subject_dir.name}")

    if len(all_trial_dfs) > 0:
        new_trial_df = pd.concat(all_trial_dfs, ignore_index=True)
        new_short_df = pd.concat(all_short_dfs, ignore_index=True)

        group_trial_file = Path(args.output_dir) / "pypillometry_trialwise_all_subjects.csv"
        group_short_file = Path(args.output_dir) / "pypillometry_short_all_subjects.csv"

        if args.append_group_outputs:
            if group_trial_file.exists():
                old_trial_df = pd.read_csv(group_trial_file)
                group_trial_df = pd.concat([old_trial_df, new_trial_df], ignore_index=True)
            else:
                group_trial_df = new_trial_df

            if group_short_file.exists():
                old_short_df = pd.read_csv(group_short_file)
                group_short_df = pd.concat([old_short_df, new_short_df], ignore_index=True)
            else:
                group_short_df = new_short_df

            # remove replaced/reprocessed rows
            if "subject" in group_trial_df.columns:
                group_trial_df = group_trial_df.drop_duplicates(
                    subset=[c for c in ["subject", "session", "event_type", "traditional", "mittner"] if c in group_trial_df.columns],
                    keep="last",
                )

            if "subject" in group_short_df.columns:
                group_short_df = group_short_df.drop_duplicates(
                    subset=[c for c in ["subject", "method_name", "aggregation"] if c in group_short_df.columns],
                    keep="last",
                )
        else:
            group_trial_df = new_trial_df
            group_short_df = new_short_df

        # sort by subject and reset index before saving
        group_trial_df = group_trial_df.sort_values(
            ["subject", "session", "event_type"]
        ).reset_index(drop=True)

        group_short_df = group_short_df.sort_values(
            ["subject", "method_name", "aggregation"]
        ).reset_index(drop=True)

        # save group-level outputs
        group_trial_df.to_csv(group_trial_file, index=False)
        group_short_df.to_csv(group_short_file, index=False)

    if len(failed_subjects) > 0:
        failed_file = Path(args.output_dir) / "pypillometry_failed_subjects.txt"
        if failed_file.exists():
            failed_file.unlink()
        
        if failed_subjects:
            logger.warning(f"Failed subjects: {failed_subjects}")
            failed_file.write_text("\n".join(failed_subjects) + "\n")
        else:
            logger.info("No failed subjects.")

    elapsed = time.perf_counter() - start_time
    logger.info(f"Processed {len(subject_dirs)} subjects in {elapsed:.2f} seconds")
    
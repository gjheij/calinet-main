#!/usr/bin/env python3
"""CLI wrapper around ``run_pspm_trim_directory``.

Given a subject folder such as ``sub-CalinetAmsterdam01``, this script:

1. runs PsPM-style trimming on that folder;
2. recreates the QA overview plot in::

       <subject-parent>/derivatives/qa/
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

from calinet.plotting import _generate_single_qa_plot
from calinet.data import run_pspm_trim_directory
from calinet.logger import init_logging


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Trim one CALINET subject directory with run_pspm_trim_directory "
            "and regenerate its QA overview plot."
        )
    )
    p.add_argument(
        "input_folder",
        type=Path,
        help="Subject folder, e.g. sub-CalinetAmsterdam01",
    )
    p.add_argument(
        "--from",
        dest="from_",
        required=True,
        help="Trim start passed to run_pspm_trim_directory (numeric or 'none').",
    )
    p.add_argument(
        "--to",
        dest="to",
        required=True,
        help="Trim end passed to run_pspm_trim_directory (numeric or 'none').",
    )
    p.add_argument(
        "--reference",
        default="marker",
        help=(
            "Reference passed to run_pspm_trim_directory. Examples: marker, file. "
            "Complex references can be passed as a Python literal with --reference-literal."
        ),
    )
    p.add_argument(
        "--reference-literal",
        action="store_true",
        help="Interpret --reference as a Python literal, e.g. '(0, 1)' or '(\"start\", \"end\")'.",
    )
    p.add_argument("--fs-fallback", type=float, default=None)
    p.add_argument("--timestamp-col", default="timestamp")
    p.add_argument("--event-time-col", default="onset")
    p.add_argument("--event-name-col", default="name")
    p.add_argument("--event-value-col", default="value")
    p.add_argument("--drop-offset-markers", action="store_true")
    p.add_argument("--prefix", default="t")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--filter",
        dest="physio_filters",
        action="append",
        default=[],
        help=(
            "Filename substring filter for physio discovery. "
            "Repeat this flag to require multiple substrings, e.g. "
            "--filter task-acquisition --filter recording-scr"
        ),
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
    )
    return p


def _coerce_trim_arg(value: str) -> Any:
    """Convert CLI string to float where possible, else keep string.

    This preserves tokens like 'none' while allowing plain numerics.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _coerce_reference(value: str, parse_literal: bool) -> Any:
    if not parse_literal:
        return value

    import ast

    try:
        return ast.literal_eval(value)
    except Exception as exc:
        raise ValueError(f"Could not parse --reference literal: {value!r}") from exc


def main() -> int:
    args = build_argparser().parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    input_folder = args.input_folder.resolve()
    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_folder}")
    if not input_folder.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_folder}")

    dataset_name = os.path.basename(os.path.dirname(input_folder))
    subject = os.path.basename(input_folder)
    qa_dir = os.path.join(os.path.dirname(input_folder), "derivatives", "qa")
    os.makedirs(qa_dir, exist_ok=True)

    init_logging(level=args.log_level)
    logger = logging.getLogger("calinet.trimming")

    logger.info("Input folder: %s", input_folder)
    logger.info("dataset_name: %s", dataset_name)
    logger.info("subject: %s", subject)
    logger.info("qa_dir: %s", qa_dir)

    from_ = _coerce_trim_arg(args.from_)
    to = _coerce_trim_arg(args.to)
    reference = _coerce_reference(args.reference, args.reference_literal)

    results = run_pspm_trim_directory(
        root_dir=input_folder,
        from_=from_,
        to=to,
        reference=reference,
        fs_fallback=args.fs_fallback,
        timestamp_col=args.timestamp_col,
        event_time_col=args.event_time_col,
        event_name_col=args.event_name_col,
        event_value_col=args.event_value_col,
        drop_offset_markers=args.drop_offset_markers,
        prefix=args.prefix,
        overwrite=args.overwrite,
        filters=args.physio_filters
    )

    ok = [r for r in results if r.get("status") == "ok"]
    skipped = [r for r in results if r.get("status") == "skipped"]
    errors = [r for r in results if r.get("status") == "error"]

    logger.info(
        "Trim complete: %d ok | %d skipped | %d errors",
        len(ok),
        len(skipped),
        len(errors),
    )

    finished_subject, fname = _generate_single_qa_plot(
        dataset_name=dataset_name,
        subject=subject,
        qa_dir=qa_dir,
    )

    logger.info("QA plot regenerated for %s", finished_subject)
    logger.info("QA plot path: %s", fname)

    if errors:
        for err in errors:
            logger.error("Trim error for %s: %s", err.get("physio_file"), err.get("error"))
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

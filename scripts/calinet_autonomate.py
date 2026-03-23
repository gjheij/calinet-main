# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import logging
import argparse
from pathlib import Path

from calinet.logger import init_logging
from calinet.exports.autonomate import convert_dataset_to_autonomate


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Convert a BIDS/CALINET physio dataset to Autonomate-compatible "
            "ASCII text files while preserving BIDS-style subject/task naming."
        )
    )

    parser.add_argument(
        "--subjects-tsv",
        type=str,
        default=None,
        help=(
            "Optional TSV file listing subjects to export. "
            "When provided for a multi-site container, output is flattened by subject."
        ),
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Input project directory, e.g. 'Z:\\CALINET2\\derivatives\\blinded\\austin'",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Output directory to save Autonomate files. "
            "If omitted, outputs are written under derivatives/autonomate/<dataset-name>."
        ),
    )

    parser.add_argument(
        "--first",
        type=int,
        default=None,
        help="Only include the first N subjects to test functionality.",
    )

    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help=(
            "Task name to export, e.g. 'acquisition' or 'extinction'. "
            "Default: both."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing Autonomate TXT files if they already exist.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Enable verbose debug-level logging.\n\n"
            "When set, the logger runs at logging.DEBUG instead of logging.INFO."
        ),
    )

    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory does not exist: '{input_dir}'")

    if args.output_dir is None:
        input_path = Path(input_dir)
        output_dir = input_path / ".." / ".." / "derivatives" / "autonomate" / input_path.stem
    else:
        output_dir = Path(args.output_dir)

    output_dir = os.path.abspath(str(output_dir))
    os.makedirs(output_dir, exist_ok=True)

    log_file = os.path.join(output_dir, "log.log")

    log_level = logging.DEBUG if args.debug else logging.INFO
    init_logging(level=log_level, logfile=log_file)

    logger = logging.getLogger("calinet.autonomate")
    logger.info(f"Log-file: {log_file}")
    logger.info(f"Input dataset: {input_dir}")
    logger.info(f"Saving Autonomate dataset to: {output_dir}")
    logger.info(
        "Output format preserves BIDS-style filenames, but writes plain-text "
        "Autonomate files (*.txt) instead of compressed physio TSV files (*.tsv.gz)."
    )

    convert_dataset_to_autonomate(
        input_dir=input_dir,
        output_dir=output_dir,
        modality='scr',
        overwrite=args.overwrite,
        include_n=args.first,
        subjects_tsv=args.subjects_tsv,
        task_name=args.task
    )
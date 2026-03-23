# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import logging
import argparse
from pathlib import Path

from calinet.logger import init_logging
from calinet.exports.ezyscr import convert_dataset_to_ezyscr


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Convert blinded CALINET data to EzySCR format."
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
        help="Input project directory, e.g. 'Z:\\CALINET2\\converted\\austin'",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory to save individual subject files (zip files). Defaults to 'Z:\\CALINET2\\derivatives\\ezyscr\\<basename input_dir>'",
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
        "--first",
        type=int,
        default=None,
        help="Only include the first N subjects to test functionality",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug-level logging",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing MAT files",
    )

    args = parser.parse_args()

    raw_data_dir = os.path.abspath(args.input_dir)
    if not os.path.exists(raw_data_dir):
        raise FileNotFoundError(f"Input directory does not exist: '{raw_data_dir}'")

    output_dir = args.output_dir
    if output_dir is None:
        input_path = Path(raw_data_dir)
        output_dir = input_path / ".." / ".." / "derivatives" / "ezyscr" /input_path.stem

    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    log_file = os.path.join(output_dir, "log.log")

    log_level = logging.DEBUG if args.debug else logging.INFO
    init_logging(level=log_level, logfile=log_file)

    logger = logging.getLogger("calinet.ezyscr")
    logger.info(f"Log-file: {log_file}")
    logger.info(f"Input dataset: {raw_data_dir}")
    logger.info(f"Output EzySCR directory: {output_dir}")

    convert_dataset_to_ezyscr(
        input_dir=raw_data_dir,
        output_dir=output_dir,
        modality='scr',
        overwrite=args.overwrite,
        include_n=args.first,
        subjects_tsv=args.subjects_tsv,
        task_name=args.task
    )

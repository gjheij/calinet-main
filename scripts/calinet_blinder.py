# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import logging
import argparse
from pathlib import Path

from calinet.logger import init_logging
from calinet.exports.blinder import blind_dataset


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Create a blinded full-copy of a CALINET project."
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
        help="Output directory to save blinded subject files.",
    )

    parser.add_argument(
        "--first",
        type=int,
        default=None,
        help="Only include the first N subjects to test functionality. Ignored when --subjects-tsv is used.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Enable verbose debug-level logging.\n\n"
            "When set, the logger runs at logging.DEBUG instead of logging.INFO."
        ),
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
        "--modalities",
        nargs="+",
        default=None,
        help=(
            "Only include selected recording modalities, e.g. "
            "--modalities scr eye2 resp. "
            "If omitted, all modalities are included."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing blinded output directory",
    )


    parser.add_argument(
        "--skip-blinding",
        action="store_true",
        help="Do not blind the stimulus types, just copy as is to send out as test samples",
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

    logger = logging.getLogger("calinet.blinder")
    logger.info(f"Log-file: {log_file}")
    logger.info(f"Raw dataset: {input_dir}")
    logger.info(f"Saving blinded dataset to: {output_dir}")

    blind_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        include_n=args.first,
        modalities=args.modalities,
        subjects_tsv=args.subjects_tsv,
        task_name=args.task,
        skip_blinding=args.skip_blinding
    )

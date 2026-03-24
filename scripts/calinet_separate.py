# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import logging
import argparse

from calinet.logger import init_logging
from calinet.exports.separator import separate_and_zip


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Arguments for the Subject Separator.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Input directory for converted BIDS data (e.g., 'Z:\\CALINET2\\converted\\austin')",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="Z:\\CALINET2\\derivatives\\calibench",
        help="Output directory to save individual subject files (zip files). Defaults to 'Z:\\CALINET2\\derivatives\\calibench'",
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
        "--first",
        type=int,
        default=None,
        help="Only include the first N subjects to test the functionality",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Enable verbose debug-level logging.\n\n"
            "When set, the logger runs at ``logging.DEBUG`` instead of "
            "``logging.INFO``. This produces substantially more detailed diagnostic "
            "output, which is useful for troubleshooting file discovery, conversion "
            "steps, metadata handling, and worker behavior."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Overwrite existing zip-files"
        ),
    )    

    args = parser.parse_args()

    raw_data_dir = args.input_dir
    if raw_data_dir is not None:
        if not os.path.exists(raw_data_dir):
            raise Exception(f"Input directory does not exist")
    else:
        raise Exception(f"You must specify an input folder (e.g., --input-dir 'Z:\\CALINET2\\converted\\austin')")

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # define log file | will be overwritten
    log_file = os.path.join(output_dir, "log.log")

    log_level = logging.DEBUG if args.debug else logging.INFO
    init_logging(level=log_level, logfile=log_file)

    logger = logging.getLogger("calinet.separate")
    logger.info(f"Log-file: {log_file}")
    logger.info(f"Raw dataset: {raw_data_dir}")
    logger.info(f"Saving separated & zipped datasets to: {output_dir}")

    separate_and_zip(
        raw_data_dir,
        output_dir,
        overwrite=args.overwrite,
        include_n=args.first,
        modalities=args.modalities
    )
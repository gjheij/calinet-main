# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import time
import logging
import argparse
from calinet.logger import init_logging
from calinet.core.dataset import convert_data
from calinet.utils import (
    cleanup_logs,
    merge_log_files,
    merge_worker_logs
)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Convert a lab-specific CALINET raw dataset into a BIDS-like converted "
            "dataset. The command scans the input directory, applies the lab-specific "
            "conversion logic implemented in ``convert_data``, writes the converted "
            "files to the output directory, and creates run logs for traceability.\n\n"
            "Typical usage:\n"
            "  python -m calinet.convert --input-dir Z:\\CALINET2\\sourcedata\\bielefeld\n"
            "  python -m calinet.convert --input-dir Z:\\CALINET2\\sourcedata\\bielefeld "
            "--output-dir Z:\\CALINET2\\converted\\bielefeld --clean\n"
            "  python -m calinet.convert --input-dir Z:\\CALINET2\\sourcedata\\bielefeld "
            "--n-workers 4 --debug\n\n"
            "Output directory behavior:\n"
            "  - If ``--output-dir`` is omitted and ``--input-dir`` contains the path "
            "    segment ``sourcedata``, the output directory is derived automatically "
            "    by replacing ``sourcedata`` with ``converted``.\n"
            "  - If the output directory already exists and is not empty, the command "
            "    stops unless ``--clean`` is provided.\n\n"
            "Logging behavior:\n"
            "  - A temporary main log is written to ``log_tmp.log`` inside the output "
            "    directory.\n"
            "  - If multiple workers are used, worker-specific logs are merged into "
            "    ``log_merged.log``.\n"
            "  - A final merged chronological log is written to ``log.log``.\n"
            "  - Intermediate log files are cleaned up at the end of a successful run."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help=(
            "Path to the raw source dataset for a single lab.\n\n"
            "This directory must already exist. It is usually a folder inside the "
            "CALINET ``sourcedata`` tree, for example:\n"
            "  Z:\\CALINET2\\sourcedata\\bielefeld\n\n"
            "The converter uses this directory as the root for discovering subject "
            "folders, raw physiology files, event files, and other lab-specific input "
            "resources required by ``convert_data``."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Destination directory for the converted BIDS-like dataset.\n\n"
            "Example:\n"
            "  Z:\\CALINET2\\converted\\bielefeld\n\n"
            "If omitted, the program tries to derive the output path automatically by "
            "replacing the path component ``sourcedata`` in ``--input-dir`` with "
            "``converted``. If automatic derivation is not possible, you must provide "
            "``--output-dir`` explicitly.\n\n"
            "By default, the command refuses to proceed if this directory already "
            "exists and is not empty. Use ``--clean`` to allow re-creation of the "
            "entire output directory contents."
        ),
    )

    parser.add_argument(
        "--clean",
        action="store_true",
        help=(
            "Allow conversion into an existing output directory by cleaning and "
            "rebuilding its contents.\n\n"
            "Use this flag when you want to rerun conversion and overwrite previous "
            "outputs. Without this flag, the command raises an error if the output "
            "directory already exists and is not empty.\n\n"
            "Caution:\n"
            "  This may remove previously generated converted files in the output "
            "  directory as part of the conversion workflow."
        ),
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
        "--n-workers",
        type=int,
        default=1,
        help=(
            "Number of parallel worker processes used for subject-level conversion.\n\n"
            "Use ``1`` for serial execution. Values greater than ``1`` enable "
            "multiprocessing, which can speed up conversion for datasets with many "
            "subjects.\n\n"
            "Examples:\n"
            "  --n-workers 1    Run in a single process\n"
            "  --n-workers 4    Run up to four worker processes in parallel\n\n"
            "When multiprocessing is enabled, worker-specific log files are created "
            "and merged automatically after processing."
        ),
    )

    # parse arguments
    args = parser.parse_args()

    # check if input exists
    input_dir = args.input_dir
    if not isinstance(input_dir, str):
        raise ValueError(f"--input-dir must point to a string")
    else:
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"The path specified with --input-dir does not exist")


    # check output dir
    output_dir = args.output_dir
    if output_dir is None:
        if "sourcedata" in input_dir:
            output_dir = input_dir.replace("sourcedata", "converted")
        else:
            raise ValueError(f"Please specify an output folder with --output-dir")

    # makedir
    if not args.clean:
        if os.path.exists(output_dir):
            if len(output_dir)>0:
                raise Exception (f"{output_dir} is not empty. Use --clean to overwrite entire directory")
        
    # make directory
    os.makedirs(output_dir, exist_ok=True)
    
    # define log file | will be overwritten
    log_file = os.path.join(output_dir, "log_tmp.log")

    log_level = logging.DEBUG if args.debug else logging.INFO
    init_logging(level=log_level, logfile=log_file)

    logger = logging.getLogger("calinet.main")
    logger.info(f"Log-file: {log_file}")

    if args.debug:
        logger.debug("Debug mode enabled")

    # make output dir
    logger.info(f"Raw dataset: {input_dir}")
    logger.info(f"Saving converted dataset to: {output_dir}")
    logger.info(f"Creating base output directory {output_dir}")

    start_time = time.perf_counter()

    worker_log = os.path.join(output_dir, "log_merged.log")
    final_log = os.path.join(output_dir, "log.log")

    try:
        convert_data(
            input_dir,
            output_dir,
            clean=args.clean,
            log_file=log_file,
            n_workers=args.n_workers,
        )

        # Only create a merged worker log if multiprocessing was actually used
        if args.n_workers and args.n_workers > 1:
            merge_worker_logs(output_dir, worker_log, remove_worker_logs=True)
            logger.info(f"Merged worker logs written to: {worker_log}")
        else:
            logger.info("Single-worker run detected; skipping worker log merge")

    except Exception as e:
        logger.exception(f"convert_data failed: {e}")

    else:
        elapsed = time.perf_counter() - start_time
        logger.info(f"Total runtime: {elapsed:.1f} seconds ({elapsed/60:.2f} minutes)")

        # Build one final chronological log after all logging is done
        if args.n_workers and args.n_workers > 1:
            merge_log_files(log_file, worker_log, final_log)
        else:
            merge_log_files(log_file, None, final_log)

        logger.info(f"Final chronological log written to: {final_log}")

        # Remove intermediate logs unless you want to keep them for debugging
        cleanup_logs(output_dir, keep_main=False)

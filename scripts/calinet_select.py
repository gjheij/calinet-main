# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import argparse
import logging
import os

from calinet.logger import init_logging
from calinet.exports.selector import (
    build_participants_registry,
    load_registry,
    make_batch_id,
    save_registry,
    select_subjects,
    write_export_package,
)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Arguments for random subject export.")
    parser.add_argument(
        "--participants-tsv",
        type=str,
        default=None,
        help="Existing participants registry TSV. If omitted, --input-dir will be used to build one.",
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default='Z:\CALINET2\converted',
        help="Root converted directory used to build participants.tsv (defaults to 'Z:\\CALINET2\\converted').",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="Z:\\CALINET2\\derivatives\\exports",
        help="Output directory for batch exports and tracking files.",
    )

    parser.add_argument(
        "--n",
        type=int,
        required=True,
        help="Number of subjects to export.",
    )

    parser.add_argument(
        "--recipient-lab",
        type=str,
        required=True,
        help="Name of the receiving lab used in metadata and batch ID.",
    )

    parser.add_argument(
        "--sampling-method",
        type=str,
        choices=["equal", "proportional"],
        default="equal",
        help="How to balance sampling across sites.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for reproducible selection.",
    )

    parser.add_argument(
        "--include-previously-exported",
        action="store_true",
        help="Allow subjects that were already exported in earlier batches.",
    )

    parser.add_argument(
        "--write-registry-only",
        action="store_true",
        help="Only build/save participants_registry.tsv and exit without creating a batch export.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Enable verbose debug-level logging.\n\n"
            "When set, the logger runs at ``logging.DEBUG`` instead of "
            "``logging.INFO``."
        ),
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, "log.log")

    log_level = logging.DEBUG if args.debug else logging.INFO
    init_logging(level=log_level, logfile=log_file)

    logger = logging.getLogger("calinet.select")
    logger.info(f"Log-file: {log_file}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Sampling method: {args.sampling_method}")
    logger.info(f"Requested N: {args.n}")
    logger.info(f"Recipient lab: {args.recipient_lab}")

    if args.participants_tsv is not None:
        if not os.path.exists(args.participants_tsv):
            raise Exception("participants.tsv does not exist")
        registry_df = load_registry(args.participants_tsv)
        logger.info(f"Loaded existing registry: {args.participants_tsv}")
    elif args.input_dir is not None:
        if not os.path.exists(args.input_dir):
            raise Exception("Input directory does not exist")
        registry_df = build_participants_registry(args.input_dir)
        logger.info(f"Built registry from raw input directory: {args.input_dir}")
    else:
        raise Exception("You must specify either --participants-tsv or --input-dir")

    registry_path = os.path.join(args.output_dir, "participants_registry.tsv")
    save_registry(registry_df, registry_path)
    logger.info(f"Saved registry to: {registry_path}")

    if args.write_registry_only:
        logger.info("--write-registry-only was set; exiting after writing registry")
        raise SystemExit(0)

    batch_id = make_batch_id(args.recipient_lab, args.n, args.seed)
    logger.info(f"Batch ID: {batch_id}")

    selected_df = select_subjects(
        registry_df,
        n=args.n,
        mode=args.sampling_method,
        seed=args.seed,
        exclude_previously_exported=not args.include_previously_exported,
    )
    logger.info(f"Selected {len(selected_df)} subjects")

    summary = write_export_package(
        registry_df,
        selected_df,
        args.output_dir,
        batch_id=batch_id,
        recipient_lab=args.recipient_lab,
        sampling_method=args.sampling_method,
        seed=args.seed,
        save_updated_registry=True,
    )

    logger.info(f"Wrote export batch to: {summary.output_dir}")
    logger.info(f"Export log: {summary.export_log_path}")
    logger.info(f"Updated registry: {summary.updated_registry_path}")

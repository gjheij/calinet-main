# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import logging
import numpy as np
import pandas as pd

from typing import Optional

import calinet.core.io as cio
from calinet.exports.utils import (
    read_table,
    find_physio_dir,
    filter_subjects,
    discover_subjects,
    find_signal_column,
    load_sampling_info,
    find_matching_files,
    is_multisite_export_mode,
    load_subjects_from_export,
    maybe_copy_participant_files,
    create_derivative_dataset_description
)

logger = logging.getLogger(__name__)


AUTONOMATE_DEFAULT_CODE = 0

def save_autonomate_ascii(
        out_path: str,
        eda_signal,
        event_signal,
        fmt: str="%.10g",
    ) -> None:
    """
    Save electrodermal and event signals to a two-column ASCII file.

    This function converts input signals into NumPy arrays, validates that
    they have matching lengths, and writes them as a whitespace-delimited
    text file compatible with Autonomate.

    Parameters
    ----------
    out_path : str
        Path where the ASCII file will be written.
    eda_signal : array-like
        Electrodermal activity (EDA) signal.
    event_signal : array-like
        Event code signal aligned with the EDA signal.
    fmt : str, default="%.10g"
        Format string passed to ``numpy.savetxt``.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the input signals do not have the same number of samples.
    IOError
        If writing the file fails.
    """

    eda_signal = np.asarray(eda_signal, dtype=np.float64).reshape(-1)
    event_signal = np.asarray(event_signal, dtype=np.float64).reshape(-1)

    if eda_signal.shape[0] != event_signal.shape[0]:
        raise ValueError(
            "EDA signal and event signal must have the same number of samples. "
            f"Got {eda_signal.shape[0]} and {event_signal.shape[0]}."
        )

    data = np.column_stack([eda_signal, event_signal])
    np.savetxt(out_path, data, fmt=fmt)


def _convert_bundle_to_ascii(
        subject: str,
        bundle: dict,
        output_dir: str,
        modality: str,
        overwrite: bool=False,
    ) -> str:
    """
    Convert a BIDS physio bundle into an Autonomate-compatible ASCII file.

    This function extracts electrodermal signals and event information from
    BIDS-formatted inputs, generates an event code vector, and writes both
    signals to a two-column ASCII file. A JSON sidecar describing the export
    is also created.

    Parameters
    ----------
    subject : str
        Subject identifier.
    bundle : dict
        Dictionary containing paths to physio TSV/JSON and events TSV files.
    output_dir : str
        Destination directory for output files.
    modality : str
        Signal modality to extract (e.g., "scr").
    overwrite : bool, default=False
        If True, overwrite existing output files.

    Returns
    -------
    str
        Path to the generated ASCII file.

    Notes
    -----
    - Event types are mapped to unique integer codes.
    - Non-event samples are assigned a default code.
    - NaNs in the signal are replaced with zeros.

    Raises
    ------
    ValueError
        If required data columns are missing or malformed.
    IOError
        If file reading or writing fails.
    """

    task = bundle["task"]
    out_name = f"{subject}_task-{task}_recording-{modality}.txt"
    out_path = os.path.join(output_dir, "physio", out_name)
    sidecar_path = os.path.splitext(out_path)[0] + ".json"

    if os.path.exists(out_path) and not overwrite:
        logger.info(f"Autonomate ASCII already exists, skipping: '{out_path}'")
        return out_path

    sampling_freq, units = load_sampling_info(bundle["physio_json"], modality=modality)

    events_df = pd.read_csv(bundle["events_tsv"], sep="\t")
    physio_df = read_table(bundle["physio_tsv"])

    signal_col = find_signal_column(physio_df, modality)
    eda_signal = pd.to_numeric(physio_df[signal_col], errors="coerce").to_numpy()

    if np.isnan(eda_signal).any():
        logger.warning(
            f"Found NaNs in '{signal_col}' for subject '{subject}', task '{task}'. "
            "Replacing NaNs with 0."
        )
        eda_signal = np.nan_to_num(eda_signal, nan=0.0)

    event_signal = np.full(
        shape=eda_signal.shape[0],
        fill_value=AUTONOMATE_DEFAULT_CODE,
        dtype=np.int32,
    )

    event_code_map: dict[str, int] = {}
    observed_event_counts: dict[str, int] = {}

    if {"event_type", "onset"}.issubset(events_df.columns):
        event_types = (
            events_df["event_type"]
            .dropna()
            .astype(str)
            .str.strip()
        )
        unique_event_types = sorted(et for et in event_types.unique() if et)

        next_code = 1
        if AUTONOMATE_DEFAULT_CODE == next_code:
            next_code += 1

        for event_type in unique_event_types:
            event_code_map[event_type] = next_code
            next_code += 1
            if AUTONOMATE_DEFAULT_CODE == next_code:
                next_code += 1

        work_df = events_df.copy()
        work_df["event_type"] = work_df["event_type"].astype(str).str.strip()
        work_df["onset"] = pd.to_numeric(work_df["onset"], errors="coerce")

        for _, row in work_df.iterrows():
            event_type = row["event_type"]
            onset = row["onset"]

            if not event_type or pd.isna(onset):
                continue

            onset_idx = int(round(float(onset) * float(sampling_freq)))
            if 0 <= onset_idx < event_signal.shape[0]:
                code = event_code_map[event_type]
                event_signal[onset_idx] = code
                observed_event_counts[event_type] = observed_event_counts.get(event_type, 0) + 1
    else:
        logger.warning(
            f"Missing required columns in events TSV for subject '{subject}', task '{task}'. "
            "Expected at least 'event_type' and 'onset'. Using only default event code."
        )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_autonomate_ascii(out_path, eda_signal, event_signal)

    sidecar = {
        "Description": "Autonomate-compatible two-column ASCII text file.",
        "Format": {
            "Type": "ASCII",
            "Delimiter": "whitespace",
            "Columns": [
                "electrodermal_data",
                "event_code",
            ],
        },
        "Source": {
            "Subject": subject,
            "Task": task,
            "Modality": modality,
            "EventsTSV": bundle["events_tsv"],
            "PhysioFile": bundle["physio_tsv"],
            "PhysioJSON": bundle["physio_json"],
        },
        "Signal": {
            "ColumnUsed": signal_col,
            "Units": units,
            "SamplingFrequency": sampling_freq,
            "NumberOfSamples": int(eda_signal.shape[0]),
        },
        "Events": {
            "DefaultCode": AUTONOMATE_DEFAULT_CODE,
            "EventTypeToCode": event_code_map,
            "CodingSummary": (
                "Column 2 contains the default code at non-event samples. "
                "Each distinct event_type is assigned a unique onset code."
            ),
            "ObservedOnsetCountsByEventType": observed_event_counts,
            "ObservedUniqueCodes": sorted(int(x) for x in np.unique(event_signal).tolist()),
        },
        "AutonomateNotes": {
            "Requirement": (
                "Autonomate ASCII import expects two columns with equal numbers of "
                "rows: electrodermal data in column 1 and event codes in column 2."
            ),
        },
    }
    cio.save_json(sidecar_path, sidecar)

    logger.info(
        f"Wrote Autonomate ASCII for subject '{subject}', task '{task}' -> '{out_path}'"
    )
    logger.info(f"Event code mapping for '{out_name}': {event_code_map}")
    return out_path


def print_autonomate_summary(
        ascii_file: str,
        bundle: dict
    ) -> None:
    """
    Print a summary of an Autonomate ASCII file and its source events.

    This function loads a generated ASCII file, validates its structure,
    and logs summary statistics including shape, unique event codes, and
    event types present in the source dataset.

    Parameters
    ----------
    ascii_file : str
        Path to the Autonomate ASCII file.
    bundle : dict
        Dictionary containing at least the path to the source events TSV.

    Returns
    -------
    None

    Notes
    -----
    - The ASCII file is expected to contain exactly two columns.
    - Logging is used for output rather than returning values.

    Raises
    ------
    IOError
        If the ASCII file cannot be read.
    """

    data = np.loadtxt(ascii_file)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    n_rows, n_cols = data.shape
    if n_cols != 2:
        logger.warning(
            f"Autonomate file '{ascii_file}' has {n_cols} columns; expected 2."
        )

    events_df = pd.read_csv(bundle["events_tsv"], sep="\t")
    event_names = sorted(events_df["event_type"].dropna().astype(str).unique()) \
        if "event_type" in events_df.columns else []

    logger.info("=" * 80)
    logger.info("Autonomate ASCII summary")
    logger.info("=" * 80)
    logger.info(f"File: {ascii_file}")
    logger.info(f"Rows (samples): {n_rows}")
    logger.info(f"Columns: {n_cols}")
    logger.info(f"Unique event codes in column 2: {sorted(np.unique(data[:, 1]).tolist())}")
    logger.info(f"Event types present in source TSV: {event_names}")
    logger.info("Column 1: electrodermal data")
    logger.info("Column 2: event onset codes")
    logger.info("=" * 80)


def convert_dataset_to_autonomate(
        input_dir: str,
        output_dir: str,
        modality: str="scr",
        task_name: Optional[str]=None,
        overwrite: bool=False,
        include_n: Optional[int]=None,
        subjects_tsv: Optional[str]=None,
    ) -> None:
    """
    Convert a BIDS dataset into Autonomate-compatible ASCII files.

    This function iterates over subjects in a dataset, extracts physio and
    event data, and exports them into two-column ASCII files suitable for
    Autonomate. It supports subject selection, modality filtering, and both
    standard and multi-site dataset layouts.

    Parameters
    ----------
    input_dir : str
        Path to the input BIDS dataset.
    output_dir : str
        Directory where converted files will be written.
    modality : str, default="scr"
        Signal modality to export.
    task_name : str
        Task name to export, such as ``"acquisition"``. If None, all tasks ['acquisition', 'extinction'] will be used.
    overwrite : bool, default=False
        If True, overwrite existing ASCII files.
    include_n : Optional[int], default=None
        Maximum number of subjects to include (ignored if ``subjects_tsv`` is provided).
    subjects_tsv : Optional[str], default=None
        Path to a TSV file specifying selected subjects.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If the input directory does not exist.
    ValueError
        If no subjects are found or no matching data is available.

    Notes
    -----
    - Supports both standard and multi-site export modes.
    - Generates JSON sidecar metadata for each ASCII file.
    - Logs progress, warnings, and summary information.
    - A final summary is printed for the last processed file.
    """

    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)
    modality = modality.lower()

    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory does not exist: '{input_dir}'")

    os.makedirs(output_dir, exist_ok=True)

    logger.info("Converting BIDS physio data to Autonomate ASCII format")
    logger.info(f"Input directory: '{input_dir}'")
    logger.info(f"Output directory: '{output_dir}'")
    logger.info(f"Requested modality: '{modality}'")

    subject_records = discover_subjects(input_dir)
    if not subject_records:
        raise ValueError(f"No subjects found in '{input_dir}'")

    all_subjects = sorted({rec["subject_id"] for rec in subject_records})

    if subjects_tsv is not None:
        selected_subjects = load_subjects_from_export(str(subjects_tsv))
        logger.info(
            f"Loaded {len(selected_subjects)} subjects from export file: '{subjects_tsv}'"
        )
    else:
        selected_subjects = filter_subjects(all_subjects, include_n)

    maybe_copy_participant_files(
        subjects_tsv=subjects_tsv,
        output_dir=output_dir,
    )

    multisite_export_mode = is_multisite_export_mode(input_dir, subjects_tsv)

    last_ascii_file = None
    last_bundle = None

    # ------------------------------------------------------------------
    # MULTI-SITE EXPORT MODE → FLATTENED SUBJECT OUTPUT
    if multisite_export_mode:
        logger.info("Running in multi-site export mode with flattened subject output")

        selected_set = set(selected_subjects)

        selected_records = [
            rec for rec in subject_records
            if rec["subject_id"] in selected_set
        ]

        found_ids = {rec["subject_id"] for rec in selected_records}
        missing_subjects = sorted(selected_set - found_ids)
        if missing_subjects:
            logger.warning(
                f"Subjects requested in export file but not found in dataset: {missing_subjects}"
            )

        for rec in selected_records:
            subject = rec["subject_id"]
            subject_dir = rec["subject_path"]
            dest_subject_dir = os.path.join(output_dir, subject)

            physio_dir = find_physio_dir(subject_dir)
            if physio_dir is None:
                logger.warning(
                    f"No physio directory found for '{subject}' from site "
                    f"'{rec['site']}', skipping"
                )
                continue

            bundles = find_matching_files(
                physio_dir,
                modality,
                task_name=task_name
            )

            if not bundles:
                logger.warning(
                    f"No matching task bundles found for subject '{subject}' "
                    f"and modality '{modality}'"
                )
                continue

            logger.info(
                f"Converting subject '{subject}' from site "
                f"'{rec['site']}' → '{dest_subject_dir}'"
            )

            for bundle in bundles:
                last_ascii_file = _convert_bundle_to_ascii(
                    subject=subject,
                    bundle=bundle,
                    output_dir=dest_subject_dir,
                    modality=modality,
                    overwrite=overwrite,
                )
                last_bundle = bundle

        create_derivative_dataset_description(
            bids_root=output_dir,
            derivative_name="CALINET Autonomate Export",
            source_dataset="CALINET Fear-Conditioning Dataset",
            generated_by_name="CALINET Autonomate Exporter",
            generated_by_description=(
                "Exports electrodermal activity (EDA) recordings from BIDS-formatted "
                "CALINET datasets into two-column ASCII text files compatible with "
                "Autonomate. Column 1 contains the EDA signal, and column 2 contains "
                "event onset codes with unique integer values per event type."
            ),
            extra_fields={
                "ExportFormat": {
                    "Name": "Autonomate ASCII",
                    "FileExtension": ".txt",
                    "Delimiter": "whitespace",
                    "Columns": [
                        "electrodermal_data",
                        "event_code",
                    ],
                },
                "EventCoding": {
                    "DefaultCode": AUTONOMATE_DEFAULT_CODE,
                    "Description": (
                        "Each distinct event_type is mapped to a unique integer code. "
                        "Non-event samples are assigned a default code."
                    ),
                },
                "Notes": (
                    "This derivative dataset is intended for import into Autonomate. "
                    "Each file contains two columns with equal length corresponding "
                    "to the number of recorded samples."
                ),
            },
        )
        
    # ------------------------------------------------------------------
    # NORMAL MODE → SITE-AWARE SUBJECT OUTPUT
    else:
        logger.info("Running in standard mode")

        selected_set = set(selected_subjects)

        selected_records = [
            rec for rec in subject_records
            if rec["subject_id"] in selected_set
        ]

        if not selected_records:
            raise ValueError("No selected subjects found in dataset")

        for rec in selected_records:
            subject = rec["subject_id"]
            subject_dir = rec["subject_path"]

            if rec["site"] is not None:
                dest_subject_dir = os.path.join(
                    output_dir,
                    rec["site"],
                    rec["subject_id"],
                )
            else:
                dest_subject_dir = os.path.join(
                    output_dir,
                    rec["subject_id"],
                )

            physio_dir = find_physio_dir(subject_dir)
            if physio_dir is None:
                logger.warning(f"No physio directory found for '{subject}', skipping")
                continue

            bundles = find_matching_files(
                physio_dir,
                modality,
                task_name=task_name
            )
            
            if not bundles:
                logger.warning(
                    f"No matching task bundles found for subject '{subject}' "
                    f"and modality '{modality}'"
                )
                continue

            logger.info(
                f"Converting subject '{subject}' → '{dest_subject_dir}'"
            )

            for bundle in bundles:
                last_ascii_file = _convert_bundle_to_ascii(
                    subject=subject,
                    bundle=bundle,
                    output_dir=dest_subject_dir,
                    modality=modality,
                    overwrite=overwrite,
                )
                last_bundle = bundle

    # ------------------------------------------------------------------
    # FINAL SUMMARY
    if last_ascii_file is not None and last_bundle is not None:
        print_autonomate_summary(last_ascii_file, last_bundle)
    else:
        logger.warning("No Autonomate files were written.")

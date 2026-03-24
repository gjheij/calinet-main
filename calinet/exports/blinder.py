# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import shutil
import logging
import pandas as pd

from pathlib import Path
from typing import Optional, Iterable

from calinet import utils
import calinet.core.io as cio
from calinet.exports.utils import (
    filter_subjects,
    should_keep_file,
    discover_subjects,
    normalize_modalities,
    is_multisite_export_mode,
    load_subjects_from_export,
    maybe_copy_participant_files
)

logger = logging.getLogger(__name__)


def _process_events_tsv(
        in_path: str,
        out_path: str
    ) -> None:
    """
    Load, sanitize, and overwrite an events TSV file with blinded content.

    This function reads a tab-separated values (TSV) file containing event
    information, normalizes the ``event_type`` column into a reduced and blinded
    representation, removes potentially identifying metadata columns, and
    enforces a fixed event duration. The transformed dataset is then written
    to a new TSV file.

    Specifically, event types starting with "CS" or "US" are mapped to
    canonical labels ("A" and "B", respectively). All other event types are
    preserved until the mapping stage, after which unmapped values become NaN.

    Parameters
    ----------
    in_path : str
        Path to the input TSV file containing event data.
    out_path : str
        Path where the processed TSV file will be written.

    Returns
    -------
    None
        This function operates via side effects only. The processed file is
        written to ``out_path``.

    Notes
    -----
    - The following columns are removed if present:
      ``stimulus_name``, ``task_name``, ``HED``.
    - A fixed duration of 1 second is assigned to all events.
    - If the ``event_type`` column is missing, a warning is logged and the
      mapping step is skipped.

    Raises
    ------
    pandas.errors.ParserError
        If the TSV file cannot be parsed.
    IOError
        If reading or writing the file fails.
    """
    df = pd.read_csv(in_path, sep="\t")

    if "event_type" in df.columns:
        df["event_type"] = df["event_type"].astype(str).apply(
            lambda x: "CS" if x.startswith("CS") else ("US" if x.startswith("US") else x)
        )
    else:
        logger.warning(f"No 'event_type' column in '{in_path}'")

    mapping = {"CS": "A", "US": "B"}
    df["event_type"] = df["event_type"].map(mapping)

    drop_cols = ["stimulus_name", "task_name", "HED"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    df["duration"] = 1
    df.to_csv(out_path, sep="\t", index=False)


def _process_events_json(
        in_path: str,
        out_path: str
    ) -> None:
    """
    Load, sanitize, and overwrite an events JSON sidecar with blinded metadata.

    This function reads a JSON metadata file associated with events data,
    removes potentially identifying or unnecessary fields, and replaces the
    ``event_type`` description with a simplified, blinded mapping. The
    resulting metadata is written to a new JSON file.

    If the input file cannot be parsed, it is copied unchanged to the output
    location and a warning is logged.

    Parameters
    ----------
    in_path : str
        Path to the input JSON file.
    out_path : str
        Path where the processed JSON file will be written.

    Returns
    -------
    None
        This function operates via side effects only.

    Notes
    -----
    - The following keys are removed if present:
      ``HED``, ``stimulus_name``, ``task_name``.
    - The ``event_type`` field is replaced with a standardized Levels mapping:
      ``{"A": "stimulus A", "B": "stimulus B"}``.
    - If a ``duration`` field exists, its description and units are overwritten.

    Raises
    ------
    IOError
        If writing the output file fails.
    """
    try:
        meta = cio.load_json(in_path)
    except Exception as e:
        logger.warning(f"Failed to read JSON '{in_path}': {e}. Copying unchanged.")
        shutil.copy2(in_path, out_path)
        return

    meta.pop("HED", None)
    meta.pop("stimulus_name", None)
    meta.pop("task_name", None)

    meta["event_type"] = {
        "Levels": {
            "A": "stimulus A",
            "B": "stimulus B"
        }
    }

    if "duration" in meta:
        meta["duration"]["Description"] = "Fixed duration (blinded)"
        meta["duration"]["Units"] = "s"

    cio.save_json(out_path, meta)


def _copy_tree(
        source_root: str,
        dest_root: str,
        modalities: Optional[set[str]]=None,
        skip_pheno: bool=False,
        found_modalities: Optional[set[str]]=None,
        task_name: Optional[str]=None,
        skip_blinding: Optional[bool]=False
    ) -> None:
    """
    Recursively copy a dataset tree with optional filtering and transformation.

    This function walks a directory tree and copies files from ``source_root``
    to ``dest_root`` while optionally filtering by modality, skipping phenotype
    directories, and applying transformations to specific file types.

    Parameters
    ----------
    source_root : str
        Root directory of the source dataset.
    dest_root : str
        Destination directory for the copied dataset.
    modalities : Optional[set[str]], default=None
        Set of modality identifiers to include. If None, all files are copied.
    skip_pheno : bool, default=False
        If True, directories named "phenotype" are skipped entirely.
    found_modalities : Optional[set[str]], default=None
        Mutable set used to collect modalities encountered during copying.
    task_name : str
        Task name to export, such as ``"acquisition"``. If None, all tasks ['acquisition', 'extinction'] will be used.
    skip_blinding: Optional[bool], default=False
        Copy selected directories as is, without blinding stimulus events

    Returns
    -------
    None

    Notes
    -----
    - Events TSV and JSON files are processed using specialized blinding
      functions.
    - Other files are copied verbatim using ``shutil.copy2``.
    - Modality detection relies on external utility functions.

    Raises
    ------
    Exception
        Propagates any exception encountered during file processing.
    """
    source_root = os.path.abspath(source_root)
    dest_root = os.path.abspath(dest_root)

    for root, dirs, files in os.walk(source_root):
        rel_dir = os.path.relpath(root, source_root)

        if skip_pheno and "phenotype" in Path(rel_dir).parts:
            dirs[:] = []
            logger.debug(f"Skipping phenotype directory: '{rel_dir}'")
            continue

        out_root = dest_root if rel_dir == "." else os.path.join(dest_root, rel_dir)
        os.makedirs(out_root, exist_ok=True)

        for fname in files:
            if not should_keep_file(fname, modalities):
                logger.debug(f"Skipping file due to modality filter: '{fname}'")
                continue
            
            if task_name is not None:
                if utils._extract_task(fname) != task_name:
                    logger.debug(f"Skipping file due to task filter: '{fname}'")
                    continue

            in_path = os.path.join(root, fname)
            rel_file = os.path.normpath(os.path.join(rel_dir, fname))
            out_path = os.path.join(out_root, fname)

            modality = utils._extract_recording(fname)
            if modality is not None and found_modalities is not None:
                found_modalities.add(modality)

            try:
                if not skip_blinding:
                    if utils._is_events_tsv(fname):
                        logger.info(f"Blinding events TSV: '{rel_file}'")
                        _process_events_tsv(in_path, out_path)
                    elif utils._is_events_json(fname):
                        logger.info(f"Blinding events JSON: '{rel_file}'")
                        _process_events_json(in_path, out_path)
                    else:
                        shutil.copy2(in_path, out_path)
                else:
                    logger.debug(f"Skipping blinding, copying '{rel_file}' as is")
                    shutil.copy2(in_path, out_path)
            except Exception as e:
                logger.error(f"Failed processing '{in_path}': {e}")
                raise


def blind_dataset(
        input_dir: str,
        output_dir: str,
        include_n: Optional[int]=None,
        modalities: Optional[Iterable[str]]=None,
        task_name: Optional[str]=None,
        subjects_tsv: Optional[str]=None,
        skip_blinding: Optional[bool]=False
    ) -> None:
    """
    Create a blinded copy of a dataset with optional subject and modality filtering.

    This function processes a dataset directory, selects a subset of subjects,
    optionally filters by modality, and writes a sanitized (blinded) version
    of the dataset to a new location. Event files are modified to remove
    identifying information, while other files are copied.

    Parameters
    ----------
    input_dir : str
        Path to the input dataset directory.
    output_dir : str
        Path where the blinded dataset will be written.
    include_n : Optional[int], default=None
        Maximum number of subjects to include (ignored if ``subjects_tsv`` is provided).
    modalities : Optional[Iterable[str]], default=None
        Modalities to include (e.g., "bold", "eeg"). If None, all modalities are included.
    task_name : str
        Task name to export, such as ``"acquisition"``. If None, all tasks ['acquisition', 'extinction'] will be used.
    subjects_tsv : Optional[str], default=None
        Path to a TSV file specifying selected subjects.
    skip_blinding: Optional[bool], default=False
        Copy selected directories as is, without blinding stimulus events

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If the input directory does not exist.
    ValueError
        If no subjects are found or selection yields no matches.

    Notes
    -----
    - Supports both flat and multi-site dataset layouts.
    - In multi-site export mode, subject directories are flattened.
    - Event files are processed to remove sensitive metadata.
    - Logging provides detailed progress and warnings.

    Side Effects
    ------------
    - Creates directories and writes files to ``output_dir``.
    - Logs informational and warning messages.
    """
    
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)

    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory does not exist: '{input_dir}'")

    os.makedirs(output_dir, exist_ok=True)

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

    selected_modalities = normalize_modalities(modalities)
    found_modalities: set[str] = set()

    maybe_copy_participant_files(
        subjects_tsv=subjects_tsv,
        output_dir=output_dir,
    )

    multisite_export_mode = is_multisite_export_mode(input_dir, subjects_tsv)

    # ------------------------------------------------------------------
    # MULTI-SITE EXPORT MODE → FLATTENED SUBJECT COPY
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
            dest_subject_dir = os.path.join(output_dir, rec["subject_id"])

            logger.info(
                f"Copying subject '{rec['subject_id']}' from site "
                f"'{rec['site']}' → '{dest_subject_dir}'"
            )

            _copy_tree(
                source_root=rec["subject_path"],
                dest_root=dest_subject_dir,
                modalities=selected_modalities,
                skip_pheno=True,
                found_modalities=found_modalities,
                task_name=task_name,
                skip_blinding=skip_blinding
            )

    # ------------------------------------------------------------------
    # NORMAL MODE → SUBJECT-LEVEL COPY (NO FULL TREE WALK)
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
            # preserve relative structure (site-aware if present)
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

            logger.info(
                f"Copying subject '{rec['subject_id']}' → '{dest_subject_dir}'"
            )

            _copy_tree(
                source_root=rec["subject_path"],
                dest_root=dest_subject_dir,
                modalities=selected_modalities,
                skip_pheno=False,
                found_modalities=found_modalities,
                task_name=task_name
            )

    # ------------------------------------------------------------------
    # FINAL LOGGING
    if found_modalities:
        logger.info(f"Modalities found in dataset: {sorted(found_modalities)}")

    if selected_modalities is not None:
        missing_modalities = sorted(selected_modalities - found_modalities)
        if missing_modalities:
            logger.warning(
                f"Requested modalities not found in dataset: {missing_modalities}"
            )

    logger.info(f"Blinded dataset written to '{output_dir}'")

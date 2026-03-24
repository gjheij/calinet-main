# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import glob
import logging
import shutil
import pandas as pd
from pathlib import Path

from calinet import utils
from calinet.exports.utils import (
    discover_subjects,
    should_keep_file,
    normalize_modalities,
)

from typing import Union, Optional, Iterable

import logging
logger = logging.getLogger(__name__)


def write_subject_phenotype(
        phenotype_dir: Union[str, Path],
        subject_name: str,
        new_subject_dir: Union[str, Path]
    ) -> None:
    """
    Write subject-specific phenotype files into a target dataset directory.

    Parameters
    ----------
    phenotype_dir : str or pathlib.Path
        Directory containing aggregated phenotype ``.tsv`` files and matching
        ``.json`` sidecars.
    subject_name : str
        Participant identifier used to filter rows, such as ``"sub-001"``.
    new_subject_dir : str or pathlib.Path
        Target directory where subject-specific phenotype files are written.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        Raised if a phenotype ``.json`` sidecar corresponding to a selected
        ``.tsv`` file does not exist.

    Notes
    -----
    This function searches ``phenotype_dir`` for all ``.tsv`` files, filters
    each table to rows where ``"participant_id"`` equals ``subject_name``, and
    writes the filtered result into ``<new_subject_dir>/phenotype``.

    For each written ``.tsv`` file, the matching ``.json`` sidecar is copied to
    the same destination with the same basename.

    If no phenotype files are found, the function logs an informational message
    and returns.

    If a phenotype file does not contain ``subject_name``, the function logs an
    informational message and returns immediately.

    This function performs filesystem reads, directory creation, file writes,
    file copies, and logging.
    """
    
    pattern = os.path.join(phenotype_dir, "*.tsv")
    if os.path.exists(phenotype_dir):
        tsv_files = glob.glob(pattern)

        if len(tsv_files)==0:
            logger.info(f"No phenotype data in '{phenotype_dir}'")
            return

        logger.info("Dealing with phenotype data")
        for tsv in tsv_files:
            df = pd.read_csv(
                tsv,
                delimiter="\t",
                na_values="n/a"
            )

            df_sub = df.loc[df["participant_id"] == subject_name].copy()
            
            if df_sub.shape==0:
                logger.info(f"'{subject_name}' not in '{tsv}'")
                return
            
            new_pheno = os.path.join(new_subject_dir, "phenotype")
            if not os.path.exists(new_pheno):
                os.makedirs(new_pheno, exist_ok=True)
            
            # write new tsv
            new_tsv = os.path.join(new_pheno, os.path.basename(tsv))
            _ = utils.common_write_tsv(
                df_sub,
                id_key=os.path.basename(tsv).replace(".tsv", ""),
                language="",
                phenotype_dir=new_pheno
            )
            
            # copy json
            src_json = tsv.replace(".tsv", ".json")
            trg_json = new_tsv.replace(".tsv", ".json")

            assert os.path.exists(src_json), FileNotFoundError(f"Could not find sidecar file '{src_json}'")
            shutil.copy(src_json, trg_json)

        logger.info("Done")


def write_dataset_info(
        input_dir: Union[str, Path],
        subject_name: str,
        new_proj_base: Union[str, Path]
    ) -> None:
    """
    Write common dataset-level files into a subject-specific export directory.

    Parameters
    ----------
    input_dir : str or pathlib.Path
        Source dataset directory containing common dataset files such as
        ``"participants.tsv"`` and ``"dataset_description.json"``.
    subject_name : str
        Participant identifier used to filter ``"participants.tsv"`` to a
        single-row subject-specific table.
    new_proj_base : str or pathlib.Path
        Target directory where copied dataset files are written.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        Raised if one of the required common dataset files does not exist in
        ``input_dir``.

    Notes
    -----
    This function expects the following files to exist in ``input_dir``:

    - ``"README"``
    - ``"participants.tsv"``
    - ``"participants.json"``
    - ``"dataset_description.json"``

    All listed files are copied into ``new_proj_base``. For
    ``"participants.tsv"``, the copied file is overwritten with a filtered table
    containing only rows where ``"participant_id"`` equals ``subject_name``.

    This function creates ``new_proj_base`` if it does not already exist.

    This function performs filesystem checks, directory creation, file copies,
    tabular file reads and writes, and logging.
    """
    
    # locate common files
    common_names = [
        "README",
        "participants.tsv",
        "participants.json",
        "dataset_description.json",
    ]

    if not os.path.exists(new_proj_base):
        os.makedirs(new_proj_base, exist_ok=True)

    logger.info(f"Dealing with common dataset files")
    for i in common_names:
        src_file = os.path.join(input_dir, i)
        assert os.path.exists(src_file), FileNotFoundError(f"Could not find '{src_file}' file..")

        # copy file
        trg_file = os.path.join(new_proj_base, i)
        shutil.copy(src_file, trg_file)

        # extract subject ID from participants.tsv
        if src_file.endswith(".tsv"):
            df = pd.read_csv(src_file, delimiter="\t", na_values="n/a")
            df_sub = df.loc[df["participant_id"] == subject_name].copy()
            
            _ = utils.common_write_tsv(
                df_sub,
                id_key=os.path.basename(src_file).replace(".tsv", ""),
                language="",
                phenotype_dir=new_proj_base
            )

        logger.debug(f" Wrote '{trg_file}'")

    logger.info("Done")


def separate_and_zip(
        input_dir: str,
        output_dir: str,
        overwrite: bool=False,
        include_n: int=None,
        modalities: Optional[Iterable[str]]=None
    ) -> None:
    """
    Create per-subject export directories, package them as ZIP archives, and
    remove temporary export folders.

    Parameters
    ----------
    input_dir : str
        Source converted dataset directory containing subject folders, common
        dataset files, and an optional ``"phenotype"`` directory.
    output_dir : str
        Destination directory where subject-specific ZIP archives are created.
    overwrite: bool
        Overwrite existing zip-files (default = False)
    include_n : int
        Include only the first N discovered subjects. This is mainly to test if
        the zipper is working as intended.
    modalities : Optional[Iterable[str]]
        Modalities to include in the packaged subject exports. If None, all
        files are included. Files without an identifiable modality are always
        retained.

    Returns
    -------
    None

    Notes
    -----
    This function uses ``discover_subjects`` to support both flat datasets
    (``input_dir/sub-*``) and multi-site datasets (``input_dir/<site>/sub-*``).
    Similar to the blinder export logic, modality filtering is applied at file
    level so only requested recording types are included while dataset-level
    files are preserved.

    For each selected subject, the function:

    1. copies the full subject directory into a temporary export directory
    2. writes dataset-level files from the corresponding dataset root into that
       export directory using ``write_dataset_info``
    3. writes subject-specific phenotype files using
       ``write_subject_phenotype`` when a phenotype directory exists
    4. creates a ZIP archive from the temporary export directory
    5. removes the temporary export directory after successful archiving

    ZIP archives are named from the subject identifier with ``"sub"`` replaced
    by ``"ds"``, for example ``"sub-001"`` -> ``"ds-001.zip"``.

    If a ZIP archive already exists for a subject, that subject is skipped
    unless ``overwrite`` is True.
    """

    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    subject_records = discover_subjects(input_dir)
    selected_modalities = normalize_modalities(modalities)
    found_modalities: set[str] = set()

    if not subject_records:
        logger.warning(f"No 'sub-' folders found in '{input_dir}'")
        return

    if include_n is not None:
        logger.info(f"Including first N={include_n} subjects from '{input_dir}'")
        subject_records = subject_records[:include_n]

    logger.info(f"Found {len(subject_records)} subjects in '{input_dir}'")

    for rec in subject_records:
        subject = rec["subject_id"]
        subj_dir = rec["subject_path"]
        dataset_root = rec["dataset_root"]
        site = rec.get("site")

        zip_stem = subject.replace('sub', 'ds', 1)
        zip_parent = Path(output_dir)
        if site is not None:
            zip_parent.mkdir(parents=True, exist_ok=True)

        new_base = str(zip_parent / zip_stem)
        new_subj = os.path.join(new_base, subject)
        zip_path = zip_parent / f"{zip_stem}.zip"

        if zip_path.is_file() and not overwrite:
            logger.info(f"Zip for '{subject}' exists. Skipping..")
            continue

        if zip_path.is_file() and overwrite:
            logger.info(f"Zip for '{subject}' exists. Overwriting..")
            zip_path.unlink()

        if os.path.isdir(new_base):
            logger.info(f"Removing existing temp dir '{new_base}'")
            shutil.rmtree(new_base)

        logger.info(f"Copying original folder {subj_dir} to {new_subj}")
        shutil.copytree(
            subj_dir,
            new_subj,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(),
        )

        if selected_modalities is not None:
            for root, _, files in os.walk(new_subj):
                for fname in files:
                    src = os.path.join(root, fname)
                    rel_file = os.path.relpath(src, new_subj)
                    if not should_keep_file(fname, selected_modalities):
                        logger.debug(f"Removing file due to modality filter: '{rel_file}'")
                        os.remove(src)
                        continue

                    modality = utils._extract_recording(fname)
                    if modality is not None:
                        found_modalities.add(modality)
        else:
            for root, _, files in os.walk(new_subj):
                for fname in files:
                    modality = utils._extract_recording(fname)
                    if modality is not None:
                        found_modalities.add(modality)

        write_dataset_info(
            dataset_root,
            subject,
            new_base
        )

        phenotype_dir = os.path.join(dataset_root, "phenotype")
        write_subject_phenotype(
            phenotype_dir,
            subject,
            new_base
        )

        shutil.make_archive(
            new_base,
            'zip',
            root_dir=new_base
        )

        logger.info(f"Created zip for subject '{subject}': {new_base}")
        logger.info(f"Removing '{subject}': {new_base}")
        shutil.rmtree(new_base)
        logger.info("Done")

    if found_modalities:
        logger.info(f"Modalities found in dataset: {sorted(found_modalities)}")

    if selected_modalities is not None:
        missing_modalities = sorted(selected_modalities - found_modalities)
        if missing_modalities:
            logger.warning(
                f"Requested modalities not found in dataset: {missing_modalities}"
            )

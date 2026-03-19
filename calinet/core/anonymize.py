# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import random
import shutil
import pandas as pd

from calinet.core.io import save_json
from calinet.core.pheno import common_write_tsv
from calinet.config import available_labs

from typing import Dict, Tuple, List, Any

import logging
logger = logging.getLogger(__name__)


def update_subject_ids(
        file: str,
        mapper: Dict[str, str]
    ) -> pd.DataFrame:
    """
    Update participant identifiers in a phenotype TSV file.

    This function reads a tab-delimited phenotype file, filters rows to
    retain only those participants present in a provided mapping, and
    replaces their ``"participant_id"`` values with standardized identifiers.
    The updated DataFrame is then sorted and written back using a common TSV
    writing utility.

    Parameters
    ----------
    file : str
        Path to the input TSV file containing a ``"participant_id"`` column.
    mapper : dict of str to str
        Dictionary mapping original participant identifiers to new
        standardized identifiers.

    Returns
    -------
    df_updated : pandas.DataFrame
        Updated DataFrame with filtered and remapped ``"participant_id"``
        values, written to disk via ``common_write_tsv``.

    Raises
    ------
    KeyError
        Raised if the ``"participant_id"`` column is missing in the input file.
    FileNotFoundError
        Raised if the input file does not exist.
    OSError
        Raised if the file cannot be read or written.

    Notes
    -----
    Rows with ``"participant_id"`` values not present in ``mapper`` are
    discarded.

    Missing values encoded as ``"n/a"`` are preserved during file reading.

    Output writing is delegated to ``common_write_tsv``, which determines
    the final file location and formatting.
    """

    # read while preserving n/a
    df = pd.read_csv(
        file,
        delimiter="\t",
        na_values=["n/a"]
    )

    # keep only rows whose participant_id exists in mapper
    df_updated = df[df["participant_id"].isin(mapper)].copy()

    # replace old IDs with new ones
    df_updated["participant_id"] = df_updated["participant_id"].map(mapper)

    # sort
    df_updated = df_updated.sort_values("participant_id").reset_index(drop=True)

    # use common_write_tsv for phenotype data
    return common_write_tsv(
        df_updated,
        id_key=os.path.basename(file).replace(".tsv", ""),
        language="",
        phenotype_dir=os.path.dirname(file)
    )


def change_sub_ids_in_pheno(
        conv_dir: str,
        anon_dict: Dict[str, str]
    ) -> Dict[str, pd.DataFrame]:
    """
    Update participant identifiers across phenotype TSV files.

    This function traverses the ``phenotype`` directory within a converted
    dataset, applies a participant ID mapping to all ``.tsv`` files, and
    rewrites them using a standardized output routine. Each processed file
    is updated in place and its resulting DataFrame is collected.

    Parameters
    ----------
    conv_dir : str
        Path to the converted dataset directory containing a ``phenotype``
        subdirectory.
    anon_dict : dict of str to str
        Dictionary mapping original participant identifiers to anonymized or
        standardized identifiers.

    Returns
    -------
    updated_files : dict of str to pandas.DataFrame
        Dictionary mapping each processed filename to its updated DataFrame.

    Raises
    ------
    FileNotFoundError
        Raised if the ``phenotype`` directory does not exist.
    KeyError
        Raised if required columns (e.g. ``"participant_id"``) are missing in
        any TSV file.
    OSError
        Raised if files cannot be read or written.

    Notes
    -----
    All ``.tsv`` files found recursively under ``phenotype`` are processed.

    File updates are delegated to ``update_subject_ids``, which handles
    filtering, remapping, sorting, and writing.

    The returned dictionary uses filenames (not full paths) as keys.
    """
    
    ddict = {}
    pheno_dir = os.path.join(conv_dir, "phenotype")
    logger.info(f"Updating subject IDs in {pheno_dir}")
    for root, _, files in os.walk(pheno_dir):
        for file in files:
            if file.endswith(".tsv"):
                file_path = os.path.join(root, file)
                ddict[os.path.basename(file)] = update_subject_ids(file_path, anon_dict)

    return ddict


def change_sub_ids_in_participants_tsv(
        conv_dir: str,
        anon_dict: Dict[str, str]
    ) -> pd.DataFrame:
    """
    Update participant identifiers in the ``participants.tsv`` file.

    This function applies a participant ID mapping to the
    ``participants.tsv`` file located in the converted dataset directory.
    The file is read, filtered, updated, and rewritten using a common TSV
    writing utility.

    Parameters
    ----------
    conv_dir : str
        Path to the converted dataset directory containing
        ``participants.tsv``.
    anon_dict : dict of str to str
        Dictionary mapping original participant identifiers to anonymized or
        standardized identifiers.

    Returns
    -------
    participants_df : pandas.DataFrame
        Updated DataFrame with remapped ``"participant_id"`` values written
        to disk.

    Raises
    ------
    FileNotFoundError
        Raised if ``participants.tsv`` does not exist in ``conv_dir``.
    KeyError
        Raised if the ``"participant_id"`` column is missing in the file.
    OSError
        Raised if the file cannot be read or written.

    Notes
    -----
    This function delegates processing to ``update_subject_ids``, which
    performs filtering, remapping, sorting, and writing.

    Only participants present in ``anon_dict`` are retained in the output.
    """

    # Make changes in participants.tsv
    participants_tsv_path = os.path.join(
        conv_dir,
        "participants.tsv"
    )

    return update_subject_ids(
        participants_tsv_path,
        anon_dict
    )


def anonymize_converted_data(
        conv_dir: str,
        lab_name: str
    ) -> Tuple[str, Dict[str, pd.DataFrame]]:
    """
    Anonymize participant identifiers in a converted dataset.

    This function generates a deterministic anonymization mapping for all
    participant directories in a converted dataset, applies the mapping to
    phenotype files and ``participants.tsv``, and renames subject folders
    and contained files accordingly.

    Parameters
    ----------
    conv_dir : str
        Path to the converted dataset directory containing subject folders
        (e.g. ``"sub-XXX"``) and associated phenotype data.
    lab_name : str
        Name of the lab/site used to generate a deterministic anonymization
        mapping.

    Returns
    -------
    mapper_file : str
        Path to the saved JSON file containing the anonymization mapping.
    updated_data : dict of str to pandas.DataFrame
        Dictionary containing updated DataFrames for all processed phenotype
        files, including ``participants.tsv``.

    Raises
    ------
    FileNotFoundError
        Raised if ``conv_dir`` does not exist or required files are missing.
    OSError
        Raised if files or directories cannot be moved, read, or written.
    KeyError
        Raised if expected participant identifiers are missing in mapping or
        data files.

    Notes
    -----
    Participant identifiers are anonymized using a deterministic mapping
    generated by ``get_anonymization_map``.

    All subject directories matching the pattern ``"sub-*"`` are renamed,
    and all filenames containing the original subject identifier are
    updated accordingly.

    Phenotype files are updated via ``change_sub_ids_in_pheno``, and the
    ``participants.tsv`` file is processed separately.

    The anonymization mapping is saved as ``mapper.json`` in ``conv_dir``.
    """

    # Get subject list
    sub_names = [
        name
        for name in os.listdir(conv_dir)
        if name.startswith('sub-') and os.path.isdir(os.path.join(conv_dir, name))
    ]

    # get mapper -> seed is fixed, so deterministic
    anon_dict = get_anonymization_map(lab_name, sub_names)

    # save mapper file
    mapper_file = os.path.join(conv_dir, "mapper.json")
    save_json(mapper_file, anon_dict)

    # change all files in phenotype/ -> returns dict
    ddict = change_sub_ids_in_pheno(conv_dir, anon_dict)

    # rename folders/files
    for sub_name in sub_names:
        new_sub_name = anon_dict.get(sub_name)
        sub_dir = os.path.join(conv_dir, sub_name)
        new_sub_dir_name = os.path.join(conv_dir, new_sub_name)

        if os.path.exists(new_sub_dir_name):
            shutil.rmtree(new_sub_dir_name)
        shutil.move(sub_dir, new_sub_dir_name)

        sub_dir = new_sub_dir_name
        for root, _, files in os.walk(sub_dir):
            for name in files:
                new_name = name.replace(sub_name, new_sub_name)
                new_path = os.path.join(root, new_name)
                shutil.move(os.path.join(root, name), new_path)

    # change IDs in participants.tsv
    part_df = change_sub_ids_in_participants_tsv(
        conv_dir,
        anon_dict
    )

    ddict["participants.tsv"] = part_df
    return mapper_file, ddict


def get_lab_seed(
        lab_name: str
    ) -> int:
    """
    Generate a deterministic seed value from a lab name.

    This function derives an integer seed from the provided ``lab_name``
    using a string-based hashing utility. The resulting seed can be used
    for reproducible operations such as anonymization or randomization.

    Parameters
    ----------
    lab_name : str
        Name of the lab/site used to generate the seed.

    Returns
    -------
    lab_seed : int
        Deterministic integer seed derived from ``lab_name``.

    Raises
    ------
    TypeError
        Raised if ``lab_name`` is not a string.
    ValueError
        Raised if ``lab_name`` is empty.

    Notes
    -----
    The seed is generated via ``get_seed_from_string``, ensuring that the
    same lab name always produces the same seed.

    Informational messages are logged before and after seed generation.
    """
    
    logger.info(f"Lab name: {lab_name}")
    lab_seed = get_seed_from_string(lab_name)
    logger.info(f"Lab seed: {lab_seed}")
    return lab_seed


def get_seed_from_string(
        s: str
    ) -> int:
    """
    Generate a deterministic integer seed from a string.

    This function computes a simple hash-like seed by summing the Unicode
    code points of all characters in the input string. The resulting value
    can be used for reproducible operations such as shuffling or
    anonymization.

    Parameters
    ----------
    s : str
        Input string used to derive the seed.

    Returns
    -------
    seed : int
        Deterministic integer seed derived from the input string.

    Raises
    ------
    TypeError
        Raised if ``s`` is not a string.

    Notes
    -----
    This method provides a simple and deterministic mapping from strings
    to integers, but it is not suitable for cryptographic purposes.
    """
    seed = sum(ord(char) for char in s)
    return seed


def shuffle_list_with_seed(
        input_list: List[Any],
        seed: int
    ) -> List[Any]:
    """
    Shuffle a list deterministically using a fixed seed.

    This function shuffles the input list in place using Python's random
    module with a fixed seed, ensuring reproducible ordering.

    Parameters
    ----------
    input_list : list of Any
        List of elements to shuffle.
    seed : int
        Seed value used to initialize the random number generator.

    Returns
    -------
    shuffled_list : list of Any
        The shuffled list (same object as input, modified in place).

    Raises
    ------
    TypeError
        Raised if ``input_list`` is not a list or ``seed`` is not an integer.

    Notes
    -----
    The input list is modified in place. If the original ordering must be
    preserved, pass a copy of the list.
    """

    random.seed(seed)
    random.shuffle(input_list)
    return input_list


def anonymize_subject_id_map(
        lab_seed: int,
        subject_id_map: Dict[str, str]
    ) -> Dict[str, str]:
    """
    Anonymize subject identifiers using a deterministic shuffle.

    This function reassigns subject identifiers by shuffling the keys of
    the input mapping using a fixed seed, and mapping them to the original
    values in a new randomized order.

    Parameters
    ----------
    lab_seed : int
        Seed used to deterministically shuffle subject identifiers.
    subject_id_map : dict of str to str
        Mapping from original subject identifiers to target identifiers.

    Returns
    -------
    shuffled_subject_map : dict of str to str
        New mapping with original keys assigned to shuffled target values.

    Raises
    ------
    TypeError
        Raised if inputs are not of the expected types.

    Notes
    -----
    The anonymization preserves all values but permutes their assignment
    across keys using a deterministic shuffle.
    """

    original_subject_ids = list(subject_id_map.keys())
    shuffled_subject_ids = shuffle_list_with_seed(original_subject_ids.copy(), lab_seed)

    shuffled_subject_map = {}
    for i in range(len(original_subject_ids)):
        shuffled_subject_map[original_subject_ids[i]] = subject_id_map[
            shuffled_subject_ids[i]
        ]

    return shuffled_subject_map


def get_subject_id_map(
        lab_data: List[str],
        lab_name: str
    ) -> Dict[str, str]:
    """
    Generate a mapping from original subject identifiers to standardized IDs.

    This function assigns new participant identifiers in a standardized
    format based on the lab name and index position of each subject.

    Parameters
    ----------
    lab_data : list of str
        List of original subject identifiers.
    lab_name : str
        Lab-specific identifier used to construct new subject IDs.

    Returns
    -------
    subject_id_map : dict of str to str
        Mapping from original subject identifiers to standardized IDs.

    Raises
    ------
    TypeError
        Raised if inputs are not of the expected types.

    Notes
    -----
    New identifiers follow the format ``"sub-Calinet{lab_name}{XX}"``,
    where ``XX`` is a zero-padded index starting from 1.
    """

    subject_id_map = {}
    for i in range(len(lab_data)):
        subject_id_map[lab_data[i]] = f"sub-Calinet{lab_name}{i+1:02d}"
    return subject_id_map


def get_anonymization_map(
        lab_name: str,
        lab_data: List[str]
    ) -> Dict[str, str]:
    """
    Generate a deterministic anonymization mapping for subject identifiers.

    This function creates a standardized subject ID mapping and then
    applies a deterministic shuffle based on a lab-specific seed to produce
    anonymized identifiers.

    Parameters
    ----------
    lab_name : str
        Name of the lab/site used to derive metadata and seed.
    lab_data : list of str
        List of original subject identifiers.

    Returns
    -------
    anon_mapping_dict : dict of str to str
        Mapping from original subject identifiers to anonymized identifiers.

    Raises
    ------
    KeyError
        Raised if ``lab_name`` is not found in ``available_labs``.
    TypeError
        Raised if inputs are not of the expected types.

    Notes
    -----
    The anonymization process consists of:
    - extracting a normalized lab meta name
    - generating a deterministic seed from the lab name
    - creating a sequential subject ID mapping
    - shuffling the mapping deterministically
    """
    
    lab_meta = available_labs.get(lab_name).get("MetaName").replace(" ", "")
    lab_seed = get_lab_seed(lab_meta)

    subject_id_map = get_subject_id_map(lab_data, lab_meta)
    anon_mapping_dict = anonymize_subject_id_map(lab_seed, subject_id_map)
    return anon_mapping_dict

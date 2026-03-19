# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import pandas as pd
from calinet.imports import biopac
from calinet.config import available_labs
from calinet.core.pheno import pad_missing_columns

from calinet.utils import (
    rename_col,
    common_write_tsv,
    convert_questionnaire_columns_to_int,
)

from typing import Any, Dict, Optional, Tuple

import logging
logger = logging.getLogger(__name__)

# lab-specific pheno info
lab_name = __name__.split(".")[-1]
lab_pheno = available_labs.get(lab_name).get("Phenotype")
language = lab_pheno.get("Language")


def read_raw_physio_file(
        raw_physio_acq: str
    ) -> Tuple[pd.DataFrame, float, Any]:
    """
    Read a Bielefeld raw physiology ``.acq`` file.

    This implementation is equivalent to the Bonn pipeline in that it reads
    a single ``.acq`` file with ``biopac.read_acq_file`` and returns the
    extracted signal table, sampling rate, and selected channel metadata.

    Parameters
    ----------
    raw_physio_acq : str
        Path to the Bielefeld ``.acq`` file to read.

    Returns
    -------
    physio_df : pandas.DataFrame
        Physiology data extracted from the input ``.acq`` file.
    sr : float
        Sampling frequency in Hz for the recorded physiology data.
    chan_info : Any
        Selected channel metadata returned by ``biopac.read_acq_file``.

    Raises
    ------
    Exception
        Raised if the input ``.acq`` file cannot be read.

    Notes
    -----
    This function does not write files.

    Channel selection is resolved from the Bielefeld lab configuration via
    ``available_labs.get(lab_name).get("ChannelRegex")``.
    """

    # Extract physiological data
    logger.info(f"Reading .acq file: {raw_physio_acq}")

    # read acqknowledge file
    lab_name = __name__.split(".")[-1]

    try:
        res = biopac.read_acq_file(
            raw_physio_acq,
            channels=available_labs.get(lab_name).get("ChannelRegex")
        )
        logger.info("Loading acq-file successfull")
    except Exception as e:
        raise Exception(f"Error while reading '{raw_physio_acq}': {e}") from e
    
    # extract dataframe and sampling rate
    physio_df = res.df
    sr = res.sampling_rate_hz
    chan_info = res.selected_channel_info

    return (physio_df, sr, chan_info)


def find_physio_acq_file(
        subject_raw_data_path: str,
        subject_name: str
    ) -> str:
    """
    Locate the Bielefeld raw physiology ``.acq`` file for one participant.

    Unlike the Bonn implementation, which expects the physiology file to be
    located directly under ``subject_raw_data_path`` and named with the
    ``"CALINET_Template{subject_id}.acq"`` pattern, the Austin reference
    example searches a sibling ``"physio"`` directory and derives a
    three-digit file name from the participant label.

    Parameters
    ----------
    subject_raw_data_path : str
        Path to the participant-specific raw data directory expected to
        contain the Bielefeld ``.acq`` file.
    subject_name : str
        Participant label used to derive the physiology file name. The value
        is split on ``"-"`` and the second token is inserted into the
        ``"CALINET_Template{subject_id}.acq"`` naming pattern.

    Returns
    -------
    physio_path : str
        Full path to the matching Bielefeld ``.acq`` file.

    Raises
    ------
    Exception
        Raised if ``subject_raw_data_path`` does not exist as a directory.
    FileNotFoundError
        Raised if the expected ``.acq`` file is not present in
        ``subject_raw_data_path``.

    Notes
    -----
    This function does not read or write files.

    File discovery is Bielefeld-specific because it uses a fixed template
    file name rather than the Bonn subject-level naming convention.
    """

    # go up one level from the subject folder to get the raw_data_dir
    raw_data_root = os.path.dirname(subject_raw_data_path)
    physio_dir = os.path.join(raw_data_root, "physio")

    if not os.path.isdir(physio_dir):
        raise FileNotFoundError(f"No physio folder at {physio_dir!r}")
    else:
        logger.info(f"Physio-folder: {physio_dir}")

    # build expected filename, e.g. "501" → "501.acq"
    subject_id = int(subject_name[-3:])
    expected_fname = f"{subject_id:03d}.acq"

    physio_path = os.path.join(physio_dir, expected_fname)
    if os.path.isfile(physio_path):
        return physio_path
    else:
        raise FileNotFoundError(f".acq file {expected_fname} not found in {physio_dir!r}")


def find_questionnaire_file(
        raw_data_dir: str
    ) -> Optional[str]:
    """
    Find the Bielefeld questionnaire source file under the raw data tree.

    Unlike the Bonn implementation, which searches for a questionnaire file
    by a generic pattern within a directory tree, the Bielefeld
    implementation walks ``raw_data_dir`` and returns the first file whose
    name contains the fixed study-specific token
    ``"data_test"``.

    Parameters
    ----------
    raw_data_dir : str
        Root directory to search recursively for the Bielefeld questionnaire
        source file.

    Returns
    -------
    questionnaire_file : str or None
        Full path to the first matching questionnaire file, or ``None`` if
        no matching file is found.

    Notes
    -----
    This function does not read the questionnaire contents and does not
    write files.

    Matching is based on a Bielefeld-specific file name substring rather
    than on questionnaire content or extension alone.
    """
    
    base_dir = os.path.join(raw_data_dir, "questionnaires")
    for root, _, files in os.walk(base_dir):
        for fn in files:
            low = fn.lower()
            if "data_test" in low and low.endswith((".xlsx", ".xls")):
                return os.path.join(root, fn)


def add_block(
        rename_cols: Dict[str, str],
        bf_prefix: str,
        name: str
    ) -> None:
    """
    Add Bielefeld questionnaire item renaming rules for one questionnaire block.

    This helper is Bielefeld-specific. Unlike the Bonn implementation, which
    primarily renames questionnaire columns during aggregation based on
    REDCap-style prefixes, Bielefeld standardizes source column names
    earlier by constructing mappings such as ``"bf11_01"`` to
    ``"gad7_1"``.

    Parameters
    ----------
    rename_cols : dict of str to str
        Mutable mapping of source column names to standardized output column
        names. This dictionary is modified in place.
    bf_prefix : str
        Bielefeld source prefix for the questionnaire block, such as
        ``"bf11"``.
    name : str
        Questionnaire base name used to derive the standardized item prefix,
        such as ``"gad"`` or ``"phq"``.

    Returns
    -------
    None

    Notes
    -----
    This function has the side effect of mutating ``rename_cols`` in place.

    The number of items added is taken from the Bielefeld phenotype
    configuration stored in ``lab_pheno``.
    """

    n = lab_pheno.get(name)
    std_name = f"{name}{n}"
    for i in range(1, n + 1):
        rename_cols[f"{bf_prefix}_{i:02d}"] = f"{std_name}_{i}"


def parse_questionnaire_file(
        questionnaire_file: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse the Bielefeld questionnaire file and standardize participant fields.

    Unlike the Bonn implementation, which reads a comma-separated REDCap
    export and applies lab-specific renaming for Bonn column names,
    Bielefeld reads an Excel questionnaire file, drops the first row,
    normalizes Bielefeld ``"bf.."`` columns to standardized questionnaire
    item names, and combines two separate source blocks into one
    ``"stai{n}"`` item series.

    Parameters
    ----------
    questionnaire_file : str
        Path to the Bielefeld questionnaire Excel file.

    Returns
    -------
    info_df : pandas.DataFrame
        Participant-level table containing ``"participant_id"``, ``"age"``,
        ``"sex"``, and ``"handedness"`` for use in ``participants.tsv``.
    pheno_df : pandas.DataFrame
        Full parsed phenotype table with standardized column names,
        questionnaire items, and participant metadata.

    Raises
    ------
    ValueError
        Raised if any required columns such as ``"id"``, ``"age"``,
        ``"sex"``, ``"handedness"``, or ``"recorded_at"`` are missing after
        renaming.

    Notes
    -----
    This function reads an Excel file but does not write output files.

    Bielefeld-specific processing includes extracting the numeric part of
    ``"id"``, formatting ``"participant_id"`` as ``"sub-XXX"``, mapping
    categorical codes for ``"sex"`` and ``"handedness"``, and sorting rows
    by ``"participant_id"``.
    """

    # read excel file
    pheno_df = pd.read_excel(questionnaire_file).iloc[1:].copy()

    # normalize column names
    pheno_df.columns = pheno_df.columns.str.strip().str.lower()

    # map Bielefeld columns to standard names
    rename_cols = {
        "bf33_01": "id",
        "started": "recorded_at",
        "bf13_01": "age",
        "bf01_01": "sex",
        "bf12_01": "handedness",
    }

    add_block(rename_cols, "bf11", "gad")
    add_block(rename_cols, "bf03", "phq")
    add_block(rename_cols, "bf02", "bfi")
    add_block(rename_cols, "bf05", "soc")
    add_block(rename_cols, "bf06", "ius")

    # STAI-40 split across two source blocks
    n_stai = lab_pheno["stai"]
    half = n_stai // 2
    stai_name = f"stai{n_stai}"

    for i in range(1, half + 1):
        rename_cols[f"bf07_{i:02d}"] = f"{stai_name}_{i}"

    for i in range(1, half + 1):
        rename_cols[f"bf08_{i:02d}"] = f"{stai_name}_{half + i}"

    pheno_df = pheno_df.rename(columns=rename_cols)

    # Check required columns
    required = ["id", "age", "sex", "handedness", "recorded_at"]
    missing = [col for col in required if col not in pheno_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for col in ["sex", "handedness"]:
        pheno_df[col] = pheno_df[col].astype(int)

    # Map gender and handedness
    pheno_df["sex"] = pheno_df["sex"].map({1: "male", 2: "female"})
    pheno_df["handedness"] = pheno_df["handedness"].map({1: "left", 2: "right"})

    # Extract numeric ID and build participant_id
    pheno_df["id"] = (
        pheno_df["id"]
        .astype(str)
        .str.extract(r"(\d+)", expand=False)
        .astype(float)
        .astype("Int64")
    )

    pheno_df["participant_id"] = pheno_df["id"].apply(lambda x: f"sub-{int(x):03d}")

    # Sort by numeric ID
    pheno_df = pheno_df.sort_values("participant_id").reset_index(drop=True)

    # Extract only the relevant columns for participants.tsv
    info_df = pheno_df[
        [
            "participant_id",
            "age",
            "sex",
            "handedness"
        ]
    ].copy()

    return info_df, pheno_df


# BFI
def aggr_bfi_data(
        pheno_df: pd.DataFrame,
        phenotype_dir: str
    ) -> Optional[pd.DataFrame]:
    """
    Aggregate Bielefeld BFI questionnaire data and write the output TSV.

    Unlike the Bonn implementation, which renames lab-specific BFI source
    columns during aggregation, the Bielefeld pipeline expects BFI item
    columns to have already been standardized in ``parse_questionnaire_file``.
    It additionally appends ``"_r"`` to reverse-scored item names before
    writing the TSV.

    Parameters
    ----------
    pheno_df : pandas.DataFrame
        Phenotype table containing ``"participant_id"`` and standardized BFI
        item columns.
    phenotype_dir : str
        Output directory where the aggregated BFI TSV file is written.

    Returns
    -------
    subset : pandas.DataFrame or None
        DataFrame returned by ``common_write_tsv`` after writing the
        aggregated BFI TSV file, or ``None`` if no usable BFI columns are
        available.

    Notes
    -----
    This function writes a questionnaire TSV file to ``phenotype_dir``.

    Reverse-scored BFI items are renamed with a ``"_r"`` suffix before
    integer conversion, missing-column padding, and TSV writing.
    """

    # define settings
    current_quest = "bfi"
    n_items = lab_pheno.get(current_quest)
    id_key = f"{current_quest}{n_items}_"

    bfi_csv_cols = ["participant_id"] + [f"{id_key}{i}" for i in range(1, 61)]

    available_cols = [col for col in bfi_csv_cols if col in pheno_df.columns]
    if not available_cols:
        logger.warning(f"BFI: no columns available for participant")
        return None

    # reverse‐scoring
    reverse_set = {
        # Extraversion
        11, 16, 26, 31, 36, 51,
        # Agreeableness
        12, 17, 22, 37, 42, 47,
        # Conscientiousness
        3, 8, 23, 28, 48, 58,
        # Neuroticism
        4, 9, 24, 29, 44, 49,
        # Openness
        5, 25, 30, 45, 50, 55,
    }

    rename_mapping = {
        f"{id_key}{i}": f"{id_key}{i}{'_r' if i in reverse_set else ''}"
        for i in range(1, 61)
        if f"{id_key}{i}" in available_cols
    }

    subset = pheno_df[["participant_id"] + list(rename_mapping.keys())]
    subset = subset.rename(columns=rename_mapping)

    # make integer
    subset = convert_questionnaire_columns_to_int(
        subset,
        id_key
    )

    # throw error if indices don't match up
    subset = pad_missing_columns(subset, n_items, id_key)

    # write
    return common_write_tsv(
        subset=subset,
        phenotype_dir=phenotype_dir,
        id_key=id_key,
        language=language
    )


# GAD
def aggr_gad_data(
        pheno_df: pd.DataFrame,
        phenotype_dir: str
    ) -> Optional[pd.DataFrame]:
    """
    Aggregate Bielefeld GAD questionnaire data and write the output TSV.

    Compared with the Bonn implementation, which renames lab-specific GAD
    source columns during aggregation, the Bielefeld pipeline aggregates
    GAD items after they have already been standardized in
    ``parse_questionnaire_file``.

    Parameters
    ----------
    pheno_df : pandas.DataFrame
        Phenotype table containing ``"participant_id"`` and standardized GAD
        item columns.
    phenotype_dir : str
        Output directory where the aggregated GAD TSV file is written.

    Returns
    -------
    subset : pandas.DataFrame or None
        DataFrame returned by ``common_write_tsv`` after writing the
        aggregated GAD TSV file, or ``None`` if no usable GAD columns are
        available.

    Notes
    -----
    This function writes a questionnaire TSV file to ``phenotype_dir``.

    Aside from earlier Bielefeld-specific column standardization, the
    aggregation workflow is equivalent to the Bonn pipeline.
    """

    # define settings
    current_quest = "gad"
    n_items = lab_pheno.get(current_quest)
    id_key = f"{current_quest}{n_items}_"

    # pre-define columns
    cols = ["participant_id"] + [c for c in pheno_df.columns if id_key in c]

    # check if they exist
    available_cols = [col for col in cols if col in pheno_df.columns]

    if available_cols:

        # reformat names
        subset = pheno_df[available_cols]
        subset = subset.rename(
            columns=lambda c: rename_col(c, old_key=id_key, new_key=id_key)
        )

        # make integer
        subset = convert_questionnaire_columns_to_int(
            subset,
            id_key
        )

        # throw error if indices don't match up
        subset = pad_missing_columns(subset, n_items, id_key)

        # write
        return common_write_tsv(
            subset=subset,
            phenotype_dir=phenotype_dir,
            id_key=id_key,
            language=language
        )
    else:
        logging.warning(f"{current_quest.upper()}: No columns available for aggregated data")
        return None
    

# IUS
def aggr_ius_data(
        pheno_df: pd.DataFrame,
        phenotype_dir: str
    ) -> Optional[pd.DataFrame]:
    """
    Aggregate Bielefeld IUS questionnaire data and write the output TSV.

    Compared with the Bonn implementation, which renames lab-specific IUS
    source columns during aggregation, the Bielefeld pipeline aggregates IUS
    items after they have already been standardized in
    ``parse_questionnaire_file``.

    Parameters
    ----------
    pheno_df : pandas.DataFrame
        Phenotype table containing ``"participant_id"`` and standardized IUS
        item columns.
    phenotype_dir : str
        Output directory where the aggregated IUS TSV file is written.

    Returns
    -------
    subset : pandas.DataFrame or None
        DataFrame returned by ``common_write_tsv`` after writing the
        aggregated GAD TSV file, or ``None`` if no usable GAD columns are
        available.

    Notes
    -----
    This function writes a questionnaire TSV file to ``phenotype_dir``.

    Aside from earlier Bielefeld-specific column standardization, the
    aggregation workflow is equivalent to the Bonn pipeline.
    """

    # define settings
    current_quest = "ius"
    n_items = lab_pheno.get(current_quest)
    id_key = f"{current_quest}{n_items}_"

    # pre-define columns
    cols = ["participant_id"] + [c for c in pheno_df.columns if id_key in c]

    # check if they exist
    available_cols = [col for col in cols if col in pheno_df.columns]

    if available_cols:

        # reformat names
        subset = pheno_df[available_cols]
        subset = subset.rename(
            columns=lambda c: rename_col(c, old_key=id_key, new_key=id_key)
        )

        # make integer
        subset = convert_questionnaire_columns_to_int(
            subset,
            id_key
        )

        # throw error if indices don't match up
        subset = pad_missing_columns(subset, n_items, id_key)

        # write
        return common_write_tsv(
            subset=subset,
            phenotype_dir=phenotype_dir,
            id_key=id_key,
            language=language
        )
    else:
        logging.warning(f"{current_quest.upper()}: No columns available for aggregated data")
        return None


# PHQ
def aggr_phq_data(
        pheno_df: pd.DataFrame,
        phenotype_dir: str
    ) -> Optional[pd.DataFrame]:
    """
    Aggregate Bielefeld PHQ questionnaire data and write the output TSV.

    Compared with the Bonn implementation, which renames lab-specific PHQ
    source columns during aggregation, the Bielefeld pipeline aggregates PHQ
    items after they have already been standardized in
    ``parse_questionnaire_file``.

    Parameters
    ----------
    pheno_df : pandas.DataFrame
        Phenotype table containing ``"participant_id"`` and standardized PHQ
        item columns.
    phenotype_dir : str
        Output directory where the aggregated PHQ TSV file is written.

    Returns
    -------
    subset : pandas.DataFrame or None
        DataFrame returned by ``common_write_tsv`` after writing the
        aggregated GAD TSV file, or ``None`` if no usable GAD columns are
        available.

    Notes
    -----
    This function writes a questionnaire TSV file to ``phenotype_dir``.

    Aside from earlier Bielefeld-specific column standardization, the
    aggregation workflow is equivalent to the Bonn pipeline.
    """

    # define settings
    current_quest = "phq"
    n_items = lab_pheno.get(current_quest)
    id_key = f"{current_quest}{n_items}_"

    # pre-define columns
    cols = ["participant_id"] + [c for c in pheno_df.columns if id_key in c]

    # check if they exist
    available_cols = [col for col in cols if col in pheno_df.columns]

    if available_cols:

        # reformat names
        subset = pheno_df[available_cols]
        subset = subset.rename(
            columns=lambda c: rename_col(c, old_key=id_key, new_key=id_key)
        )

        # make integer
        subset = convert_questionnaire_columns_to_int(
            subset,
            id_key
        )

        # throw error if indices don't match up
        subset = pad_missing_columns(subset, n_items, id_key)

        # write
        return common_write_tsv(
            subset=subset,
            phenotype_dir=phenotype_dir,
            id_key=id_key,
            language=language
        )
    else:
        logging.warning(f"{current_quest.upper()}: No columns available for aggregated data")
        return None


# SOC
def aggr_soc_data(
        pheno_df: pd.DataFrame,
        phenotype_dir: str
    ) -> Optional[pd.DataFrame]:
    """
    Aggregate Bielefeld SOC questionnaire data and write the output TSV.

    Compared with the Bonn implementation, which renames lab-specific SOC
    source columns during aggregation, the Bielefeld pipeline aggregates SOC
    items after they have already been standardized in
    ``parse_questionnaire_file``.

    Parameters
    ----------
    pheno_df : pandas.DataFrame
        Phenotype table containing ``"participant_id"`` and standardized SOC
        item columns.
    phenotype_dir : str
        Output directory where the aggregated SOC TSV file is written.

    Returns
    -------
    subset : pandas.DataFrame or None
        DataFrame returned by ``common_write_tsv`` after writing the
        aggregated SOC TSV file, or ``None`` if no usable SOC columns are
        available.

    Notes
    -----
    This function writes a questionnaire TSV file to ``phenotype_dir``.

    Aside from earlier Bielefeld-specific column standardization, the
    aggregation workflow is equivalent to the Bonn pipeline.
    """

    # define settings
    current_quest = "soc"
    n_items = lab_pheno.get(current_quest)
    id_key = f"{current_quest}{n_items}_"

    # pre-define columns
    cols = ["participant_id"] + [c for c in pheno_df.columns if id_key in c]

    # check if they exist
    available_cols = [col for col in cols if col in pheno_df.columns]

    if available_cols:

        # reformat names
        subset = pheno_df[available_cols]
        subset = subset.rename(
            columns=lambda c: rename_col(c, old_key=id_key, new_key=id_key)
        )

        # make integer
        subset = convert_questionnaire_columns_to_int(
            subset,
            id_key
        )

        # throw error if indices don't match up
        subset = pad_missing_columns(subset, n_items, id_key)

        # write
        return common_write_tsv(
            subset=subset,
            phenotype_dir=phenotype_dir,
            id_key=id_key,
            language=language
        )
    else:
        logging.warning(f"{current_quest.upper()}: No columns available for aggregated data")
        return None


# STAI
def aggr_stai_data(
        pheno_df: pd.DataFrame,
        phenotype_dir: str
    ) -> Optional[pd.DataFrame]:
    """
    Aggregate Bielefeld STAI questionnaire data and write the output TSV.

    Unlike the Bonn implementation, which renames lab-specific STAI source
    columns during aggregation from one source prefix, the Bielefeld
    pipeline first merges two source blocks into one standardized
    ``"stai{n}_"`` item series in ``parse_questionnaire_file`` and then
    aggregates the combined items here.

    Parameters
    ----------
    pheno_df : pandas.DataFrame
        Phenotype table containing ``"participant_id"`` and standardized
        STAI item columns.
    phenotype_dir : str
        Output directory where the aggregated STAI TSV file is written.

    Returns
    -------
    subset : pandas.DataFrame or None
        DataFrame returned by ``common_write_tsv`` after writing the
        aggregated STAI TSV file, or ``None`` if no usable STAI columns are
        available.

    Notes
    -----
    This function writes a questionnaire TSV file to ``phenotype_dir``.

    Bielefeld-specific handling occurs before aggregation by combining STAI
    items from two questionnaire blocks into one ordered standardized
    series.
    """

    # define settings
    current_quest = "stai"
    n_items = lab_pheno.get(current_quest)
    id_key = f"{current_quest}{n_items}_"

    # pre-define columns
    cols = ["participant_id"] + [c for c in pheno_df.columns if id_key in c]

    # check if they exist
    available_cols = [col for col in cols if col in pheno_df.columns]

    if available_cols:

        # reformat names
        subset = pheno_df[available_cols]
        subset = subset.rename(
            columns=lambda c: rename_col(c, old_key=id_key, new_key=id_key)
        )

        # make integer
        subset = convert_questionnaire_columns_to_int(
            subset,
            id_key
        )

        # throw error if indices don't match up
        subset = pad_missing_columns(subset, n_items, id_key)

        # write
        return common_write_tsv(
            subset=subset,
            phenotype_dir=phenotype_dir,
            id_key=id_key,
            language=language
        )
    else:
        logging.warning(f"{current_quest.upper()}: No columns available for aggregated data")
        return None

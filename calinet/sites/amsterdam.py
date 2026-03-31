# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import pandas as pd
from calinet.imports import mat
from calinet.config import available_labs
from calinet.core.pheno import pad_missing_columns

from calinet.utils import (
    rename_col,
    common_write_tsv,
    convert_questionnaire_columns_to_int,
)

from typing import Any, Optional, Tuple

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
    Read an Amsterdam raw physiology ``.mat`` file for one participant.

    Unlike the Bonn implementation, which reads a single physiology
    ``.acq`` file with ``biopac.read_acq_file``, the Amsterdam pipeline reads 
    ``.mat`` file input using ``mat.read_mat_file``. The TTL channel contains
    multiple peaks, and are filtered between 60-95th percentile (exclude high 
    peaks denoting start-of-block and low peaks denoting US onsets). Arguments
    to ``mat.read_mat_file`` can be specified within the ``available_labs``
    dict in calinet.config.py (locate 'amsterdam' -> 'ChannelRegex')

    Parameters
    ----------
    raw_physio_acq : str
        Path to the Bonn ``.acq`` file to read.

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

    Channel selection is based on the Bonn lab configuration via
    ``available_labs.get(lab_name).get("ChannelRegex")``.
    """

    # Extract physiological data
    logger.info(f"Amsterdam .mat file: {raw_physio_acq}")

    # read acqknowledge file
    lab_name = __name__.split(".")[-1]

    args = available_labs.get(lab_name).get("ChannelRegex")
    
    logger.debug(f"Received arguments for Amsterdam file: {args}")
    try:
        res = mat.read_mat_file(raw_physio_acq, **args)
        logger.info("Loading mat-file successfull")
    except Exception as e:
        raise Exception(f"Error while Amsterdam '{raw_physio_acq}': {e}") from e
    
    # extract dataframe and sampling rate
    physio_df = res.df
    sr = res.sampling_rate_hz
    chan_info = None # -> will look for SamplingFrequency in metadata.csv

    return (physio_df, sr, chan_info)


def find_physio_acq_file(
        subject_raw_data_path: str,
        subject_name: str=None
    ) -> str:
    """
    Locate the Amsterdam raw physiology ``.mat`` file for one participant.

    Unlike the Bonn implementation, which reads a single physiology
    ``.acq`` file with ``biopac.read_acq_file``, the Amsterdam pipeline reads 
    ``.mat`` file input using ``mat.read_mat_file``. The mat-file is expected
    to be in the ssub-folder.

    Parameters
    ----------
    subject_raw_data_path : str
        Path to the participant-specific raw data directory containing the
        Amsterdam ``.mat`` file.
    subject_name : str
        Participant label used to derive the physiology file name. The value
        is split on ``"-"`` and the second token is inserted into the file
        name template.

    Returns
    -------
    physio_path : str
        Full path to the expected Amsterdam ``.mat`` file.

    Raises
    ------
    Exception
        Raised if ``subject_raw_data_path`` does not exist.
    FileNotFoundError
        Raised if the expected ``.mat`` file is not found.

    Notes
    -----
    This function does not read or write files.
    """

    if not os.path.isdir(subject_raw_data_path):
        raise Exception(f"No physio folder at {subject_raw_data_path!r}")
    
    matches = []

    for root, _, files in os.walk(subject_raw_data_path):
        for filename in files:
            if filename.lower().endswith(".mat"):
                full_path = os.path.join(root, filename)
                matches.append(full_path)

    if not matches:
        logger.warning(f"No MAT-files found in {subject_raw_data_path}")
        return None

    logger.info(f"Found {len(matches)} matching MAT file(s)")

    # if more matches exist, take largest file
    if len(matches)>1:
        mat_file = max(matches, key=os.path.getsize)
        size_mb = os.path.getsize(mat_file) / (1024 * 1024)
        logger.info(f"Selected largest MAT file of {size_mb:.2f} MB")

    mat_file = max(matches, key=os.path.getsize)
    return mat_file


def find_questionnaire_file(
        raw_data_dir: str
    ) -> Optional[str]:
    """
    Find the Bonn questionnaire source file under the raw data tree.

    The Bonn implementation searches recursively for a questionnaire export
    file called 'Calinet participant data'.

    Parameters
    ----------
    raw_data_dir : str
        Root directory to search recursively for the Bonn questionnaire
        source file.

    Returns
    -------
    questionnaire_file : str or None
        Full path to the first matching questionnaire file, or ``None`` if
        no matching file is found.

    Notes
    -----
    This function does not read questionnaire contents and does not write
    files.

    Matching is based on Bonn-specific file naming patterns.
    """
    
    questionnaire_file = None
    for root, _, files in os.walk(raw_data_dir):
        for filename in files:
            if "Calinet participant data" in filename:
                questionnaire_file = os.path.join(root, filename)
                break
    return questionnaire_file


def parse_questionnaire_file(
        questionnaire_file: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse the Bonn questionnaire file and standardize participant fields.

    This is the reference implementation for questionnaire parsing. It reads
    a Bonn questionnaire export (typically CSV), normalizes column names,
    and extracts participant-level metadata and questionnaire item data.

    Parameters
    ----------
    questionnaire_file : str
        Path to the Bonn questionnaire file.

    Returns
    -------
    info_df : pandas.DataFrame
        Participant-level table containing ``"participant_id"``, ``"age"``,
        ``"sex"``, and ``"handedness"`` for use in ``participants.tsv``.
    pheno_df : pandas.DataFrame
        Full parsed phenotype table including questionnaire items and
        participant metadata.

    Raises
    ------
    ValueError
        Raised if required columns such as ``"participant_id"``, ``"age"``,
        ``"sex"``, or ``"handedness"`` are missing.

    Notes
    -----
    This function reads a questionnaire file but does not write output
    files.

    Column normalization and mapping follow the Bonn reference schema.
    """

    # read excel file
    pheno_df = pd.read_excel(questionnaire_file)

    # Normalize column names
    pheno_df.columns = pheno_df.columns.str.strip().str.lower()

    # lab specific renaming
    pheno_df.rename(
        columns={
            "subject": "participant_id",
            "room_temp": "room_temperature",
            "room_humid": "humidity",
            "time": "recorded_at"
        },
        inplace=True
    )

    # Ensure the required columns exist
    required_columns = ["participant_id", "age", "sex", "handedness"]
    for col in required_columns:
        if col not in pheno_df.columns:
            raise ValueError(
                f"Missing required column '{col}' in questionnaire file: {questionnaire_file}"
            )

    handedness_mapping = {"right-handed": "right", "left-handed": "left"}

    pheno_df["handedness"] = pheno_df["handedness"].map(handedness_mapping).astype(object)
    
    # Convert participant_id to the format sub-01, sub-02, etc.
    pheno_df["participant_id"] = pheno_df["participant_id"].apply(lambda x: f"sub-{x:03d}")

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
    Aggregate Amsterdam BFI questionnaire data and write the output TSV.

    Compared with the Bonn implementation, the Amsterdam pipeline matches BFI
    item columns using the exact source prefix ``"bfi_"`` extracted from log
    files.

    Parameters
    ----------
    pheno_df : pandas.DataFrame
        Phenotype table containing ``"participant_id"`` and Amsterdam BFI item
        columns matched by the prefix ``"bfi_"``.
    phenotype_dir : str
        Output directory where the aggregated BFI TSV file is written.

    Returns
    -------
    subset : pandas.DataFrame or None
        DataFrame returned by ``common_write_tsv`` after writing the TSV
        file, or ``None`` if no usable BFI columns are available.

    Notes
    -----
    This function writes a questionnaire TSV file to ``phenotype_dir``.

    Amsterdam-specific matching uses the exact prefix ``"bfi_"`` and renames
    it to ``"bfi{n_items}_"`` before writing. :contentReference[oaicite:4]{index=4}
    """

    # define settings
    current_quest = "bfi"
    replace_key = f"{current_quest}_"
    n_items = lab_pheno.get(current_quest)
    id_key = f"{current_quest}{n_items}_"

    # pre-define columns
    cols = ["participant_id"] + [c for c in pheno_df.columns if replace_key in c]

    # check if they exist
    available_cols = [col for col in cols if col in pheno_df.columns]

    if available_cols:

        # reformat names
        subset = pheno_df[available_cols]
        subset = subset.rename(
            columns=lambda c: rename_col(c, old_key=replace_key, new_key=id_key)
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


# GAD
def aggr_gad_data(
        pheno_df: pd.DataFrame,
        phenotype_dir: str
    ) -> Optional[pd.DataFrame]:
    """
    Aggregate Amsterdam GAD questionnaire data and write the output TSV.

    Compared with the Bonn implementation, the Amsterdam pipeline matches GAD
    item columns using the exact source prefix ``"gad_"`` and excludes
    columns ending with ``"_total"`` before aggregation.

    Parameters
    ----------
    pheno_df : pandas.DataFrame
        Phenotype table containing ``"participant_id"`` and Amsterdam GAD item
        columns matched by the prefix ``"gad_"``.
    phenotype_dir : str
        Output directory where the aggregated GAD TSV file is written.

    Returns
    -------
    subset : pandas.DataFrame or None
        DataFrame returned by ``common_write_tsv`` after writing the TSV
        file, or ``None`` if no usable GAD columns are available.

    Notes
    -----
    This function writes a questionnaire TSV file to ``phenotype_dir``.

    Amsterdam-specific matching uses the exact prefix ``"gad_"`` and excludes
    columns ending in ``"_total"``. :contentReference[oaicite:5]{index=5}
    """

    # Remove columns that end with '_total'
    pheno_df = pheno_df[
        [col for col in pheno_df.columns if not col.endswith("_total")]
    ]

    # define settings
    current_quest = "gad"
    replace_key = f"{current_quest}_"
    n_items = lab_pheno.get(current_quest)
    id_key = f"{current_quest}{n_items}_"

    # pre-define columns
    cols = ["participant_id"] + [c for c in pheno_df.columns if replace_key in c]

    # check if they exist
    available_cols = [col for col in cols if col in pheno_df.columns]

    if available_cols:

        # reformat names
        subset = pheno_df[available_cols]
        subset = subset.rename(
            columns=lambda c: rename_col(c, old_key=replace_key, new_key=id_key)
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
    Aggregate Amsterdam IUS questionnaire data and write the output TSV.

    Compared with the Bonn implementation, the Amsterdam pipeline matches IUS
    item columns using the exact source prefix ``"ius_"`` and excludes
    columns ending with ``"_total"`` before aggregation.

    Parameters
    ----------
    pheno_df : pandas.DataFrame
        Phenotype table containing ``"participant_id"`` and Amsterdam IUS item
        columns matched by the prefix ``"ius_"``.
    phenotype_dir : str
        Output directory where the aggregated IUS TSV file is written.

    Returns
    -------
    subset : pandas.DataFrame or None
        DataFrame returned by ``common_write_tsv`` after writing the TSV
        file, or ``None`` if no usable IUS columns are available.

    Notes
    -----
    This function writes a questionnaire TSV file to ``phenotype_dir``.

    Amsterdam-specific matching uses the exact prefix ``"ius_"`` and excludes
    columns ending in ``"_total"``. :contentReference[oaicite:6]{index=6}
    """

    # Remove columns that end with '_total'
    pheno_df = pheno_df[
        [col for col in pheno_df.columns if not col.endswith("_total")]
    ]

    # define settings
    current_quest = "ius"
    replace_key = f"{current_quest}_"
    n_items = lab_pheno.get(current_quest)
    id_key = f"{current_quest}{n_items}_"

    # pre-define columns
    cols = ["participant_id"] + [c for c in pheno_df.columns if replace_key in c]

    # check if they exist
    available_cols = [col for col in cols if col in pheno_df.columns]

    if available_cols:

        # reformat names
        subset = pheno_df[available_cols]
        subset = subset.rename(
            columns=lambda c: rename_col(c, old_key=replace_key, new_key=id_key)
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
    Aggregate Amsterdam PHQ questionnaire data and write the output TSV.

    Compared with the Bonn implementation, the Amsterdam pipeline matches PHQ
    item columns using the exact source prefix ``"phq_"`` and excludes
    columns ending with ``"_total"`` before aggregation.

    Parameters
    ----------
    pheno_df : pandas.DataFrame
        Phenotype table containing ``"participant_id"`` and Amsterdam PHQ item
        columns matched by the prefix ``"phq_"``.
    phenotype_dir : str
        Output directory where the aggregated PHQ TSV file is written.

    Returns
    -------
    subset : pandas.DataFrame or None
        DataFrame returned by ``common_write_tsv`` after writing the TSV
        file, or ``None`` if no usable PHQ columns are available.

    Notes
    -----
    This function writes a questionnaire TSV file to ``phenotype_dir``.

    Amsterdam-specific matching uses the exact prefix ``"phq_"`` and excludes
    columns ending in ``"_total"``. :contentReference[oaicite:7]{index=7}
    """

    # Remove columns that end with '_total'
    pheno_df = pheno_df[
        [col for col in pheno_df.columns if not col.endswith("_total")]
    ]

    # define settings
    current_quest = "phq"
    replace_key = f"{current_quest}_"
    n_items = lab_pheno.get(current_quest)
    id_key = f"{current_quest}{n_items}_"

    # pre-define columns
    cols = ["participant_id"] + [c for c in pheno_df.columns if replace_key in c]

    # check if they exist
    available_cols = [col for col in cols if col in pheno_df.columns]

    if available_cols:

        # reformat names
        subset = pheno_df[available_cols]
        subset = subset.rename(
            columns=lambda c: rename_col(c, old_key=replace_key, new_key=id_key)
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
    Aggregate Amsterdam SOC questionnaire data and write the output TSV.

    Unlike the Bonn implementation, which may use different SOC naming,
    the Amsterdam pipeline matches SOC item columns using the exact source
    prefix ``"soc_"`` extracted from log files.

    Parameters
    ----------
    pheno_df : pandas.DataFrame
        Phenotype table containing ``"participant_id"`` and Amsterdam SOC item
        columns matched by the prefix ``"soc_"``.
    phenotype_dir : str
        Output directory where the aggregated SOC TSV file is written.

    Returns
    -------
    subset : pandas.DataFrame or None
        DataFrame returned by ``common_write_tsv`` after writing the TSV
        file, or ``None`` if no usable SOC columns are available.

    Notes
    -----
    This function writes a questionnaire TSV file to ``phenotype_dir``.

    Amsterdam-specific matching uses the exact prefix ``"soc_"`` and renames
    it to ``"soc{n_items}_"`` before writing. :contentReference[oaicite:8]{index=8}
    """

    # Remove columns that end with '_total'
    pheno_df = pheno_df[
        [col for col in pheno_df.columns if not col.endswith("_total")]
    ]

    # define settings
    current_quest = "soc"
    replace_key = f"soc_"
    n_items = lab_pheno.get(current_quest)
    id_key = f"{current_quest}{n_items}_"

    # pre-define columns
    cols = ["participant_id"] + [c for c in pheno_df.columns if replace_key in c]

    # check if they exist
    available_cols = [col for col in cols if col in pheno_df.columns]

    if available_cols:

        # reformat names
        subset = pheno_df[available_cols]
        subset = subset.rename(
            columns=lambda c: rename_col(c, old_key=replace_key, new_key=id_key)
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
    Aggregate Amsterdam STAI questionnaire data and write the output TSV.

    Unlike the Bonn implementation, which uses Bonn-specific STAI naming,
    the Amsterdam pipeline matches STAI item columns using the exact source
    prefix ``"stait_"``.

    Parameters
    ----------
    pheno_df : pandas.DataFrame
        Phenotype table containing ``"participant_id"`` and Amsterdam STAI
        item columns matched by the prefix ``"stait_"``.
    phenotype_dir : str
        Output directory where the aggregated STAI TSV file is written.

    Returns
    -------
    subset : pandas.DataFrame or None
        DataFrame returned by ``common_write_tsv`` after writing the TSV
        file, or ``None`` if no usable STAI columns are available.

    Notes
    -----
    This function writes a questionnaire TSV file to ``phenotype_dir``.

    Amsterdam-specific matching uses the exact prefix ``"stait_"`` and
    renames it to ``"stai{n_items}_"`` before writing. :contentReference[oaicite:9]{index=9}
    """

    # define settings
    current_quest = "stai"
    replace_key = f"{current_quest}t_"
    n_items = lab_pheno.get(current_quest)
    id_key = f"{current_quest}{n_items}_"

    # pre-define columns
    cols = ["participant_id"] + [c for c in pheno_df.columns if replace_key in c]

    # check if they exist
    available_cols = [col for col in cols if col in pheno_df.columns]

    if available_cols:

        # reformat names
        subset = pheno_df[available_cols]
        subset = subset.rename(
            columns=lambda c: rename_col(c, old_key=replace_key, new_key=id_key)
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

# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import re
import glob
import pandas as pd
from calinet.imports import pspm
from calinet.config import available_labs
from calinet.core.pheno import (
    rename_col,
    common_write_tsv,
    pad_missing_columns,
    convert_questionnaire_columns_to_int,
)

from typing import Any, Optional, Tuple, Union, List

import logging
logger = logging.getLogger(__name__)

# lab-specific pheno info
lab_name = __name__.split(".")[-1]
lab_pheno = available_labs.get(lab_name).get("Phenotype")
language = lab_pheno.get("Language")


opd = os.path.dirname


def read_raw_physio_file(
        raw_physio_acq: Union[str, List[str]]
    ) -> Tuple[pd.DataFrame, float, Any]:
    """
    Read one or more London raw physiology PsPM ``.mat`` files.

    Unlike the Bonn implementation, which reads a single physiology
    ``.acq`` file with ``biopac.read_acq_file``, the London pipeline reads
    PsPM ``.mat`` file input using ``pspm.read_pspm_files``. London data may
    be provided as one file or as multiple files, for example when the two
    tasks are stored separately.

    Parameters
    ----------
    raw_physio_acq : str or list of str
        Path to one London PsPM ``.mat`` file, or list of paths to multiple
        London PsPM ``.mat`` files to read together.

    Returns
    -------
    physio_df : pandas.DataFrame
        Physiology data extracted from the input PsPM file or files.
    sr : float
        Sampling frequency in Hz returned by the PsPM reader.
    chan_info : Any
        Channel metadata returned by ``pspm.read_pspm_files``.

    Raises
    ------
    Exception
        Raised if any input PsPM file cannot be read.

    Notes
    -----
    This function does not write files.

    London-specific physiology import uses ``pspm.read_pspm_files`` instead
    of the Bonn ``biopac.read_acq_file`` implementation.
    """

    # Extract physiological data
    logger.info(f"Reading .mat file(s): {raw_physio_acq}")

    try:
        # read PsPM files
        res = pspm.read_pspm_files(raw_physio_acq)
        logger.info("Loading PsPM-file(s) successfull")
    except Exception as e:
        raise Exception(f"Error while reading '{raw_physio_acq}': {e}") from e
    
    # extract dataframe and sampling rate
    physio_df = res.df
    sr = res.sampling_rate_hz
    chan_info = res.channel_info

    return (physio_df, sr, chan_info)


def find_physio_acq_file(
        subject_raw_data_path: str,
        subject_name: str
    ) -> List[str]:
    """
    Locate London raw physiology PsPM ``.mat`` files for one participant.

    Unlike the Bonn implementation, which expects a single
    ``"CALINET_Template{subject_id}.acq"`` file, the London pipeline looks
    directly inside ``subject_raw_data_path`` for one or more files matching
    the glob pattern ``"pspm_*.mat"``.

    Parameters
    ----------
    subject_raw_data_path : str
        Path to the participant-specific raw data directory expected to
        contain London PsPM files.
    subject_name : str
        Participant label. This argument is accepted for interface
        compatibility with the Bonn reference implementation but is not used
        for file matching in the London pipeline.

    Returns
    -------
    physio_files : list of str
        List of full paths to all matching London PsPM files found in
        ``subject_raw_data_path``.

    Raises
    ------
    FileNotFoundError
        Raised if no files matching ``"pspm_*.mat"`` are found in
        ``subject_raw_data_path``.

    Notes
    -----
    This function does not read or write files.

    London-specific discovery uses the exact filename prefix
    ``"pspm_"`` and the exact filename suffix ``".mat"``.
    """

    # london has PsPM-files; sometimes split for the two tasks
    pattern = os.path.join(subject_raw_data_path, f"pspm_*.mat")
    matches = glob.glob(pattern)

    if not matches:
        raise FileNotFoundError(f"No pspm_*.mat file(s) found in {subject_raw_data_path}")

    return matches

    
def find_questionnaire_file(
        raw_data_dir: str
    ) -> List[str]:
    """
    Find London questionnaire CSV files under the ``"RedCap"`` directory.

    Unlike the Bonn implementation, which searches for a single
    questionnaire source file, the London pipeline searches recursively
    under ``raw_data_dir/RedCap`` and returns all matching questionnaire
    files for later merging.

    Parameters
    ----------
    raw_data_dir : str
        Root raw data directory containing the London ``"RedCap"``
        questionnaire subdirectory.

    Returns
    -------
    questionnaire_file : list of str
        List of full paths to questionnaire files whose names contain the
        substring ``"calinet"`` and end with ``".csv"``.

    Notes
    -----
    This function does not read questionnaire contents and does not write
    files.

    London-specific matching uses the filename substring ``"calinet"``,
    the filename suffix ``".csv"``, and excludes any files located in
    directory paths containing ``"exclude"``.
    """

    # Look for the questionnaire file in the base folder
    questionnaire_file = None
    for root, _, files in os.walk(raw_data_dir):
        for filename in files:
            if "questionnaire" in filename.lower() and filename.lower().endswith(
                ".csv"
            ):
                questionnaire_file = os.path.join(root, filename)
                break
    return questionnaire_file

    
def find_questionnaire_file(raw_data_dir):
    """
    Look in raw_data_dir/questionnaires for an .xlsx file containing 'data_test'
    """
    
    pheno_dir = os.path.join(raw_data_dir, "RedCap")
    questionnaire_file = []
    for root, _, files in os.walk(pheno_dir):
        for filename in files:
            if "exclude" not in root.lower():
                if "calinet" in filename.lower() and filename.lower().endswith(
                    ".csv"
                ):
                    questionnaire_file.append(os.path.join(root, filename))

    return questionnaire_file


def parse_single_file(
        file: str
    ) -> pd.DataFrame:
    """
    Parse one London questionnaire CSV file.

    Unlike the Bonn implementation, which parses a single questionnaire
    export for the full site, the London pipeline first reads multiple
    RedCap CSV files separately and standardizes each file before merging.

    Parameters
    ----------
    file : str
        Path to one London questionnaire CSV file.

    Returns
    -------
    pheno_df : pandas.DataFrame
        Parsed questionnaire table with normalized lowercase column names
        and a derived ``"participant_id"`` column.

    Notes
    -----
    This function reads a CSV file but does not write output files.

    London-specific processing derives ``"participant_id"`` from
    ``"record_id"`` using the zero-padded format ``"sub-XXX"``.
    """

    # read excel file
    pheno_df = pd.read_csv(file, delimiter=",")

    # Normalize column names
    pheno_df.columns = pheno_df.columns.str.strip().str.lower()

    # Convert participant_id to the format sub-01, sub-02, etc.
    pheno_df["participant_id"] = pheno_df["record_id"].apply(lambda x: f"sub-{x:03d}")

    return pheno_df


def combine_redcap_files(
        files: List[str]
    ) -> pd.DataFrame:
    """
    Combine multiple London RedCap questionnaire CSV files into one table.

    Unlike the Bonn implementation, which processes one questionnaire export,
    the London pipeline reads multiple questionnaire files and merges them
    column-wise on ``"participant_id"``.

    Parameters
    ----------
    files : list of str
        Paths to London questionnaire CSV files to parse and merge.

    Returns
    -------
    merged_df : pandas.DataFrame
        Combined questionnaire table produced by concatenating parsed input
        files along columns after indexing by ``"participant_id"``.

    Raises
    ------
    ValueError
        Raised if ``files`` is empty or if no parsed DataFrames are
        available to concatenate.

    Notes
    -----
    This function reads multiple CSV files but does not write output files.

    London-specific merging uses ``pd.concat(..., axis=1, join="inner")``
    after setting ``"participant_id"`` as the index in each parsed file.
    """

    pheno_df = []
    for f in files:
        pheno_df.append(parse_single_file(f))

    if len(pheno_df)>0:
        merged_df = pd.concat(
            [df.set_index("participant_id") for df in pheno_df],
            axis=1,
            join="inner"
        ).reset_index()

        return merged_df
    else:
        raise ValueError(f"Empty list of dataframes, cannot concatenate")
    

def parse_questionnaire_file(
        questionnaire_file: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse London questionnaire files and standardize participant fields.

    Unlike the Bonn implementation, which parses one questionnaire export,
    the London pipeline first merges multiple RedCap CSV files into one
    phenotype table and then applies compatibility renaming and value
    mapping.

    Parameters
    ----------
    questionnaire_file : list of str
        Paths to London questionnaire CSV files to combine and parse.

    Returns
    -------
    info_df : pandas.DataFrame
        Participant-level table containing ``"participant_id"``, ``"age"``,
        ``"sex"``, and ``"handedness"`` for use in ``participants.tsv``.
    pheno_df : pandas.DataFrame
        Full parsed phenotype table with merged questionnaire item columns
        and standardized participant metadata.

    Raises
    ------
    ValueError
        Raised if any required columns such as ``"record_id"``, ``"age"``,
        ``"sex"``, or ``"handedness"`` are missing after merging.

    Notes
    -----
    This function reads multiple CSV files but does not write output files.

    London-specific processing renames ``"record_id_timestamp"`` to
    ``"recorded_at"``, maps ``1`` and ``2`` in ``"sex"`` to ``"male"``
    and ``"female"``, maps ``1`` and ``2`` in ``"handedness"`` to
    ``"left"`` and ``"right"``, removes duplicated columns after merging,
    and sorts rows by ``"participant_id"``.
    """

    # combine single questionnaire files into 1 dataframe
    pheno_df = combine_redcap_files(questionnaire_file)
    pheno_df = pheno_df.loc[:, ~pheno_df.columns.duplicated()]

    # rename for compatibility
    pheno_df.rename(columns={
        "record_id_timestamp": "recorded_at",
    }, inplace=True)

    # Ensure the required columns exist
    required_columns = ["record_id", "age", "sex", "handedness"]
    for col in required_columns:
        if col not in pheno_df.columns:
            raise ValueError(
                f"Missing required column '{col}' in questionnaire file: {questionnaire_file}"
            )

    # ensure int
    for col in ["sex"]:
        pheno_df[col] = pheno_df[col].astype("Int64")

    # Map gender and handedness values for all rows
    gender_mapping = {1: "male", 2: "female"}
    handedness_mapping = {1: "left", 2: "right"}

    pheno_df["sex"] = pheno_df["sex"].map(gender_mapping).astype(object)
    pheno_df["handedness"] = pheno_df["handedness"].map(handedness_mapping).astype(object)
    
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


def filter_columns(
        df: pd.DataFrame,
        key: str="bfi",
        n_items: Optional[int]=None
    ) -> List[str]:
    """
    Select London questionnaire item columns for one scale, preferring
    reverse-scored columns when available.

    Unlike the Bonn implementation, which aggregates questionnaire columns
    primarily by simple prefix matching, the London pipeline uses explicit
    regular-expression matching for item columns of the form
    ``"{key}_{number}"`` and ``"{key}_{number}_rev"``.

    Parameters
    ----------
    df : pandas.DataFrame
        Phenotype table containing questionnaire item columns.
    key : str, default="bfi"
        Questionnaire key used to build the matching patterns. For example,
        ``"bfi"`` matches ``"bfi_1"`` and ``"bfi_1_rev"``.
    n_items : int or None, default=None
        Expected number of matched item columns. If provided, the number of
        selected columns must equal this value.

    Returns
    -------
    keep_cols : list of str
        Ordered list of selected questionnaire item columns. For each item
        number, the ``"{key}_{number}_rev"`` column is preferred over the
        base ``"{key}_{number}"`` column when both are present.

    Raises
    ------
    ValueError
        Raised if no columns match the expected patterns for ``key``.
    ValueError
        Raised if ``n_items`` is provided and the number of matched columns
        does not equal ``n_items``.

    Notes
    -----
    This function does not read or write files.

    London-specific matching uses the exact base pattern
    ``"^{key}_(\\d+)$"`` and the exact reverse-scored pattern
    ``"^{key}_(\\d+)_rev$"``.
    """

    cols = df.columns

    # base score columns like bfi30_1, bfi30_2, ...
    base_pat = re.compile(rf"^{key}_(\d+)$")
    rev_pat  = re.compile(rf"^{key}_(\d+)_rev$")

    base_items = {}
    rev_items = {}

    for col in cols:
        m_base = base_pat.match(col)
        if m_base:
            base_items[int(m_base.group(1))] = col
            continue

        m_rev = rev_pat.match(col)
        if m_rev:
            rev_items[int(m_rev.group(1))] = col

    # keep _rev when both exist, otherwise keep the base column
    keep_cols = [rev_items.get(i, base_items[i]) for i in sorted(base_items)]
    
    if len(keep_cols)<1:
        raise ValueError(f"{key} resulted in empty list")
    
    if n_items is not None:
        if len(keep_cols) != n_items:
            raise ValueError(f"Not enough columns ({len(keep_cols)}) for the number of items ({n_items})")
        
    return keep_cols

# BFI
def aggr_bfi_data(
        pheno_df: pd.DataFrame,
        phenotype_dir: str
    ) -> Optional[pd.DataFrame]:
    """
    Aggregate London BFI questionnaire data and write the output TSV.

    Unlike the Bonn implementation, which aggregates BFI item columns using
    Bonn-specific matching rules, the London pipeline selects BFI items with
    ``filter_columns`` using the exact source prefixes ``"bfi_"`` and
    ``"bfi_{number}_rev"``. When both versions exist for the same item,
    London keeps the reverse-scored ``"_rev"`` column.

    Parameters
    ----------
    pheno_df : pandas.DataFrame
        Phenotype table containing ``"participant_id"`` and London BFI item
        columns matched from the exact patterns ``"bfi_{number}"`` and
        ``"bfi_{number}_rev"``.
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

    London-specific renaming replaces the source prefix ``"bfi_"`` with the
    standardized output prefix ``"bfi{n_items}_"`` after column selection.
    """

    # define settings
    current_quest = "bfi"
    replace_key = f"{current_quest}_"
    n_items = lab_pheno.get(current_quest)
    id_key = f"{current_quest}{n_items}_"

    # pre-define columns
    bfi_cols = filter_columns(
        pheno_df,
        current_quest,
        n_items=n_items
    )

    cols = ["participant_id"] + bfi_cols
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
    Aggregate London GAD questionnaire data and write the output TSV.

    Unlike the Bonn implementation, which aggregates GAD item columns using
    Bonn-specific matching rules, the London pipeline selects GAD items with
    ``filter_columns`` using the exact source prefixes ``"gad_"`` and
    ``"gad_{number}_rev"``. When both versions exist for the same item,
    London keeps the reverse-scored ``"_rev"`` column.

    Parameters
    ----------
    pheno_df : pandas.DataFrame
        Phenotype table containing ``"participant_id"`` and London GAD item
        columns matched from the exact patterns ``"gad_{number}"`` and
        ``"gad_{number}_rev"``.
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

    London-specific renaming replaces the source prefix ``"gad_"`` with the
    standardized output prefix ``"gad{n_items}_"`` after column selection.
    """

    # define settings
    current_quest = "gad"
    replace_key = f"{current_quest}_"
    n_items = lab_pheno.get(current_quest)
    id_key = f"{current_quest}{n_items}_"

    # pre-define columns
    cols = filter_columns(
        pheno_df,
        current_quest,
        n_items=n_items
    )

    cols = ["participant_id"] + cols
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
    Aggregate London IUS questionnaire data and write the output TSV.

    Unlike the Bonn implementation, which aggregates IUS item columns using
    Bonn-specific matching rules, the London pipeline selects IUS items with
    ``filter_columns`` using the exact source prefixes ``"ius_"`` and
    ``"ius_{number}_rev"``. When both versions exist for the same item,
    London keeps the reverse-scored ``"_rev"`` column.

    Parameters
    ----------
    pheno_df : pandas.DataFrame
        Phenotype table containing ``"participant_id"`` and London IUS item
        columns matched from the exact patterns ``"ius_{number}"`` and
        ``"ius_{number}_rev"``.
    phenotype_dir : str
        Output directory where the aggregated IUS TSV file is written.

    Returns
    -------
    subset : pandas.DataFrame or None
        DataFrame returned by ``common_write_tsv`` after writing the
        aggregated IUS TSV file, or ``None`` if no usable IUS columns are
        available.

    Notes
    -----
    This function writes a questionnaire TSV file to ``phenotype_dir``.

    London-specific renaming replaces the source prefix ``"ius_"`` with the
    standardized output prefix ``"ius{n_items}_"`` after column selection.
    """

    # define settings
    current_quest = "ius"
    replace_key = f"{current_quest}_"
    n_items = lab_pheno.get(current_quest)
    id_key = f"{current_quest}{n_items}_"

    # pre-define columns
    cols = filter_columns(
        pheno_df,
        current_quest,
        n_items=n_items
    )

    cols = ["participant_id"] + cols
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
    Aggregate London PHQ questionnaire data and write the output TSV.

    Unlike the Bonn implementation, which aggregates PHQ item columns using
    Bonn-specific matching rules, the London pipeline selects PHQ items with
    ``filter_columns`` using the exact source prefixes ``"phq_"`` and
    ``"phq_{number}_rev"``. When both versions exist for the same item,
    London keeps the reverse-scored ``"_rev"`` column.

    Parameters
    ----------
    pheno_df : pandas.DataFrame
        Phenotype table containing ``"participant_id"`` and London PHQ item
        columns matched from the exact patterns ``"phq_{number}"`` and
        ``"phq_{number}_rev"``.
    phenotype_dir : str
        Output directory where the aggregated PHQ TSV file is written.

    Returns
    -------
    subset : pandas.DataFrame or None
        DataFrame returned by ``common_write_tsv`` after writing the
        aggregated PHQ TSV file, or ``None`` if no usable PHQ columns are
        available.

    Notes
    -----
    This function writes a questionnaire TSV file to ``phenotype_dir``.

    London-specific renaming replaces the source prefix ``"phq_"`` with the
    standardized output prefix ``"phq{n_items}_"`` after column selection.
    """

    # define settings
    current_quest = "phq"
    replace_key = f"{current_quest}_"
    n_items = lab_pheno.get(current_quest)
    id_key = f"{current_quest}{n_items}_"

    # pre-define columns
    cols = filter_columns(
        pheno_df,
        current_quest,
        n_items=n_items
    )

    cols = ["participant_id"] + cols
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
    Aggregate London SOC questionnaire data and write the output TSV.

    Unlike the Bonn implementation, which aggregates SOC items using Bonn
    source naming, the London pipeline matches SOC questionnaire columns
    using the exact source prefix ``"midi_"`` and keeps only columns that
    match the exact numeric pattern ``"midi_{number}"``.

    Parameters
    ----------
    pheno_df : pandas.DataFrame
        Phenotype table containing ``"participant_id"`` and London SOC item
        columns matched from the exact prefix ``"midi_"`` and filtered to
        the exact pattern ``"midi_{number}"``.
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

    London-specific renaming replaces the source prefix ``"midi_"`` with the
    standardized output prefix ``"soc{n_items}_"`` after exact regex
    filtering to numeric item columns only.
    """

    # define settings
    current_quest = "soc"
    replace_key = "midi_"
    n_items = lab_pheno.get(current_quest)
    id_key = f"{current_quest}{n_items}_"

    # pre-define columns
    cols = ["participant_id"] + [c for c in pheno_df.columns if replace_key in c]

    # remove unwanted columns
    pheno_df = pheno_df.filter(regex=rf'^(participant_id|{replace_key}\d+)$')

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
    Aggregate London STAI questionnaire data and write the output TSV.

    Unlike the Bonn implementation, which aggregates STAI item columns using
    Bonn-specific matching rules, the London pipeline selects STAI items
    with ``filter_columns`` using the exact source prefixes ``"stai_"`` and
    ``"stai_{number}_rev"``. When both versions exist for the same item,
    London keeps the reverse-scored ``"_rev"`` column.

    Parameters
    ----------
    pheno_df : pandas.DataFrame
        Phenotype table containing ``"participant_id"`` and London STAI item
        columns matched from the exact patterns ``"stai_{number}"`` and
        ``"stai_{number}_rev"``.
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

    London-specific renaming replaces the source prefix ``"stai_"`` with the
    standardized output prefix ``"stai{n_items}_"`` after column selection.
    """

    # define settings
    current_quest = "stai"
    replace_key = f"{current_quest}_"
    n_items = lab_pheno.get(current_quest)
    id_key = f"{current_quest}{n_items}_"

    # pre-define columns
    cols = filter_columns(
        pheno_df,
        current_quest,
        n_items=n_items
    )

    cols = ["participant_id"] + cols
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

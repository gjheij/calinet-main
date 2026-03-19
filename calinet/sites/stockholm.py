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
    append_acq_date_to_df,
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
    Read a Stockholm raw physiology ``.acq`` file.

    This implementation is equivalent to the Bonn pipeline in that it reads
    a single ``.acq`` file with ``biopac.read_acq_file`` and returns the
    extracted signal table, sampling rate, and selected channel metadata.

    Parameters
    ----------
    raw_physio_acq : str
        Path to the Stockholm ``.acq`` file to read.

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

    Channel selection is resolved from the Stockholm lab configuration via
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
    Locate the Stockholm raw physiology ``.acq`` file for one participant.

    Unlike the Bonn implementation, which expects a fixed filename pattern
    of the form ``"CALINET_Template{subject_id}.acq"``, the Stockholm
    pipeline scans ``subject_raw_data_path`` for the first file whose name
    starts with ``"CALINET_{subject_id}"`` and ends with ``".acq"``.

    Parameters
    ----------
    subject_raw_data_path : str
        Path to the participant-specific raw data directory expected to
        contain the Stockholm ``.acq`` file.
    subject_name : str
        Participant label whose last three characters are used as
        ``subject_id`` for filename matching.

    Returns
    -------
    physio_path : str
        Full path to the matching Stockholm ``.acq`` file.

    Raises
    ------
    FileNotFoundError
        Raised if no matching ``.acq`` file is found in
        ``subject_raw_data_path``.

    Notes
    -----
    This function does not read or write files.

    Stockholm-specific discovery uses the exact filename prefix
    ``"CALINET_{subject_id}"`` and accepts any matching ``".acq"`` suffix.
    """

    # go up one level from the subject folder to get the raw_data_dir
    subject_id = subject_name[-3:]
    for file in os.listdir(subject_raw_data_path):
        if file.startswith(f"CALINET_{subject_id}") and file.lower().endswith(".acq"):
            file_name = os.path.join(subject_raw_data_path, file)
            break

    # build expected filename, e.g. "501" → "501.acq"
    if file_name and os.path.isfile(file_name):
        return file_name
    else:
        raise FileNotFoundError(f".acq file {file_name} not found in {subject_raw_data_path!r}")


def find_questionnaire_file(
        raw_data_dir: str
    ) -> Optional[str]:
    """
    Find the Stockholm questionnaire text file under the raw data tree.

    Unlike the Bonn implementation, which searches Bonn-specific
    questionnaire exports, the Stockholm pipeline walks ``raw_data_dir``
    recursively and returns the first file whose name contains
    ``"questionnaire"`` and ends with ``".txt"``.

    Parameters
    ----------
    raw_data_dir : str
        Root directory to search recursively for the Stockholm questionnaire
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

    Matching is based on the Stockholm-specific filename substring
    ``"questionnaire"`` and the exact filename suffix ``".txt"``.
    """
    
    questionnaire_file = None
    for root, _, files in os.walk(raw_data_dir):
        for filename in files:
            if "questionnaire" in filename.lower() and filename.lower().endswith(
                ".txt"
            ):
                questionnaire_file = os.path.join(root, filename)
                break
        if questionnaire_file:
            return questionnaire_file


def parse_questionnaire_file(
        questionnaire_file: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse the Stockholm questionnaire file and standardize participant fields.

    Unlike the Bonn implementation, which parses a Bonn questionnaire
    export, the Stockholm pipeline reads a tab-delimited text file, drops
    fully blank columns, normalizes column names to lowercase, and derives
    ``"participant_id"`` from ``"ppn"``.

    Parameters
    ----------
    questionnaire_file : str
        Path to the Stockholm questionnaire text file.

    Returns
    -------
    info_df : pandas.DataFrame
        Participant-level table containing ``"participant_id"``, ``"age"``,
        ``"sex"``, and ``"handedness"`` for use in ``participants.tsv``.
    pheno_df : pandas.DataFrame
        Full parsed phenotype table with standardized participant metadata
        and questionnaire item columns.

    Raises
    ------
    ValueError
        Raised if any required columns such as ``"participant_id"``,
        ``"age"``, ``"sex"``, or ``"handedness"`` are missing after
        normalization.

    Notes
    -----
    This function reads a tab-delimited text file but does not write output
    files.

    Stockholm-specific processing converts ``"ppn"`` to zero-padded
    participant labels in the form ``"sub-XXX"``, maps ``0`` and ``1`` in
    ``"sex"`` to ``"M"`` and ``"F"``, maps ``1`` and ``0`` in
    ``"handedness"`` to ``"left"`` and ``"right"``, and sorts rows by
    ``"participant_id"``.
    """

    # read excel file
    pheno_df = pd.read_csv(questionnaire_file, delimiter="\t")

    # Drop columns which are unnamed and completely blank
    pheno_df.dropna(axis=1, how="all", inplace=True)

    pheno_df.columns = pheno_df.columns.str.strip().str.lower()
    pheno_df.dropna(subset=["ppn"], inplace=True)

    # Ensure PPN column has uniform three digits
    pheno_df["participant_id"] = pheno_df["ppn"].apply(lambda x: f"sub-{int(x):03d}")

    # Ensure the required columns exist
    required_columns = ["participant_id", "age", "sex", "handedness"]
    for col in required_columns:
        if col not in pheno_df.columns:
            raise ValueError(
                f"Missing required column '{col}' in questionnaire file: {questionnaire_file}"
            )

    # rename bunch of columns
    pheno_df.rename(
        columns={
            "room_humidity": "humidity",
            "time_of_day": "recorded_at"
        },
        inplace=True
    )

    # Convert questionnaire time-of-day values like 10.21 -> 10:21:00
    s = pheno_df["recorded_at"].astype(str).str.strip()

    extracted = s.str.extract(r"^(?P<hour>\d{1,2})\.(?P<minute>\d{1,2})$")
    time_str = (
        extracted["hour"].str.zfill(2)
        + ":"
        + extracted["minute"].str.zfill(2)
        + ":00"
    )

    # Map each participant to the date of their physiology acquisition file
    raw_path = os.path.dirname(questionnaire_file)
    pheno_df = append_acq_date_to_df(
        pheno_df,
        raw_path,
    )
        
    # Combine acquisition date + parsed time
    date_str = pheno_df["acq_date"].dt.strftime("%Y-%m-%d")
    pheno_df["recorded_at"] = pd.to_datetime(
        date_str + " " + time_str,
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce",
    ).dt.strftime("%Y-%m-%dT%H:%M:%S")

    # Optional: fail loudly if some recorded_at values could not be parsed
    bad_times = pheno_df.loc[pheno_df["recorded_at"].isna(), ["participant_id"]]
    if not bad_times.empty:
        raise ValueError(
            "Failed to parse recorded_at for participants: "
            + ", ".join(bad_times["participant_id"].tolist())
        )

    # Map gender and handedness values for all rows
    gender_mapping = {0: "M", 1: "F"}
    handedness_mapping = {1: "left", 0: "right"}

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


# BFI
def aggr_bfi_data(
        pheno_df: pd.DataFrame,
        phenotype_dir: str
    ) -> Optional[pd.DataFrame]:
    """
    Aggregate Stockholm BFI questionnaire data and write the output TSV.

    Compared with the Bonn implementation, the Stockholm pipeline matches
    BFI item columns using the exact source prefix ``"bfi_"``.

    Parameters
    ----------
    pheno_df : pandas.DataFrame
        Phenotype table containing ``"participant_id"`` and Stockholm BFI
        item columns matched by the prefix ``"bfi_"``.
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

    Stockholm-specific matching uses the exact prefix ``"bfi_"`` and
    renames it to ``"bfi{n_items}_"`` before integer conversion, missing
    column padding, and TSV writing.
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
    Aggregate Stockholm GAD questionnaire data and write the output TSV.

    Compared with the Bonn implementation, the Stockholm pipeline first
    removes columns ending with ``"_total"`` and then matches GAD item
    columns using the exact source prefix ``"gad_"``.

    Parameters
    ----------
    pheno_df : pandas.DataFrame
        Phenotype table containing ``"participant_id"`` and Stockholm GAD
        item columns matched by the prefix ``"gad_"`` after excluding
        columns ending with ``"_total"``.
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

    Stockholm-specific matching uses the exact prefix ``"gad_"`` and
    excludes columns ending in ``"_total"`` before renaming to
    ``"gad{n_items}_"``.
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
    Aggregate Stockholm IUS questionnaire data and write the output TSV.

    Compared with the Bonn implementation, the Stockholm pipeline first
    removes columns ending with ``"_total"`` and then matches IUS item
    columns using the exact source prefix ``"ius_"``.

    Parameters
    ----------
    pheno_df : pandas.DataFrame
        Phenotype table containing ``"participant_id"`` and Stockholm IUS
        item columns matched by the prefix ``"ius_"`` after excluding
        columns ending with ``"_total"``.
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

    Stockholm-specific matching uses the exact prefix ``"ius_"`` and
    excludes columns ending in ``"_total"`` before renaming to
    ``"ius{n_items}_"``.
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
    Aggregate Stockholm PHQ questionnaire data and write the output TSV.

    Compared with the Bonn implementation, the Stockholm pipeline first
    removes columns ending with ``"_total"`` and then matches PHQ item
    columns using the exact source prefix ``"phq_"``.

    Parameters
    ----------
    pheno_df : pandas.DataFrame
        Phenotype table containing ``"participant_id"`` and Stockholm PHQ
        item columns matched by the prefix ``"phq_"`` after excluding
        columns ending with ``"_total"``.
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

    Stockholm-specific matching uses the exact prefix ``"phq_"`` and
    excludes columns ending in ``"_total"`` before renaming to
    ``"phq{n_items}_"``.
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
    Aggregate Stockholm SOC questionnaire data and write the output TSV.

    Unlike the Bonn implementation, which may use different SOC source
    naming, the Stockholm pipeline first removes columns ending with
    ``"_total"`` and then matches SOC item columns using the exact source
    prefix ``"soc_"``.

    Parameters
    ----------
    pheno_df : pandas.DataFrame
        Phenotype table containing ``"participant_id"`` and Stockholm SOC
        item columns matched by the prefix ``"soc_"`` after excluding
        columns ending with ``"_total"``.
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

    Stockholm-specific matching uses the exact prefix ``"soc_"`` and
    excludes columns ending in ``"_total"`` before renaming to
    ``"soc{n_items}_"``.
    """

    # Remove columns that end with '_total'
    pheno_df = pheno_df[
        [col for col in pheno_df.columns if not col.endswith("_total")]
    ]

    # define settings
    current_quest = "soc"
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


# STAI
def aggr_stai_data(
        pheno_df: pd.DataFrame,
        phenotype_dir: str
    ) -> Optional[pd.DataFrame]:
    """
    Aggregate Stockholm STAI questionnaire data and write the output TSV.

    Compared with the Bonn implementation, the Stockholm pipeline matches
    STAI item columns using the exact source prefix ``"stai-t_"`` rather
    than the Bonn STAI source naming scheme.

    Parameters
    ----------
    pheno_df : pandas.DataFrame
        Phenotype table containing ``"participant_id"`` and Stockholm STAI
        item columns matched by the prefix ``"stai-t_"``.
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

    Stockholm-specific matching uses the exact prefix ``"stai-t_"`` and
    renames it to ``"stai{n_items}_"`` before integer conversion, missing
    column padding, and TSV writing.
    """

    # define settings
    current_quest = "stai"
    replace_key = f"{current_quest}-t_"
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

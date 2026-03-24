# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import pandas as pd
from calinet.imports import biopac
from calinet.core.metadata import df_meta, _get, _meta_for
from calinet.config import available_labs
from calinet.core.pheno import pad_missing_columns

from calinet.utils import (
    rename_col,
    common_write_tsv,
    append_acq_date_to_df,
    convert_questionnaire_columns_to_int
)

from typing import Tuple, Any, Optional

import logging
logger = logging.getLogger(__name__)

# lab-specific pheno info
lab_name = __name__.split(".")[-1]
lab_meta = available_labs.get(lab_name).get("MetaName")
lab_pheno = available_labs.get(lab_name).get("Phenotype")
language = lab_pheno.get("Language")


def read_raw_physio_file(
        raw_physio_acq: str
    ) -> Tuple[pd.DataFrame, float, Any]:
    """
    Read a Wuerzburg raw physiology text file.

    Unlike the Bonn implementation, which reads binary ``.acq`` files using
    ``biopac.read_acq_file``, the Wuerzburg pipeline reads plain-text
    physiology exports using ``biopac.read_biopac_txt_noheader`` and
    retrieves the sampling rate from ``metadata.csv``.

    Parameters
    ----------
    raw_physio_acq : str
        Path to the Wuerzburg physiology text file to read.

    Returns
    -------
    physio_df : pandas.DataFrame
        Physiology data extracted from the input text file.
    sr : float
        Sampling frequency in Hz retrieved from metadata and returned by
        the reader.
    chan_info : Any
        Selected channel metadata returned by
        ``biopac.read_biopac_txt_noheader``.

    Raises
    ------
    ValueError
        Raised if the sampling frequency cannot be extracted from
        ``metadata.csv``.
    Exception
        Raised if the input file cannot be read.

    Notes
    -----
    This function does not write files.

    Wuerzburg-specific processing reads the sampling rate from
    ``metadata.csv`` via ``_meta_for`` and ``_get`` before passing it to the
    text reader. :contentReference[oaicite:0]{index=0}
    """

    # Extract physiological data
    logger.info(f"Reading .acq file: {raw_physio_acq}")

    # read SamplingFrequency from metadata.csv
    meta = _meta_for(df_meta, "EDA")
    sr = _get(
        meta,
        "Sampling Rate",
        lab_meta
    )

    logger.info(f"Reading SamplingFrequency from {lab_name}'s metadata: {sr}")
    if sr is None:
        raise ValueError(f"Could not extract SamplingFrequency from metadata.csv for {lab_name}-lab")
    
    try:
        res = biopac.read_biopac_txt_noheader(
            raw_physio_acq,
            sampling_rate_hz=sr,
            column_names=available_labs.get(lab_name).get("ChannelRegex")
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
    Locate the Wuerzburg raw physiology text file for one participant.

    Unlike the Bonn implementation, which expects a file named
    ``"CALINET_Template{subject_id}.acq"``, the Wuerzburg pipeline expects a
    plain-text file named ``"{subject_id}.txt"`` directly inside
    ``subject_raw_data_path``.

    Parameters
    ----------
    subject_raw_data_path : str
        Path to the participant-specific raw data directory.
    subject_name : str
        Participant label whose last three characters are used as
        ``subject_id``.

    Returns
    -------
    physio_path : str
        Full path to the matching Wuerzburg physiology text file.

    Raises
    ------
    FileNotFoundError
        Raised if the expected ``"{subject_id}.txt"`` file does not exist.

    Notes
    -----
    This function does not read or write files.

    Wuerzburg-specific discovery uses the exact filename pattern
    ``"{subject_id}.txt"`` with no additional directory traversal. :contentReference[oaicite:1]{index=1}
    """

    # go up one level from the subject folder to get the raw_data_dir
    subject_id = subject_name[-3:]
    file_name = os.path.join(subject_raw_data_path, f"{subject_id}.txt")

    # build expected filename, e.g. "501" → "501.acq"
    if file_name and os.path.isfile(file_name):
        return file_name
    else:
        raise FileNotFoundError(f".acq file {file_name} not found in {subject_raw_data_path!r}")


def find_questionnaire_file(
        raw_data_dir: str
    ) -> Optional[str]:
    """
    Find the Wuerzburg questionnaire CSV file under the raw data tree.

    Unlike the Bonn implementation, which searches for a Bonn-specific
    questionnaire export, the Wuerzburg pipeline scans ``raw_data_dir``
    recursively and returns the first file whose name contains
    ``"questionnaire"`` and ends with ``".csv"``.

    Parameters
    ----------
    raw_data_dir : str
        Root directory to search recursively for the questionnaire file.

    Returns
    -------
    questionnaire_file : str or None
        Full path to the matching CSV file, or ``None`` if no file is found.

    Notes
    -----
    This function does not read questionnaire contents and does not write
    files.

    Matching is based on the filename substring ``"questionnaire"`` and the
    exact suffix ``".csv"``. :contentReference[oaicite:2]{index=2}
    """

    questionnaire_file = None
    for root, _, files in os.walk(raw_data_dir):
        for filename in files:
            if "questionnaire" in filename.lower() and filename.lower().endswith(
                ".csv"
            ):
                questionnaire_file = os.path.join(root, filename)
                break
        if questionnaire_file:
            return questionnaire_file


def parse_questionnaire_file(
        questionnaire_file: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse the Wuerzburg questionnaire CSV file and standardize participant fields.

    Unlike the Bonn implementation, which parses a Bonn questionnaire
    export, the Wuerzburg pipeline reads a semicolon-delimited CSV file and
    derives participant metadata from columns ``"id"``, ``"age"``,
    ``"gender"``, and ``"handedness"``.

    Parameters
    ----------
    questionnaire_file : str
        Path to the Wuerzburg questionnaire CSV file.

    Returns
    -------
    info_df : pandas.DataFrame
        Participant-level table containing ``"participant_id"``, ``"age"``,
        ``"sex"``, and ``"handedness"`` for use in ``participants.tsv``.
    pheno_df : pandas.DataFrame
        Full parsed phenotype table including questionnaire item columns
        and standardized participant metadata.

    Raises
    ------
    ValueError
        Raised if any required columns (``"id"``, ``"age"``, ``"gender"``,
        ``"handedness"``) are missing.

    Notes
    -----
    This function reads a CSV file but does not write output files.

    Wuerzburg-specific processing maps ``1`` and ``2`` in ``"gender"`` to
    ``"M"`` and ``"F"``, maps ``1`` and ``2`` in ``"handedness"`` to
    ``"left"`` and ``"right"``, and formats ``"participant_id"`` as
    ``"sub-XXX"`` from ``"id"``. :contentReference[oaicite:3]{index=3}
    """

    pheno_df = pd.read_csv(questionnaire_file, delimiter=";")

    # Strip whitespace from columns and convert to lowercase
    pheno_df.columns = pheno_df.columns.str.strip().str.lower()

    # rename bunch of columns
    pheno_df.rename(
        columns={
            "id": "participant_id",
            "gender": "sex",
            "room_temp": "room_temperature",
            "room_humidity": "humidity",
            "time_of_day": "recorded_at"
        },
        inplace=True
    )

    # Enforce floats for room temperature
    pheno_df["room_temperature"] = pheno_df["room_temperature"].str.replace(",", ".").astype(float)

    # Ensure the required columns exist
    required_columns = ["participant_id", "age", "sex", "handedness"]
    for col in required_columns:
        if col not in pheno_df.columns:
            raise ValueError(
                f"Missing required column '{col}' in questionnaire file: {questionnaire_file}"
            )

    # Map gender and handedness values for all rows
    gender_mapping = {1: "M", 2: "F"}
    handedness_mapping = {1: "left", 2: "right"}

    pheno_df["sex"] = pheno_df["sex"].map(gender_mapping).astype(object)
    pheno_df["handedness"] = pheno_df["handedness"].map(handedness_mapping).astype(object)
    
    # Sort by numeric ID
    pheno_df["participant_id"] = pheno_df["participant_id"].apply(lambda x: f"sub-{x:03d}")
    pheno_df = pheno_df.sort_values("participant_id").reset_index(drop=True)


    # Map each participant to the date of their physiology acquisition file
    raw_path = os.path.dirname(questionnaire_file)
    pheno_df = append_acq_date_to_df(
        pheno_df,
        raw_path,
    )

    acq_date = pd.to_datetime(pheno_df["acq_date"], errors="coerce")
    raw_time = pd.to_datetime(pheno_df["recorded_at"], errors="coerce")  # replace with your real time column

    # invalid if missing or sentinel 1900-01-01
    invalid_time = raw_time.isna() | raw_time.dt.normalize().eq(pd.Timestamp("1900-01-01"))

    bad_ids = pheno_df.loc[invalid_time, "participant_id"].tolist()
    if bad_ids:
        logger.warning(
            f"Missing/invalid derived time for {len(bad_ids)} participants; "
            f"falling back to acq_date only: {bad_ids}"
        )

    # default: acq_date at 00:00:00
    recorded_at = acq_date.copy()

    # where time is valid, add the time-of-day
    valid_time = ~invalid_time
    time_of_day = raw_time.loc[valid_time] - raw_time.loc[valid_time].dt.normalize()
    recorded_at.loc[valid_time] = acq_date.loc[valid_time] + time_of_day

    pheno_df["recorded_at"] = recorded_at.dt.strftime("%Y-%m-%dT%H:%M:%S")
    pheno_df.loc[acq_date.isna(), "recorded_at"] = None

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
    Aggregate Wuerzburg BFI questionnaire data and write the output TSV.

    Compared with the Bonn implementation, the Wuerzburg pipeline matches
    BFI item columns using the exact source prefix ``"bfi_"`` and restricts
    columns to the numeric pattern ``"bfi_{number}"``.

    Parameters
    ----------
    pheno_df : pandas.DataFrame
        Phenotype table containing ``"participant_id"`` and Wuerzburg BFI
        item columns matched by the prefix ``"bfi_"``.
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

    Wuerzburg-specific filtering keeps only columns matching the exact
    regex ``"bfi_\\d+"`` before renaming to ``"bfi{n_items}_"``. :contentReference[oaicite:4]{index=4}
    """

    # define settings
    current_quest = "bfi"
    replace_key = f"{current_quest}_"
    n_items = lab_pheno.get(current_quest)
    id_key = f"{current_quest}{n_items}_"

    # pre-define columns
    cols = ["participant_id"] + [c for c in pheno_df.columns if replace_key in c]

    # remove unwanted midi summary columns
    pheno_df = pheno_df.filter(regex=r'^(participant_id|bfi_\d+)$')

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
    Aggregate Wuerzburg GAD questionnaire data and write the output TSV.

    Compared with the Bonn implementation, the Wuerzburg pipeline removes
    columns ending with ``"_total"`` and matches GAD item columns using the
    exact source prefix ``"gad7_"``.

    Parameters
    ----------
    pheno_df : pandas.DataFrame
        Phenotype table containing ``"participant_id"`` and Wuerzburg GAD
        item columns matched by the prefix ``"gad7_"``.
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

    Wuerzburg-specific matching uses the exact prefix ``"gad7_"`` and
    renames it to ``"gad{n_items}_"`` after filtering. :contentReference[oaicite:5]{index=5}
    """

    # Remove columns that end with '_total'
    pheno_df = pheno_df[
        [col for col in pheno_df.columns if not col.endswith("_total")]
    ]

    # define settings
    current_quest = "gad"
    replace_key = f"{current_quest}7_"
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
    Aggregate Wuerzburg IUS questionnaire data and write the output TSV.

    Compared with the Bonn implementation, the Wuerzburg pipeline removes
    columns ending with ``"_total"`` and matches IUS item columns using the
    exact source prefix ``"ui_"``.

    Parameters
    ----------
    pheno_df : pandas.DataFrame
        Phenotype table containing ``"participant_id"`` and Wuerzburg IUS
        item columns matched by the prefix ``"ui_"``.
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

    Wuerzburg-specific matching uses the exact prefix ``"ui_"`` and renames
    it to ``"ius{n_items}_"`` after filtering. :contentReference[oaicite:6]{index=6}
    """

    # Remove columns that end with '_total'
    pheno_df = pheno_df[
        [col for col in pheno_df.columns if not col.endswith("_total")]
    ]

    # define settings
    current_quest = "ius"
    replace_key = f"ui_"
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
    Aggregate Wuerzburg PHQ questionnaire data and write the output TSV.

    Compared with the Bonn implementation, the Wuerzburg pipeline removes
    columns ending with ``"_total"`` and matches PHQ item columns using the
    exact source prefix ``"phq_"``.

    Parameters
    ----------
    pheno_df : pandas.DataFrame
        Phenotype table containing ``"participant_id"`` and Wuerzburg PHQ
        item columns matched by the prefix ``"phq_"``.
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

    Wuerzburg-specific matching uses the exact prefix ``"phq_"`` and
    renames it to ``"phq{n_items}_"`` after filtering. :contentReference[oaicite:7]{index=7}
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
    Aggregate Wuerzburg SOC questionnaire data and write the output TSV.

    Unlike the Bonn implementation, which may use different SOC source
    naming, the Wuerzburg pipeline removes summary columns and matches SOC
    item columns using the exact source prefix ``"midi_"``.

    Parameters
    ----------
    pheno_df : pandas.DataFrame
        Phenotype table containing ``"participant_id"`` and Wuerzburg SOC
        item columns matched by the prefix ``"midi_"``.
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

    Wuerzburg-specific filtering keeps only columns matching
    ``"midi_\\d+"`` before renaming to ``"soc{n_items}_"``. :contentReference[oaicite:8]{index=8}
    """

    # Remove columns that end with '_total'
    pheno_df = pheno_df[
        [col for col in pheno_df.columns if not col.endswith("_total")]
    ]

    # remove unwanted midi summary columns
    pheno_df = pheno_df.filter(regex=r'^(participant_id|midi_\d+)$')

    # define settings
    current_quest = "soc"
    replace_key = "midi_"
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
    Aggregate Wuerzburg STAI questionnaire data and write the output TSV.

    Compared with the Bonn implementation, the Wuerzburg pipeline matches
    STAI item columns using the exact source prefix ``"stait_"`` rather
    than the Bonn STAI naming scheme.

    Parameters
    ----------
    pheno_df : pandas.DataFrame
        Phenotype table containing ``"participant_id"`` and Wuerzburg STAI
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

    Wuerzburg-specific matching uses the exact prefix ``"stait_"`` and
    renames it to ``"stai{n_items}_"`` after filtering. :contentReference[oaicite:9]{index=9}
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

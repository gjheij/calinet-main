# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import glob
import numpy as np
import pandas as pd
from calinet.imports import biopac
from calinet.config import available_labs
from calinet.utils import find_sub_dirs, extract_subject_id
from calinet.core.pheno import (
    rename_col,
    common_write_tsv,
    pad_missing_columns,
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
    Read a New York raw physiology ``.acq`` file.

    This implementation is equivalent to the Bonn pipeline in that it reads
    a single ``.acq`` file with ``biopac.read_acq_file`` and returns the
    extracted signal table, sampling rate, and selected channel metadata.

    Parameters
    ----------
    raw_physio_acq : str
        Path to the New York ``.acq`` file to read.

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

    Channel selection is resolved from the New York lab configuration via
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
    Locate the New York raw physiology ``.acq`` file for one participant.

    Unlike the Bonn implementation, which expects a specific
    ``"CALINET_Template{subject_id}.acq"`` file name directly under the
    participant directory, the New York pipeline searches recursively under
    ``subject_raw_data_path`` for any file ending in ``".acq"`` and returns
    the first match.

    Parameters
    ----------
    subject_raw_data_path : str
        Path to the participant-specific raw data directory searched
        recursively for physiology files.
    subject_name : str
        Participant label used only in error messages. Unlike the Bonn
        reference implementation, this value is not used to derive the
        expected file name.

    Returns
    -------
    physio_path : str
        Full path to the first matching New York ``.acq`` file.

    Raises
    ------
    FileNotFoundError
        Raised if no ``.acq`` file is found under ``subject_raw_data_path``.

    Notes
    -----
    This function does not read or write files.

    New York-specific discovery uses the exact filename suffix ``".acq"``
    and recursive matching with no subject-specific filename prefix.

    If the matched filename contains ``"scored"``, the function logs that
    only a scored file is present. Otherwise it logs that an unscored file
    is being used.
    """

    matches = glob.glob(os.path.join(subject_raw_data_path, "**", "*.acq"), recursive=True)

    if len(matches)==0:
        raise FileNotFoundError(f".acq file not found for '{subject_name}' in {subject_raw_data_path!r}")
    else:
        file_name = matches[0]

    if "scored" in os.path.basename(file_name).lower():
        logger.info("Only file with 'scored' present")
    else:
        logger.info("Using 'unscored' file")

    return file_name

    
def find_questionnaire_file(
        raw_data_dir: str
    ) -> str:
    """
    Return the New York questionnaire root directory.

    Unlike the Bonn implementation, which searches for a single
    questionnaire source file, the New York pipeline stores questionnaire
    data across multiple subject-specific Excel files. This function does
    not search for one file and instead returns ``raw_data_dir`` unchanged
    so downstream code can aggregate per-subject questionnaire files.

    Parameters
    ----------
    raw_data_dir : str
        Root directory containing New York subject-level questionnaire data.

    Returns
    -------
    questionnaire_file : str
        The input ``raw_data_dir`` passed through unchanged for later
        subject-level parsing.

    Notes
    -----
    This function does not read questionnaire contents and does not write
    files.

    New York-specific questionnaire discovery is directory-based rather than
    file-based because downstream helpers search subject subdirectories for
    files matching ``"*MIDI_BFI*"``, ``"*GAD7*"``, and ``"*IU_STAI*"``.
    """
    
    # NY has subject-specific xlsx files, so we first find the subject dirs, 
    # then group phenotype data per subject
    return raw_data_dir


def aggregate_pheno(
        raw_path: str
    ) -> pd.DataFrame:
    """
    Aggregate New York subject-level questionnaire data from multiple Excel files.

    Unlike the Bonn implementation, which parses one questionnaire export,
    the New York pipeline reads up to three separate subject-level Excel
    files and concatenates their extracted questionnaire columns
    column-wise.

    Parameters
    ----------
    raw_path : str
        Path to one participant directory containing questionnaire files.

    Returns
    -------
    df : pandas.DataFrame
        Concatenated questionnaire data for one participant assembled from
        the outputs of ``read_midi_bfi_xlsx``, ``read_gad7_phq_xlsx``, and
        ``read_ius_stai_xlsx``.

    Notes
    -----
    This function reads questionnaire Excel files but does not write output
    files.

    New York-specific aggregation expects questionnaire filenames matching
    the exact substrings ``"*MIDI_BFI*"``, ``"*GAD7*"``, and
    ``"*IU_STAI*"``.
    """

    ddict = {
        "MIDI/BFI": read_midi_bfi_xlsx,
        "GAD/PHQ": read_gad7_phq_xlsx,
        "IUS/STAI": read_ius_stai_xlsx
    }

    dfs = []
    for _, val in ddict.items():
        df = val(raw_path)
        dfs.append(df)
        
    if len(dfs)>0:
        dfs = pd.concat(dfs, axis=1)

    return dfs


def read_midi_bfi_xlsx(
        raw_path: str
    ) -> pd.DataFrame:
    """
    Read New York SOC and BFI questionnaire data from one Excel file.

    Unlike the Bonn implementation, which aggregates questionnaire items
    from a site-level export, the New York pipeline reads one
    subject-specific Excel workbook matching ``"*MIDI_BFI*"`` and extracts
    SOC items from the ``"MIDI"`` sheet and BFI items from the ``"BFI"``
    sheet.

    Parameters
    ----------
    raw_path : str
        Path to one participant directory searched for a file whose name
        matches the exact substring pattern ``"*MIDI_BFI*"``.

    Returns
    -------
    df : pandas.DataFrame
        One-row DataFrame containing SOC and BFI questionnaire items. SOC
        columns use the exact standardized prefix ``"soc{n_soc}_"`` and BFI
        columns use the exact standardized prefix ``"bfi{n_bfi}_"``.

    Notes
    -----
    This function reads an Excel workbook but does not write output files.

    If no matching file is found, the function returns a one-row DataFrame
    filled with ``NaN`` values for all expected SOC and BFI columns.

    Column prefixes used to construct the output are exactly
    ``"soc{n_soc}_"`` for SOC items and ``"bfi{n_bfi}_"`` for BFI items.
    """

    # expected columns
    n_soc = lab_pheno.get("soc")
    n_bfi = lab_pheno.get("bfi")
    soc_cols = [f"soc{n_soc}_{i}" for i in range(1, n_soc+1)]
    bfi_cols = [f"bfi{n_bfi}_{i}" for i in range(1, n_bfi+1)]
    all_cols = soc_cols + bfi_cols

    pattern = os.path.join(raw_path, "*MIDI_BFI*")
    matches = glob.glob(pattern)

    # return empty dataframe if file missing
    if len(matches) == 0:
        logger.warning(f"No SOC/BFI data for {os.path.basename(raw_path)}; returning NaNs")
        return pd.DataFrame([[np.nan] * len(all_cols)], columns=all_cols)

    # read file
    quest_file = matches[0]
    soc = pd.read_excel(quest_file, sheet_name="MIDI").iloc[:-2]
    bfi = pd.read_excel(quest_file, sheet_name="BFI")
    bfi.sort_values(["Question"], inplace=True)

    soc_scores = soc["Response"].values[np.newaxis, :]
    bfi_scores = bfi["Response"].values[np.newaxis, :]

    soc_df = pd.DataFrame(
        soc_scores,
        columns=soc_cols
    )

    bfi_df = pd.DataFrame(
        bfi_scores,
        columns=bfi_cols
    )

    df = pd.concat([soc_df, bfi_df], axis=1)

    return df


def read_gad7_phq_xlsx(
        raw_path: str
    ) -> pd.DataFrame:
    """
    Read New York GAD and PHQ questionnaire data from one Excel file.

    Unlike the Bonn implementation, which aggregates questionnaire items
    from a site-level export, the New York pipeline reads one
    subject-specific Excel workbook matching ``"*GAD7*"`` and extracts GAD
    items from the ``"GAD-7 Data"`` sheet and PHQ items from the
    ``"PHQ-9 Data"`` sheet.

    Parameters
    ----------
    raw_path : str
        Path to one participant directory searched for a file whose name
        matches the exact substring pattern ``"*GAD7*"``.

    Returns
    -------
    df : pandas.DataFrame
        One-row DataFrame containing GAD and PHQ questionnaire items. GAD
        columns use the exact standardized prefix ``"gad{n_gad}_"`` and PHQ
        columns use the exact standardized prefix ``"phq{n_phq}_"``.

    Notes
    -----
    This function reads an Excel workbook but does not write output files.

    If no matching file is found, the function returns a one-row DataFrame
    filled with ``NaN`` values for all expected GAD and PHQ columns.

    Column prefixes used to construct the output are exactly
    ``"gad{n_gad}_"`` for GAD items and ``"phq{n_phq}_"`` for PHQ items.
    """

    # expected columns
    n_soc = lab_pheno.get("gad")
    n_bfi = lab_pheno.get("phq")
    gad_cols = [f"gad{n_soc}_{i}" for i in range(1, n_soc+1)]
    phq_cols = [f"phq{n_bfi}_{i}" for i in range(1, n_bfi+1)]
    all_cols = gad_cols + phq_cols

    # find file
    pattern = os.path.join(raw_path, "*GAD7*")
    matches = glob.glob(pattern)

    # return empty dataframe if file missing
    if len(matches) == 0:
        logger.warning(f"No GAD/PHQ data for {os.path.basename(raw_path)}; returning NaNs")
        return pd.DataFrame([[np.nan] * len(all_cols)], columns=all_cols)

    # read file
    quest_file = matches[0]
    gad = pd.read_excel(quest_file, sheet_name="GAD-7 Data").iloc[:-1]
    phq = pd.read_excel(quest_file, sheet_name="PHQ-9 Data").iloc[:-2]

    # get scores
    gad_scores = gad["Response"].values[np.newaxis, :]
    phq_scores = phq["Response"].values[np.newaxis, :]

    # build dataframe
    gad_df = pd.DataFrame(
        gad_scores,
        columns=gad_cols
    )

    phq_df = pd.DataFrame(
        phq_scores,
        columns=phq_cols
    )

    df = pd.concat([gad_df, phq_df], axis=1)

    return df


def read_ius_stai_xlsx(
        raw_path: str
    ) -> pd.DataFrame:
    """
    Read New York IUS and STAI questionnaire data from one Excel file.

    Unlike the Bonn implementation, which aggregates questionnaire items
    from a site-level export, the New York pipeline reads one
    subject-specific Excel workbook matching ``"*IU_STAI*"`` and extracts
    IUS items from the ``"IUS"`` sheet and STAI items from the ``"STAI"``
    sheet.

    Parameters
    ----------
    raw_path : str
        Path to one participant directory searched for a file whose name
        matches the exact substring pattern ``"*IU_STAI*"``.

    Returns
    -------
    df : pandas.DataFrame
        One-row DataFrame containing IUS and STAI questionnaire items. IUS
        columns use the exact standardized prefix ``"ius{n_ius}_"`` and
        STAI columns use the exact standardized prefix ``"stai{n_stai}_"``.

    Notes
    -----
    This function reads an Excel workbook but does not write output files.

    If no matching file is found, the function returns a one-row DataFrame
    filled with ``NaN`` values for all expected IUS and STAI columns.

    Column prefixes used to construct the output are exactly
    ``"ius{n_ius}_"`` for IUS items and ``"stai{n_stai}_"`` for STAI items.
    """

    # expected columns
    n_ius = lab_pheno.get("ius")
    n_stai = lab_pheno.get("stai")
    ius_cols = [f"ius{n_ius}_{i}" for i in range(1, n_ius+1)]
    stai_cols = [f"stai{n_stai}_{i}" for i in range(1, n_stai+1)]
    all_cols = ius_cols + stai_cols

    # find file
    pattern = os.path.join(raw_path, "*IU_STAI*")
    matches = glob.glob(pattern)

    # return empty dataframe if file missing
    if len(matches) == 0:
        logger.warning(f"No IUS/STAI data for {os.path.basename(raw_path)}; returning NaNs")
        return pd.DataFrame([[np.nan] * len(all_cols)], columns=all_cols)

    # read file
    quest_file = matches[0]
    ius = pd.read_excel(quest_file, sheet_name="IUS").iloc[:-3]
    stai = pd.read_excel(quest_file, sheet_name="STAI").iloc[:-1]

    # extract scores
    ius_scores = ius["Response"].values[np.newaxis, :]
    stai_scores = stai["Response"].values[np.newaxis, :]

    # build dataframe
    ius_df = pd.DataFrame(
        ius_scores,
        columns=ius_cols
    )

    stai_df = pd.DataFrame(
        stai_scores,
        columns=stai_cols
    )

    df = pd.concat([ius_df, stai_df], axis=1)
    return df


def read_master_data(
        raw_data_dir: str
    ) -> pd.DataFrame:
    """
    Read New York participant-level master data from the master workbook.

    Unlike the Bonn implementation, which parses participant information
    from the main questionnaire export, the New York pipeline reads a
    separate Excel workbook whose filename contains ``"master_data"`` and
    derives participant metadata from that file.

    Parameters
    ----------
    raw_data_dir : str
        Root directory searched recursively for the master workbook whose
        filename contains ``"master_data"`` and ends with ``".xlsx"``.

    Returns
    -------
    pheno_df : pandas.DataFrame
        Participant-level metadata table containing standardized fields such
        as ``"participant_id"``, ``"age"``, ``"sex"``, ``"handedness"``,
        ``"room_temperature"``, and ``"recorded_at"``.

    Raises
    ------
    FileNotFoundError
        Raised if no master workbook matching the New York search rule is
        found.
    ValueError
        Raised if any required columns such as ``"participant_id"``,
        ``"age"``, ``"sex"``, or ``"handedness"`` are missing after
        normalization.
    Exception
        Raised if the master workbook cannot be processed.

    Notes
    -----
    This function reads an Excel workbook but does not write output files.

    New York-specific processing renames ``"subject number"`` to
    ``"participant_id"``, renames ``"temperature"`` to
    ``"temperature_fahrenheit"``, computes ``"room_temperature"`` in
    Celsius, maps ``"M"`` and ``"F"`` in ``"sex"`` to ``"male"`` and
    ``"female"``, maps ``"L"`` and ``"R"`` in ``"handedness"`` to
    ``"left"`` and ``"right"``, combines ``"date"`` and ``"time"`` into
    ``"recorded_at"``, and formats ``"participant_id"`` as ``"sub-XXX"``.
    """

    # Look for the questionnaire file in the base folder
    questionnaire_file = None
    for root, _, files in os.walk(raw_data_dir):
        for filename in files:
            if "master_data" in filename.lower() and filename.lower().endswith(".xlsx"):
                questionnaire_file = os.path.join(root, filename)
                break
        if questionnaire_file:
            break

    if not questionnaire_file:
        raise FileNotFoundError(f"No questionnaire file found in {raw_data_dir}")

    logger.info(f"Reading master data: {questionnaire_file}")

    try:
        pheno_df = pd.read_excel(questionnaire_file)
        # Drop columns, rows which are unnamed and completely blank
        pheno_df = pheno_df.head(33)
        pheno_df.dropna(axis=1, how="all", inplace=True)
        pheno_df.dropna(axis=0, how="all", inplace=True)

        # Strip whitespace from columns and convert to lowercase
        pheno_df.columns = pheno_df.columns.str.strip().str.lower()

        # Rename 'ppn' column to 'id' if it exists
        if "subject number" in pheno_df.columns:
            pheno_df.rename(columns={
                "subject number": "participant_id",
                "temperature": "temperature_fahrenheit"
            }, inplace=True)

        pheno_df["room_temperature"] = ((pheno_df["temperature_fahrenheit"] - 32) * 5/9).round(1)
        
        # Ensure the required columns exist
        required_columns = ["participant_id", "age", "sex", "handedness"]
        for col in required_columns:
            if col not in pheno_df.columns:
                raise ValueError(
                    f"Missing required column '{col}' in questionnaire file: {questionnaire_file}"
                )

        # Map sex and handedness values for all rows
        sex_mapping = {"M": "male", "F": "female"}
        handedness_mapping = {"L": "left", "R": "right"}

        pheno_df["sex"] = pheno_df["sex"].map(sex_mapping).astype(object)
        pheno_df["handedness"] = (
            pheno_df["handedness"].map(handedness_mapping).astype(object)
        )
        
        # add recorded_at
        pheno_df["recorded_at"] = pd.to_datetime(
            pheno_df["date"].astype(str) + " " + pheno_df["time"].astype(str),
            errors="coerce"
        )

        # Extract only the relevant columns for participants.tsv
        pheno_df["participant_id"] = pheno_df["participant_id"].apply(lambda x: f"sub-{int(x):03d}")

        return pheno_df

    except Exception as e:
        logging.error(f"Error processing questionnaire file: {e}")
        raise Exception(f"Error processing questionnaire file: {e}") from e


def parse_subject_questionnaire(
        raw_path: str
    ) -> pd.DataFrame:
    """
    Parse New York subject-specific questionnaire files for all participants.

    Unlike the Bonn implementation, which parses one site-level
    questionnaire export, the New York pipeline iterates over participant
    subdirectories and aggregates questionnaire data separately for each
    subject.

    Parameters
    ----------
    raw_path : str
        Root directory containing participant subdirectories with
        subject-specific questionnaire Excel files.

    Returns
    -------
    pheno_df : pandas.DataFrame
        Combined questionnaire table for all participants with one row per
        subject and a standardized ``"participant_id"`` column.

    Notes
    -----
    This function reads multiple questionnaire Excel files but does not
    write output files.

    New York-specific parsing finds participant directories with
    ``find_sub_dirs(raw_path)``, aggregates each subject with
    ``aggregate_pheno``, derives participant labels with
    ``extract_subject_id``, and formats them as ``"sub-XXX"``.
    """

    # find subject dirs
    subject_dirs = find_sub_dirs(raw_path)

    # read subject-specific pheno data
    logger.info(f"Found {len(subject_dirs)} subjects in {raw_path}")
    pheno_df = []
    for i in subject_dirs:
        df = aggregate_pheno(i)
        df["participant_id"] = extract_subject_id(os.path.basename(i))
        pheno_df.append(df)
    
    # concatenate
    if len(pheno_df)>0:
        pheno_df = pd.concat(pheno_df)

    pheno_df["participant_id"] = pheno_df["participant_id"].apply(lambda x: f"sub-{int(x):03d}")
    
    return pheno_df


def parse_questionnaire_file(
        raw_path: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse New York questionnaire data and standardize participant fields.

    Unlike the Bonn implementation, which parses one questionnaire export,
    the New York pipeline merges participant metadata from a separate master
    workbook with subject-specific questionnaire data aggregated from
    multiple Excel files per participant.

    Parameters
    ----------
    raw_path : str
        Root directory containing New York subject-specific questionnaire
        files. Participant metadata are read from the parent directory of
        this path.

    Returns
    -------
    info_df : pandas.DataFrame
        Participant-level table containing ``"participant_id"``, ``"age"``,
        ``"sex"``, and ``"handedness"`` for use in ``participants.tsv``.
    pheno_df : pandas.DataFrame
        Full parsed phenotype table created by merging master participant
        metadata with aggregated subject-specific questionnaire data.

    Notes
    -----
    This function reads multiple Excel files but does not write output
    files.

    New York-specific parsing uses ``read_master_data`` for participant
    metadata, ``parse_subject_questionnaire`` for questionnaire scores, and
    merges both tables on ``"participant_id"`` with ``how="right"`` so the
    subject-specific questionnaire data define the reference set of rows.
    """

    # read info from master file
    logger.info(f"Fetching participant info for New York")
    ref_directory = os.path.dirname(raw_path)
    master_pheno = read_master_data(ref_directory)

    # read pheno data
    logger.info(f"Reading subject-specific questionnaire data")
    tmp_pheno = parse_subject_questionnaire(raw_path)

    # merge:
    pheno_df = pd.merge(
        master_pheno,
        tmp_pheno,
        on="participant_id",
        how="right",   # take pheno as ref
    )

    # extract info for participants.tsv
    info_df = master_pheno[
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
    Aggregate New York BFI questionnaire data and write the output TSV.

    Unlike lab implementations that match source columns by loose prefixes,
    the New York pipeline already stores BFI questionnaire items using the
    exact standardized prefix ``"bfi{n_items}_"``. Aggregation selects all
    columns whose names contain that exact prefix.

    Parameters
    ----------
    pheno_df : pandas.DataFrame
        Phenotype table containing ``"participant_id"`` and New York BFI
        item columns matched by the exact prefix ``"bfi{n_items}_"``.
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

    New York-specific matching uses the exact standardized source prefix
    ``"bfi{n_items}_"`` with no additional renaming of questionnaire scale
    prefixes.
    """

    # define settings
    current_quest = "bfi"
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


# GAD
def aggr_gad_data(
        pheno_df: pd.DataFrame,
        phenotype_dir: str
    ) -> Optional[pd.DataFrame]:
    """
    Aggregate New York GAD questionnaire data and write the output TSV.

    Unlike lab implementations that match source columns by alternative raw
    prefixes, the New York pipeline already stores GAD questionnaire items
    using the exact standardized prefix ``"gad{n_items}_"``. Aggregation
    first excludes columns ending with ``"_total"`` and then selects all
    columns whose names contain that exact prefix.

    Parameters
    ----------
    pheno_df : pandas.DataFrame
        Phenotype table containing ``"participant_id"`` and New York GAD
        item columns matched by the exact prefix ``"gad{n_items}_"`` after
        excluding columns ending with ``"_total"``.
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

    New York-specific matching uses the exact standardized source prefix
    ``"gad{n_items}_"`` and excludes columns ending in ``"_total"`` before
    aggregation.
    """

    # Remove columns that end with '_total'
    pheno_df = pheno_df[
        [col for col in pheno_df.columns if not col.endswith("_total")]
    ]

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
    Aggregate New York IUS questionnaire data and write the output TSV.

    Unlike lab implementations that match source columns by alternative raw
    prefixes, the New York pipeline already stores IUS questionnaire items
    using the exact standardized prefix ``"ius{n_items}_"``. Aggregation
    first excludes columns ending with ``"_total"`` and then selects all
    columns whose names contain that exact prefix.

    Parameters
    ----------
    pheno_df : pandas.DataFrame
        Phenotype table containing ``"participant_id"`` and New York IUS
        item columns matched by the exact prefix ``"ius{n_items}_"`` after
        excluding columns ending with ``"_total"``.
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

    New York-specific matching uses the exact standardized source prefix
    ``"ius{n_items}_"`` and excludes columns ending in ``"_total"`` before
    aggregation.
    """

    # Remove columns that end with '_total'
    pheno_df = pheno_df[
        [col for col in pheno_df.columns if not col.endswith("_total")]
    ]

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
    Aggregate New York PHQ questionnaire data and write the output TSV.

    Unlike lab implementations that match source columns by alternative raw
    prefixes, the New York pipeline already stores PHQ questionnaire items
    using the exact standardized prefix ``"phq{n_items}_"``. Aggregation
    first excludes columns ending with ``"_total"`` and then selects all
    columns whose names contain that exact prefix.

    Parameters
    ----------
    pheno_df : pandas.DataFrame
        Phenotype table containing ``"participant_id"`` and New York PHQ
        item columns matched by the exact prefix ``"phq{n_items}_"`` after
        excluding columns ending with ``"_total"``.
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

    New York-specific matching uses the exact standardized source prefix
    ``"phq{n_items}_"`` and excludes columns ending in ``"_total"`` before
    aggregation.
    """

    # Remove columns that end with '_total'
    pheno_df = pheno_df[
        [col for col in pheno_df.columns if not col.endswith("_total")]
    ]

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
    Aggregate New York SOC questionnaire data and write the output TSV.

    Unlike lab implementations that match source columns by alternative raw
    prefixes, the New York pipeline already stores SOC questionnaire items
    using the exact standardized prefix ``"soc{n_items}_"``. Aggregation
    first excludes columns ending with ``"_total"`` and then selects all
    columns whose names contain that exact prefix.

    Parameters
    ----------
    pheno_df : pandas.DataFrame
        Phenotype table containing ``"participant_id"`` and New York SOC
        item columns matched by the exact prefix ``"soc{n_items}_"`` after
        excluding columns ending with ``"_total"``.
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

    New York-specific matching uses the exact standardized source prefix
    ``"soc{n_items}_"`` and excludes columns ending in ``"_total"`` before
    aggregation.
    """

    # Remove columns that end with '_total'
    pheno_df = pheno_df[
        [col for col in pheno_df.columns if not col.endswith("_total")]
    ]

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
    Aggregate New York STAI questionnaire data and write the output TSV.

    Unlike lab implementations that match source columns by alternative raw
    prefixes, the New York pipeline already stores STAI questionnaire items
    using the exact standardized prefix ``"stai{n_items}_"``. Aggregation
    selects all columns whose names contain that exact prefix.

    Parameters
    ----------
    pheno_df : pandas.DataFrame
        Phenotype table containing ``"participant_id"`` and New York STAI
        item columns matched by the exact prefix ``"stai{n_items}_"``.
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

    New York-specific matching uses the exact standardized source prefix
    ``"stai{n_items}_"`` with no ``"_total"`` exclusion step.
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

# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import glob
import pandas as pd
from calinet.imports import biopac
from calinet.config import available_labs
from calinet.core.pheno import pad_missing_columns

from calinet.utils import (
    rename_col,
    common_write_tsv,
    convert_questionnaire_columns_to_int,
)

from typing import List, Any, Optional, Tuple

import logging
logger = logging.getLogger(__name__)

opd = os.path.dirname

# lab-specific pheno info
lab_name = __name__.split(".")[-1]
lab_pheno = available_labs.get(lab_name).get("Phenotype")
language = lab_pheno.get("Language")


def read_raw_physio_file(
        raw_physio_acq: List[str]
    ) -> Tuple[pd.DataFrame, float, Any]:
    """
    Read and combine Austin raw physiology ``.acq`` files.

    Unlike the Bonn implementation, which reads a single ``.acq`` file,
    Austin physiology data may be split across multiple files under the
    processed ``scr`` directory. All input files are read in order and
    concatenated row-wise into a single DataFrame.

    Parameters
    ----------
    raw_physio_acq : list of str
        Paths to Austin ``.acq`` files to read and combine.

    Returns
    -------
    physio_df : pandas.DataFrame
        Concatenated physiology data from all successfully read ``.acq``
        files.
    sr : float
        Sampling frequency in Hz taken from the last successfully read
        file.
    chan_info : Any
        Selected channel metadata returned by ``biopac.read_acq_file`` for
        the last successfully read file.

    Raises
    ------
    Exception
        Raised if any input ``.acq`` file cannot be read.

    Notes
    -----
    This function reads multiple files and concatenates them with
    ``pd.concat``.

    The sampling rate and channel metadata are taken from the final file
    processed in ``raw_physio_acq``.
    """

    lab_name = __name__.split(".")[-1]

    # Extract physiological data
    logger.info(f"Reading .acq files")
    physio_df = []
    for f in raw_physio_acq:

        # read acqknowledge file
        try:
            logger.info(f"Loading {f}")
            res = biopac.read_acq_file(
                f,
                channels=available_labs.get(lab_name).get("ChannelRegex")
            )
        except Exception as e:
            raise Exception(f"Error while reading '{f}': {e}") from e

        physio_df.append(res.df)

    # extract dataframe and sampling rate
    logger.info("Combining data from acq-files")
    physio_df = pd.concat(physio_df)
    sr = res.sampling_rate_hz
    chan_info = res.selected_channel_info

    return (physio_df, sr, chan_info)


def find_physio_acq_file(
        subject_raw_data_path: str,
        subject_name: str
    ) -> List[str]:
    """
    Find Austin raw physiology ``.acq`` files for a subject.

    Unlike the Bonn implementation, which expects a single file with a
    fixed template name, Austin searches for one or more matching
    ``.acq`` files under ``scr/processed/<subject>`` relative to
    ``subject_raw_data_path``.

    Parameters
    ----------
    subject_raw_data_path : str
        Subject-specific raw data directory.
    subject_name : str
        Subject identifier. This argument is accepted by the function
        signature but is not used by the current implementation.

    Returns
    -------
    matches : list of str
        List of matching ``.acq`` file paths for the subject.

    Raises
    ------
    Exception
        Raised if ``subject_raw_data_path`` does not exist or is not a
        directory.
    FileNotFoundError
        Raised if no matching ``.acq`` files are found in the Austin
        processed physiology directory.

    Notes
    -----
    The search pattern is ``"<subject_basename>*.acq"`` inside
    ``<parent>/scr/processed/<subject_basename>``.

    The function may return multiple files and logs a warning when only
    one file is found, because Austin commonly expects two files.
    """

    if not os.path.isdir(subject_raw_data_path):
        raise Exception(f"No physio folder at {subject_raw_data_path!r}")

    # build expected filename
    orig_sub = os.path.basename(subject_raw_data_path)
    scr_dir = os.path.join(
        opd(subject_raw_data_path),
        "scr",
        "processed",
        orig_sub
    )

    pattern = f"{orig_sub}*.acq"
    matches = glob.glob(os.path.join(scr_dir, pattern))
    if not matches:
        raise FileNotFoundError(f"No ACQ file found for {orig_sub} in {scr_dir}")
    else:
        if len(matches)<2:
            logger.warning(f"Found only 1 file, but expected 2!")
        else:
            logger.info(f"Found {len(matches)} files:")

        for m in matches:
            logger.info(f" {m}")

    return matches

    
def find_questionnaire_file(
        raw_data_dir: str
    ) -> Optional[str]:
    """
    Find the Austin questionnaire export file in the raw data directory.

    Recursively searches ``raw_data_dir`` for a file whose name contains
    the Austin-specific export marker ``"CALINET2_DATA_2026-03-03_1107"``.

    Parameters
    ----------
    raw_data_dir : str
        Root directory to search for the questionnaire export.

    Returns
    -------
    questionnaire_file : str or None
        Full path to the first matching questionnaire file, or ``None`` if
        no matching file is found.

    Notes
    -----
    This function uses an Austin-specific filename convention that differs
    from the Bonn export name.
    """
    
    questionnaire_file = None
    for root, _, files in os.walk(raw_data_dir):
        for filename in files:
            if "CALINET2_DATA_2026-03-03_1107" in filename:
                questionnaire_file = os.path.join(root, filename)
                break
    return questionnaire_file


def parse_questionnaire_file(
        questionnaire_file: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse the Austin questionnaire export into participant and phenotype tables.

    Unlike the Bonn implementation, Austin questionnaire exports may
    contain repeated ``record_id`` rows and lab-specific REDCap-style
    column names. Rows are collapsed to one row per participant,
    demographic fields are renamed to the common schema, and
    ``"handedness"`` is added as missing because it is not available in
    the Austin export.

    Parameters
    ----------
    questionnaire_file : str
        Path to the Austin questionnaire CSV export.

    Returns
    -------
    info_df : pandas.DataFrame
        Participant-level table containing ``"participant_id"``,
        ``"age"``, ``"sex"``, and ``"handedness"``.
    pheno_df : pandas.DataFrame
        Full normalized phenotype table after column cleanup, row
        collapsing, renaming, and participant ID formatting.

    Raises
    ------
    ValueError
        Raised if any required columns are missing after Austin-specific
        renaming.

    Notes
    -----
    Column names are normalized to lowercase and stripped of surrounding
    whitespace.

    String values equal to ``"None"`` are replaced with missing values.

    Duplicate participant rows are collapsed with
    ``groupby("record_id", as_index=False).first()``.

    Austin-specific demographic columns such as ``"age_213e53"`` and
    ``"gender_5b52fd"`` are renamed to the shared schema.

    Participant IDs are converted to ``"sub-XXX"`` format using the
    trailing numeric portion of ``"participant_id"``.

    ``"handedness"`` is not collected in this export and is therefore set
    to missing for all rows.
    """

    # read excel file
    pheno_df = pd.read_csv(questionnaire_file, delimiter=",")

    # Normalize column names
    pheno_df.columns = pheno_df.columns.str.strip().str.lower()

    # normalize None's
    pheno_df = pheno_df.replace("None", pd.NA)

    # melt rows together
    pheno_df = (
        pheno_df
        .groupby("record_id", as_index=False)
        .first()
    )

    pheno_df.rename(columns={
        "record_id": "participant_id",
        "demographics_form_timestamp": "recorded_at",
        "age_213e53": "age",
        "gender_5b52fd": "sex",
        },
        inplace=True
    )

    # Ensure the required columns exist
    required_columns = ["participant_id", "age", "sex"]
    for col in required_columns:
        if col not in pheno_df.columns:
            raise ValueError(
                f"Missing required column '{col}' in questionnaire file: {questionnaire_file}"
            )

    
    # Convert participant_id to the format sub-01, sub-02, etc.
    pheno_df["participant_id"] = (
        pheno_df["participant_id"]
        .astype(str)
        .str.extract(r"(\d+)$")[0]
        .astype(int)
        .apply(lambda x: f"sub-{x:03d}")
    )

    # Sort by numeric ID
    pheno_df = pheno_df.sort_values("participant_id").reset_index(drop=True)

    # Extract only the relevant columns for participants.tsv
    info_df = pheno_df[
        [
            "participant_id",
            "age",
            "sex",
        ]
    ].copy()

    # no handedness available
    info_df["handedness"] = None

    return info_df, pheno_df


# BFI
def aggr_bfi_data(
        pheno_df: pd.DataFrame,
        phenotype_dir: str
    ) -> Optional[pd.DataFrame]:
    """
    Aggregate Austin BFI questionnaire data and write the output TSV.

    Compared with the Bonn implementation, the Austin pipeline matches BFI
    item columns using the exact source prefix ``"bfi_"``. All columns
    whose names contain ``"bfi_"`` are selected and then renamed to the
    standardized output prefix ``"bfi{n_items}_"``.

    Parameters
    ----------
    pheno_df : pandas.DataFrame
        Phenotype table containing ``"participant_id"`` and Austin BFI item
        columns matched by the prefix ``"bfi_"``.
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

    Austin-specific matching uses the exact prefix ``"bfi_"`` and renames
    it to ``"bfi{n_items}_"`` before integer conversion, missing-column
    padding, and TSV writing.
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
    Aggregate Austin GAD questionnaire data and write the output TSV.

    Compared with the Bonn implementation, which typically matches GAD
    items using the prefix ``"gad_"``, the Austin pipeline uses the more
    specific source prefix ``"gad_7_"`` to identify GAD-7 questionnaire
    items.

    Parameters
    ----------
    pheno_df : pandas.DataFrame
        Phenotype table containing ``"participant_id"`` and Austin GAD item
        columns matched by the exact prefix ``"gad_7_"``.
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

    Austin-specific matching uses the exact prefix ``"gad_7_"`` and renames
    it to ``"gad{n_items}_"`` before integer conversion, missing-column
    padding, and TSV writing.
    """

    # define settings
    current_quest = "gad"
    replace_key = f"{current_quest}_7_"
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
    ) -> Optional[str]:
    """
    Aggregate Austin IUS questionnaire data and write the output TSV.

    Unlike the Bonn implementation, Austin derives the final IUS output
    from a larger questionnaire block. The function locates IUS columns
    between ``"ius_timestamp"`` and ``"ius_complete"``, excludes summary
    fields such as ``"total_score"``, remaps the remaining items to a
    temporary ordered series, and then selects the Austin-specific IUS-12
    subset before writing the result.

    Parameters
    ----------
    pheno_df : pandas.DataFrame
        Phenotype table containing ``"participant_id"`` and Austin IUS
        questionnaire columns.
    phenotype_dir : str
        Output directory where the aggregated TSV file is written.

    Returns
    -------
    out_file : str or None
        Path to the written TSV file, or ``None`` if no usable IUS columns
        are available.

    Raises
    ------
    KeyError
        Raised if required anchor columns for the Austin IUS block are
        missing.
    ValueError
        Raised if the selected IUS columns cannot be transformed into the
        expected output structure.

    Notes
    -----
    Austin-specific item handling differs from Bonn and is based on a
    selected IUS-12 subset rather than a direct prefix-only rename.

    This function writes a TSV file via ``common_write_tsv``.

    Questionnaire item columns are converted to integer values and padded
    to the expected item count before writing.
    """

    # define settings
    current_quest = "ius"
    ius12_idx = lab_pheno.get(current_quest)
    n_items = len(ius12_idx)
    id_key = f"{current_quest}{n_items}_"

    logger.debug(f"IUS12 items: {ius12_idx}")

    # find IUS block
    start_col = "ius_timestamp"
    end_col = "ius_complete"

    # find column range
    cols = pheno_df.columns
    start_idx = cols.get_loc(start_col)
    end_idx = cols.get_loc(end_col)

    ius_cols = cols[start_idx + 1:end_idx]  # skip timestamp and complete

    # remove unwanted columns
    ius_cols = [
        c for c in ius_cols
        if c not in ["total_score"]
    ]

    # rename to ius_1..ius_27
    rename_map = {col: f"{id_key}{i+1}" for i, col in enumerate(ius_cols)}
    pheno_df = pheno_df.rename(columns=rename_map)
    
    # select IUS-12 subset
    ius12_cols = [f"{id_key}{i}" for i in ius12_idx]
    subset = pheno_df[["participant_id"] + ius12_cols]
    subset = subset.rename(
        columns={f"{id_key}{i}": f"{id_key}{j+1}" for j, i in enumerate(ius12_idx)}
    )
    logger.debug("Reformatted IUS columns, continuing as normal")

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


# PHQ
def aggr_phq_data(
        pheno_df: pd.DataFrame,
        phenotype_dir: str
    ) -> Optional[pd.DataFrame]:
    """
    Aggregate Austin PHQ questionnaire data and write the output TSV.

    Compared with the Bonn implementation, which typically matches PHQ
    items using the prefix ``"phq_"``, the Austin pipeline uses the more
    specific source prefix ``"phq_9_"`` to identify PHQ-9 questionnaire
    items.

    Parameters
    ----------
    pheno_df : pandas.DataFrame
        Phenotype table containing ``"participant_id"`` and Austin PHQ item
        columns matched by the exact prefix ``"phq_9_"``.
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

    Austin-specific matching uses the exact prefix ``"phq_9_"`` and renames
    it to ``"phq{n_items}_"`` before integer conversion, missing-column
    padding, and TSV writing.
    """

    # define settings
    current_quest = "phq"
    replace_key = f"{current_quest}_9_"
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
    ) -> Optional[str]:
    """
    Aggregate Austin social connectedness questionnaire data and write the output TSV.

    Unlike the Bonn implementation, which reads social connectedness items
    from ``"soc_q"`` columns, Austin stores these items as ``"midi_*"``
    columns. The function filters the phenotype table to
    ``"participant_id"`` plus numeric ``"midi_*"`` item columns, renames
    them to the shared ``"soc"`` schema, converts them to integers, pads
    missing items, and writes the result.

    Parameters
    ----------
    pheno_df : pandas.DataFrame
        Phenotype table containing ``"participant_id"`` and Austin
        ``"midi_*"`` questionnaire columns.
    phenotype_dir : str
        Output directory where the aggregated TSV file is written.

    Returns
    -------
    out_file : str or None
        Path to the written TSV file, or ``None`` if no usable social
        connectedness columns are available.

    Notes
    -----
    Austin-specific SOC data are sourced from ``"midi_*"`` columns rather
    than ``"soc_*"`` columns.

    Non-item ``"midi_*"`` columns are removed by filtering to
    ``"participant_id"`` and columns matching numeric item names only.

    This function writes a TSV file via ``common_write_tsv``.
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
    Aggregate Austin STAI questionnaire data and write the output TSV.

    Compared with the Bonn implementation, which typically matches STAI
    items using a Bonn-specific STAI source prefix, the Austin pipeline
    uses the more specific source prefix ``"stai_x2_"`` to identify STAI
    questionnaire items.

    Parameters
    ----------
    pheno_df : pandas.DataFrame
        Phenotype table containing ``"participant_id"`` and Austin STAI
        item columns matched by the exact prefix ``"stai_x2_"``.
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

    Austin-specific matching uses the exact prefix ``"stai_x2_"`` and
    renames it to ``"stai{n_items}_"`` before integer conversion,
    missing-column padding, and TSV writing.
    """

    # define settings
    current_quest = "stai"
    replace_key = f"{current_quest}_x2_"
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

# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import re
import os
import pandas as pd
from calinet.imports import biopac
from calinet.config import available_labs
from calinet.utils import (
    map_handedness,
    _read_file_lines,
    _normalize_question_text,
)

from calinet.core.pheno import (
    rename_col,
    common_write_tsv,
    pad_missing_columns,
    convert_questionnaire_columns_to_int,
)

from typing import Any, Optional, Tuple, List, Dict

import logging
logger = logging.getLogger(__name__)

opd = os.path.dirname

# lab-specific pheno info
lab_name = __name__.split(".")[-1]
lab_pheno = available_labs.get(lab_name).get("Phenotype")
language = lab_pheno.get("Language")


def read_raw_physio_file(
        raw_physio_acq: str
    ) -> Tuple[pd.DataFrame, float, Any]:
    """
    Read a Southampton raw physiology ``.acq`` file.

    This implementation is equivalent to the Bonn pipeline in that it reads
    a single ``.acq`` file with ``biopac.read_acq_file`` and returns the
    extracted signal table, sampling rate, and selected channel metadata.

    Parameters
    ----------
    raw_physio_acq : str
        Path to the Southampton ``.acq`` file to read.

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

    Channel selection is resolved from the Southampton lab configuration via
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
        raw_data_dir: str,
        subject_name: str
    ) -> str:
    """
    Locate the Southampton raw physiology ``.acq`` file for one participant.

    Unlike the Bonn implementation, which expects the physiology file
    directly inside the participant directory with the fixed naming pattern
    ``"CALINET_Template{subject_id}.acq"``, the Southampton pipeline moves
    two directory levels up from ``raw_data_dir`` and searches the nested
    directory ``"physio/physio/Task_PhysioData"`` for a file named
    ``"Task_PhysioData_sub{subject_id}.acq"``.

    Parameters
    ----------
    raw_data_dir : str
        Path to the participant-specific raw data directory. The
        Southampton implementation derives the physiology root by applying
        ``dirname(dirname(raw_data_dir))``.
    subject_name : str
        Participant label whose last two characters are used as
        ``subject_id`` in the expected physiology filename.

    Returns
    -------
    physio_path : str
        Full path to the matching Southampton ``.acq`` file.

    Raises
    ------
    FileNotFoundError
        Raised if the expected physiology directory does not exist or if
        the expected ``.acq`` file is not present.

    Notes
    -----
    This function does not read or write files.

    Southampton-specific discovery uses the exact directory
    ``"physio/physio/Task_PhysioData"`` and the exact filename prefix
    ``"Task_PhysioData_sub"``.
    """

    # input is southampton/task/<subject>, so dirname, then into physio
    root_dir = opd(opd(raw_data_dir))
    physio_dir = os.path.join(root_dir, "physio", "physio", "Task_PhysioData")
    if not os.path.isdir(physio_dir):
        raise FileNotFoundError(f"No physio folder at {physio_dir!r}")
    else:
        logger.info(f"Physio-folder: {physio_dir}")

    subject_id = subject_id = subject_name[-2:]
    file_key = f"Task_PhysioData_sub{subject_id}.acq"

    file_name = os.path.join(physio_dir, file_key)
    if file_name and os.path.isfile(file_name):
        return file_name
    else:
        raise FileNotFoundError(f".acq file {file_name} not found")

    
def find_quest_txt(
        raw_data_dir: str
    ) -> List[str]:
    """
    Locate Southampton questionnaire text log files under the raw data tree.

    This helper is Southampton-specific and has no direct Bonn equivalent.
    Unlike the Bonn implementation, which parses a single questionnaire
    export, Southampton stores questionnaire responses across multiple text
    log files and searches recursively for files whose names contain
    ``"quest1"`` or ``"quest2"``.

    Parameters
    ----------
    raw_data_dir : str
        Root directory to search recursively for questionnaire text files.

    Returns
    -------
    questionnaire_files : list of str
        List of full paths to matching questionnaire text files.

    Notes
    -----
    This function does not read file contents and does not write files.

    Matching is case-insensitive and requires filenames to contain either
    ``"quest1"`` or ``"quest2"`` and to end with ``".txt"``.
    """

    questionnaire_files = []
    for root, _, files in os.walk(raw_data_dir):
        for fname in files:
            low = fname.lower()
            if (("quest1" in low) or ("quest2" in low)) and low.endswith(".txt"):
                questionnaire_files.append(os.path.join(root, fname))
    return questionnaire_files


def find_questionnaire_file(
        raw_data_dir: str
    ) -> str:
    """
    Return the Southampton questionnaire root directory.

    Unlike the Bonn implementation, which searches for a single
    questionnaire source file, the Southampton pipeline stores
    questionnaire-related data across multiple text log files and a
    separate Excel metadata file. This function returns ``raw_data_dir``
    unchanged so downstream helpers can locate all required inputs.

    Parameters
    ----------
    raw_data_dir : str
        Root directory containing Southampton questionnaire files.

    Returns
    -------
    questionnaire_file : str
        The input ``raw_data_dir`` passed through unchanged.

    Notes
    -----
    This function does not read questionnaire contents and does not write
    files.

    Southampton-specific questionnaire handling relies on downstream
    helpers that search for ``"quest1"`` and ``"quest2"`` text logs and
    the fixed Excel filename ``"Participant_Info.xlsx"``.
    """

    return raw_data_dir


def parse_quest_txt(
        filepath: str
    ) -> Dict[str, Any]:
    """
    Parse one Southampton questionnaire log text file into structured data.

    Unlike the Bonn implementation, which parses a structured questionnaire
    export, the Southampton pipeline parses raw experiment log files with
    ``*** LogFrame Start ***`` sections and extracts questionnaire
    responses from procedure-specific blocks.

    Parameters
    ----------
    filepath : str
        Path to one Southampton questionnaire text log file.

    Returns
    -------
    participant_dict : dict of str to Any
        Dictionary containing ``"participant_id"`` and extracted
        questionnaire responses.

    Raises
    ------
    ValueError
        Raised if no subject identifier can be extracted from the log file
        header or filename.

    Notes
    -----
    This function reads a text file but does not write output files.

    Southampton-specific extraction uses explicit procedure-based source
    prefixes:
    ``"ius_"`` from ``"iusquest"``,
    ``"stai_"`` from ``"staiquest"``,
    ``"gad_"`` from ``"gadquest"``,
    ``"phq_"`` from ``"phqquest"``,
    ``"midi_"`` from ``"midiquest"``,
    and ``"bfi_"`` from ``"bfiquest"``.

    ``"demoquest"`` responses are converted to normalized free-text column
    names with ``_normalize_question_text``.
    """

    lines = _read_file_lines(filepath)
    participant_dict = {}

    subj = None
    for line in lines:
        m = re.search(r"(?i)^\s*subject\s*:\s*(\d+)", line)
        if m:
            subj = int(m.group(1))
            source = "header"
            break
    if subj is None:
        fname = os.path.basename(filepath)
        m2 = re.search(r"quest1[-_](\d+)[-_]", fname, re.IGNORECASE)
        if m2:
            subj = int(m2.group(1))
            source = "filename"
        else:
            raise ValueError(f"No Subject found in header or filename of {filepath}")

    participant_id = f"sub-{subj:03d}"
    participant_dict["participant_id"] = participant_id
    logger.debug(f"Extracted subject {subj} from {source} in {filepath}")

    # ---  Walk frames and echo everything you see ---
    i = 0
    frame_no = 0
    while i < len(lines):
        if lines[i].strip() == "*** LogFrame Start ***":
            frame_no += 1
            frame = []
            i += 1
            while i < len(lines) and lines[i].strip() != "*** LogFrame End ***":
                frame.append(lines[i])
                i += 1

            # DETECT PROCEDURE
            proc = None
            for L in frame:
                s = L.strip().lower()
                if s.startswith("procedure:"):
                    proc = L.split(":",1)[1].strip().lower()
                    #print(f">>> Procedure = {proc}")
                    break
            if not proc:
                i += 1
                continue

            # --- quest1 sections ---
            if proc == "demoquest":
                qtxt = resp = None
                for L in frame:
                    s = L.strip()
                    if s.startswith("demoQ:"):
                        qtxt = s.split(":",1)[1].strip()
                    elif s.startswith("demoslide.RESP:"):
                        raw = s.split(":",1)[1].strip()
                        resp = raw.split("{",1)[0].strip()
                if qtxt and resp is not None:
                    col = _normalize_question_text(qtxt)
                    participant_dict[col] = resp

            elif proc in ("iusquest","staiquest"):
                prefix = proc.replace("quest","")  # 'ius' or 'stai'
                num = resp = None
                for L in frame:
                    low = L.strip().lower()
                    if low.startswith(f"{prefix}list:"):
                        num = L.split(":",1)[1].strip()
                    elif low.startswith(f"{prefix}slide.resp:"):
                        resp = L.split(":",1)[1].strip()
                if num and resp is not None:
                    col = f"{prefix}_{num}"
                    participant_dict[col] = resp

            # --- quest2 sections ---
            elif proc in ("gadquest","phqquest","midiquest","bfiquest"):
                prefix = proc.replace("quest","")  # e.g. 'gad','phq','midi','bfi'
                num = resp = None
                for L in frame:
                    line = L.strip()
                    low = line.lower()
                    if prefix == "midi":
                        # midi files use "MIDIist:" instead of "MIDIlist:"
                        if low.startswith("midiist:"):
                            num = line.split(":", 1)[1].strip()
                    else:
                        if low.startswith(f"{prefix}list:"):
                            num = line.split(":", 1)[1].strip()
                    if low.startswith(f"{prefix}slide.resp:"):
                        resp = line.split(":",1)[1].strip()
                if num and resp is not None:
                    col = f"{prefix}_{num}"
                    participant_dict[col] = resp

        i += 1

    #print(f"=== Finished {os.path.basename(filepath)} → {participant_dict}\n")
    return participant_dict


def parse_all_quest_txt_files(
        raw_data_dir: str
    ) -> pd.DataFrame:
    """
    Parse all Southampton questionnaire text files and combine them by participant.

    Unlike the Bonn implementation, which parses one questionnaire export,
    the Southampton pipeline parses multiple ``"quest1"`` and ``"quest2"``
    text log files and aggregates them into one row per participant.

    Parameters
    ----------
    raw_data_dir : str
        Root directory containing Southampton questionnaire text files.

    Returns
    -------
    df : pandas.DataFrame
        Aggregated questionnaire table with one row per
        ``"participant_id"`` and columns for all extracted questionnaire
        items.

    Notes
    -----
    This function reads multiple text files but does not write output
    files.

    Duplicate participant entries are collapsed with
    ``groupby("participant_id").first()``.

    Unlike the Reading implementation, handedness is not derived from text
    questionnaire responses here and is instead taken from the separate
    participant metadata file.
    """

    files = find_quest_txt(raw_data_dir)
    if not files:
        logger.warning(f"No 'quest1' files found in {raw_data_dir}; skipping participant pheno.")
        return pd.DataFrame(columns=["participant_id", "age", "sex", "ethnicity", "nationality"])

    all_dicts = []
    for fp in files:
        try:
            d = parse_quest_txt(fp)
            all_dicts.append(d)
        except Exception as e:
            logger.error(f"Failed parsing {fp}: {e}")
            continue

    if not all_dicts:
        return pd.DataFrame(columns=["participant_id", "age", "sex", "ethnicity", "nationality"])

    df = pd.DataFrame(all_dicts)
    df = df.groupby("participant_id", as_index=False).first()

    # Ensure 'participant_id' is first
    cols = df.columns.tolist()
    if "participant_id" in cols:
        cols.insert(0, cols.pop(cols.index("participant_id")))
    df = df[cols]

    return df


def read_participant_file(
        raw_data_dir: str
    ) -> pd.DataFrame:
    """
    Read Southampton participant metadata from the site-specific Excel file.

    Unlike the Bonn implementation, which reads participant metadata from
    the main questionnaire export, the Southampton pipeline reads a
    separate Excel workbook named ``"Participant_Info.xlsx"``.

    Parameters
    ----------
    raw_data_dir : str
        Root directory containing the Excel file
        ``"Participant_Info.xlsx"``.

    Returns
    -------
    pheno_df : pandas.DataFrame
        Participant-level metadata table containing standardized columns
        such as ``"participant_id"``, ``"age"``, ``"sex"``,
        ``"handedness"``, and ``"recorded_at"``.

    Raises
    ------
    ValueError
        Raised if required columns such as ``"participant_id"``, ``"age"``,
        ``"sex"``, or ``"handedness"`` are missing after renaming.

    Notes
    -----
    This function reads an Excel file but does not write output files.

    Southampton-specific processing includes renaming
    ``"Participant_ID"`` to ``"participant_id"``, combining ``"Date"``
    and ``"Time"`` into ``"recorded_at"``, mapping the misspelled source
    column ``"handedeness"`` to ``"handedness"``, normalizing handedness
    with ``map_handedness``, and converting humidity values to percentage
    scale when needed.
    """

    questionnaire_file = os.path.join(raw_data_dir, "Participant_Info.xlsx")

    # If there is no questionnaire, warn and return empty table
    if not questionnaire_file:
        logging.warning(
            f"No questionnaire file found in {raw_data_dir}; skipping participant info."
        )
        print(
            f"No questionnaire file found in {raw_data_dir}; skipping participant info.",
        )
        return pd.DataFrame(columns=["participant_id", "age", "sex", "handedness"])

    pheno_df = pd.read_excel(questionnaire_file)
    pheno_df.columns = pheno_df.columns.str.strip()

    pheno_df.rename(
        columns={
            "Participant_ID": "participant_id",
            "Date": "date",
            "Time": "time",
            "handedeness": "handedness",
        },
        inplace=True
    )

    pheno_df["participant_id"] = pheno_df["participant_id"].apply(lambda x: f"sub-{int(x):03d}")

    # Overwrite time_of_day with "YYYY-MM-DD HH:MM:SS"
    pheno_df["recorded_at"] = pheno_df.apply(
        lambda r: f"{r['date']} {r['time']}" 
                  if pd.notna(r["date"]) and pd.notna(r["time"]) 
                  else pd.NA,
        axis=1
    )

    required_columns = ["participant_id", "age", "sex", "handedness"]
    for col in required_columns:
        if col not in pheno_df.columns:
            print(
                f"Missing required column '{col}' in questionnaire file: {questionnaire_file}"
            )
            raise ValueError(
                f"Missing required column '{col}' in questionnaire file: {questionnaire_file}"
            )

    pheno_df["handedness"] = pheno_df["handedness"].apply(map_handedness)
    
    return pheno_df


def parse_questionnaire_file(
        raw_data_dir: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse Southampton questionnaire data and standardize participant fields.

    Unlike the Bonn implementation, which parses a single questionnaire
    export, the Southampton pipeline combines questionnaire responses from
    text log files with participant metadata from a separate Excel file.

    Parameters
    ----------
    raw_data_dir : str
        Root directory containing Southampton questionnaire text files and
        participant metadata.

    Returns
    -------
    info_df : pandas.DataFrame
        Participant-level table containing ``"participant_id"``, ``"age"``,
        ``"sex"``, and ``"handedness"`` for use in ``participants.tsv``.
    pheno_df : pandas.DataFrame
        Full phenotype table combining questionnaire responses and
        participant metadata.

    Notes
    -----
    This function reads multiple text and Excel files but does not write
    output files.

    Southampton-specific processing removes overlapping columns from the
    participant metadata table before merging, merges on
    ``"participant_id"`` with ``how="inner"``, and then sets
    ``"room_temperature"`` and ``"humidity"`` to ``None`` in the merged
    phenotype table.
    """

    # read subject-wise questionnaire txt files
    df_pheno = parse_all_quest_txt_files(raw_data_dir)

    # read general info
    df_part = read_participant_file(raw_data_dir)

    # remove duplicates
    df_part = df_part.drop(
        columns=df_pheno.columns.intersection(df_part.columns).drop("participant_id")
    )

    # merge
    pheno_df = pd.merge(
        df_part,
        df_pheno,
        on="participant_id",
        how="inner",
        indicator=True
    )

    na_columns = ["room_temperature", "humidity"]
    for col in na_columns:
        pheno_df[col] = None

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
    Aggregate Southampton BFI questionnaire data and write the output TSV.

    Compared with the Bonn implementation, the Southampton pipeline matches
    BFI item columns using the exact source prefix ``"bfi_"`` extracted
    from questionnaire log files.

    Parameters
    ----------
    pheno_df : pandas.DataFrame
        Phenotype table containing ``"participant_id"`` and Southampton BFI
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

    Southampton-specific matching uses the exact prefix ``"bfi_"`` and
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
    Aggregate Southampton GAD questionnaire data and write the output TSV.

    Compared with the Bonn implementation, the Southampton pipeline matches
    GAD item columns using the exact source prefix ``"gad_"`` extracted
    from questionnaire log files.

    Parameters
    ----------
    pheno_df : pandas.DataFrame
        Phenotype table containing ``"participant_id"`` and Southampton GAD
        item columns matched by the prefix ``"gad_"``.
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

    Southampton-specific matching uses the exact prefix ``"gad_"`` and
    renames it to ``"gad{n_items}_"`` before integer conversion, missing
    column padding, and TSV writing.
    """

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
    Aggregate Southampton IUS questionnaire data and write the output TSV.

    Compared with the Bonn implementation, the Southampton pipeline matches
    IUS item columns using the exact source prefix ``"ius_"`` extracted
    from questionnaire log files.

    Parameters
    ----------
    pheno_df : pandas.DataFrame
        Phenotype table containing ``"participant_id"`` and Southampton IUS
        item columns matched by the prefix ``"ius_"``.
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

    Southampton-specific matching uses the exact prefix ``"ius_"`` and
    renames it to ``"ius{n_items}_"`` before integer conversion, missing
    column padding, and TSV writing.
    """

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
    Aggregate Southampton PHQ questionnaire data and write the output TSV.

    Compared with the Bonn implementation, the Southampton pipeline matches
    PHQ item columns using the exact source prefix ``"phq_"`` extracted
    from questionnaire log files.

    Parameters
    ----------
    pheno_df : pandas.DataFrame
        Phenotype table containing ``"participant_id"`` and Southampton PHQ
        item columns matched by the prefix ``"phq_"``.
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

    Southampton-specific matching uses the exact prefix ``"phq_"`` and
    renames it to ``"phq{n_items}_"`` before integer conversion, missing
    column padding, and TSV writing.
    """

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
    Aggregate Southampton SOC questionnaire data and write the output TSV.

    Unlike the Bonn implementation, which may use different SOC source
    naming, the Southampton pipeline matches SOC questionnaire item
    columns using the exact source prefix ``"midi_"`` extracted from
    questionnaire log files.

    Parameters
    ----------
    pheno_df : pandas.DataFrame
        Phenotype table containing ``"participant_id"`` and Southampton SOC
        item columns matched by the prefix ``"midi_"``.
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

    Southampton-specific matching uses the exact prefix ``"midi_"`` and
    renames it to ``"soc{n_items}_"`` before integer conversion, missing
    column padding, and TSV writing.
    """

    # define settings
    current_quest = "soc"
    replace_key = f"midi_"
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
    Aggregate Southampton STAI questionnaire data and write the output TSV.

    Compared with the Bonn implementation, the Southampton pipeline matches
    STAI item columns using the exact source prefix ``"stai_"`` extracted
    from questionnaire log files.

    Parameters
    ----------
    pheno_df : pandas.DataFrame
        Phenotype table containing ``"participant_id"`` and Southampton
        STAI item columns matched by the prefix ``"stai_"``.
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

    Southampton-specific matching uses the exact prefix ``"stai_"`` and
    renames it to ``"stai{n_items}_"`` before integer conversion, missing
    column padding, and TSV writing.
    """

    # define settings
    current_quest = "stai"
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

        # pad with None if indices don't match up
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

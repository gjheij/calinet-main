# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import re
import os
import pandas as pd
from calinet.imports import biopac
from calinet.config import available_labs
from calinet.core.pheno import pad_missing_columns

from calinet.utils import (
    rename_col,
    map_handedness,
    common_write_tsv,
    _read_file_lines,
    extract_subject_id,
    append_acq_date_to_df,
    _normalize_question_text,
    convert_questionnaire_columns_to_int,
)

from typing import Any, Optional, Tuple, Dict, List

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
    Read a Reading raw physiology text file.

    Unlike the Bonn implementation, which reads binary ``.acq`` files using
    ``biopac.read_acq_file``, the Reading pipeline reads physiology data
    from a plain-text export using ``biopac.read_txt_file``.

    Parameters
    ----------
    raw_physio_acq : str
        Path to the Reading physiology text file to read.

    Returns
    -------
    physio_df : pandas.DataFrame
        Physiology data extracted from the input text file.
    sr : float
        Sampling frequency in Hz returned by the text reader.
    chan_info : Any
        Selected channel metadata returned by ``biopac.read_txt_file``.

    Raises
    ------
    Exception
        Raised if the input text file cannot be read.

    Notes
    -----
    This function does not write files.

    Reading-specific physiology import uses ``biopac.read_txt_file`` instead
    of the Bonn ``biopac.read_acq_file`` implementation.
    """

    # Extract physiological data
    logger.info(f"Reading .acq file: {raw_physio_acq}")

    # read acqknowledge file
    lab_name = __name__.split(".")[-1]

    try:
        res = biopac.read_txt_file(
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
    Locate the Reading physiology text file for one participant.

    Unlike the Bonn implementation, which expects a fixed
    ``"CALINET_Template{subject_id}.acq"`` file name, the Reading pipeline
    constructs the exact filename
    ``"Calinet_AcqExt_PhysioData_{subject_id}.txt"`` and searches for it in
    multiple candidate directories derived from ``raw_data_dir``.

    Parameters
    ----------
    raw_data_dir : str
        Path to the participant-specific raw data directory used to derive
        candidate search locations.
    subject_name : str
        Participant label used to derive ``subject_id`` via
        ``extract_subject_id(subject_name)``.

    Returns
    -------
    physio_path : str
        Full path to the matching Reading physiology text file.

    Raises
    ------
    FileNotFoundError
        Raised if the expected text file is not found in any candidate
        directory.

    Notes
    -----
    This function does not read or write files.

    Reading-specific discovery uses the exact filename pattern
    ``"Calinet_AcqExt_PhysioData_{subject_id}.txt"`` and searches both
    ``<base>/<subject_id>/`` and ``<base>/`` across up to three directory
    levels above ``raw_data_dir``. :contentReference[oaicite:0]{index=0}
    """

    subject_id = extract_subject_id(subject_name)
    file_key = f"Calinet_AcqExt_PhysioData_{subject_id}.txt"

    bases = []
    cur = os.path.normpath(raw_data_dir)
    for _ in range(3):
        if cur and cur not in bases:
            bases.append(cur)
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent

    for base in bases:
        for cand in (
            os.path.join(base, subject_id, file_key),
            os.path.join(base, file_key),
        ):
            if os.path.exists(cand):
                return cand

    raise FileNotFoundError(f".acq file '{file_key}' not found in {raw_data_dir!r}")

    
def find_quest_txt(
        raw_data_dir: str
    ) -> List[str]:
    """
    Locate Reading questionnaire text files under the raw data directory.

    This function is Reading-specific and has no equivalent in the Bonn
    pipeline. It recursively scans ``raw_data_dir`` and returns all text
    files whose filenames contain either ``"quest1"`` or ``"quest2"``.

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
    ``"quest1"`` or ``"quest2"`` and end with ``".txt"``.
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
    Return the Reading questionnaire root directory.

    Unlike the Bonn implementation, which searches for a single
    questionnaire file, the Reading pipeline stores questionnaire data
    across multiple text files and a separate Excel file. This function
    returns ``raw_data_dir`` unchanged so downstream helpers can locate all
    required files.

    Parameters
    ----------
    raw_data_dir : str
        Root directory containing Reading questionnaire files.

    Returns
    -------
    questionnaire_file : str
        The input ``raw_data_dir`` passed through unchanged.

    Notes
    -----
    This function does not read questionnaire contents and does not write
    files.

    Reading-specific questionnaire handling relies on downstream functions
    that search for files matching ``"quest1"`` and ``"quest2"`` in text
    files and a fixed Excel filename for participant metadata. :contentReference[oaicite:1]{index=1}
    """

    return raw_data_dir


def parse_quest_txt(
        filepath: str
    ) -> Dict[str, Any]:
    """
    Parse one Reading questionnaire log text file into a participant record.

    Unlike the Bonn implementation, which parses structured questionnaire
    exports, the Reading pipeline parses raw experiment log files with
    ``*** LogFrame Start ***`` blocks and extracts questionnaire responses
    from multiple procedure types.

    Parameters
    ----------
    filepath : str
        Path to one Reading questionnaire text file.

    Returns
    -------
    participant_dict : dict of str to Any
        Dictionary containing ``"participant_id"`` and questionnaire item
        responses extracted from the log file.

    Raises
    ------
    ValueError
        Raised if no subject identifier can be extracted from the file
        header or filename.

    Notes
    -----
    This function reads a text file but does not write output files.

    Reading-specific parsing extracts questionnaire items using explicit
    procedure-based prefixes:
    ``"ius_"`` and ``"stai_"`` from ``"iusquest"`` and ``"staiquest"``,
    ``"gad_"`` from ``"gadquest"``,
    ``"phq_"`` from ``"phqquest"``,
    ``"midi_"`` from ``"midiquest"``,
    and ``"bfi_"`` from ``"bfiquest"``. :contentReference[oaicite:2]{index=2}
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
    Parse all Reading questionnaire text files and combine them per participant.

    This function is Reading-specific and has no equivalent in the Bonn
    pipeline. It parses all files returned by ``find_quest_txt`` and
    aggregates questionnaire responses into one row per participant.

    Parameters
    ----------
    raw_data_dir : str
        Root directory containing Reading questionnaire text files.

    Returns
    -------
    df : pandas.DataFrame
        Combined questionnaire table with one row per ``"participant_id"``
        and columns for all extracted questionnaire items and derived
        variables.

    Notes
    -----
    This function reads multiple text files but does not write output files.

    If no questionnaire files are found or successfully parsed, an empty
    DataFrame with default participant columns is returned.

    Handedness is derived from the column
    ``"are_you_lefthanded_righthanded_or_ambidextrous"`` using
    ``map_handedness``.
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

    df["handedness"] = df[
        "are_you_lefthanded_righthanded_or_ambidextrous"
    ].apply(map_handedness)

    return df


def read_participant_file(
        raw_data_dir: str
    ) -> pd.DataFrame:
    """
    Read Reading participant metadata from the room conditions Excel file.

    This function is Reading-specific and has no equivalent in the Bonn
    pipeline. Unlike Bonn, where participant metadata are included in the
    questionnaire export, Reading stores participant-level information in a
    separate Excel file.

    Parameters
    ----------
    raw_data_dir : str
        Root directory containing the Excel file
        ``"Calinet_Hormonal questionnaire_roomconditions.xlsx"``.

    Returns
    -------
    pheno_df : pandas.DataFrame
        Participant-level metadata table containing standardized columns
        such as ``"participant_id"``, ``"age"``, ``"sex"``,
        ``"recorded_at"``, ``"room_temperature"``, and ``"humidity"``.

    Raises
    ------
    ValueError
        Raised if required columns such as ``"participant_id"``, ``"Age"``,
        or ``"Sex"`` are missing in the Excel file.

    Notes
    -----
    This function reads an Excel file but does not write output files.

    Reading-specific processing includes:
    - Renaming ``"PPN"`` to ``"participant_id"``
    - Formatting ``"participant_id"`` as ``"sub-XXX"``
    - Renaming columns such as ``"Time_of_day"`` to ``"recorded_at"``
    - Converting ``"Room_humidity"`` to percentage if values are in fraction
      form
    """

    questionnaire_file = "Calinet_Hormonal questionnaire_roomconditions.xlsx"
    questionnaire_file = os.path.join(raw_data_dir, questionnaire_file)

    # If there is no questionnaire, warn and return empty table
    if not questionnaire_file:
        logging.warning(
            f"No questionnaire file found in {raw_data_dir}; skipping participant info."
        )
        print(
            f"No questionnaire file found in {raw_data_dir}; skipping participant info.",
        )
        return pd.DataFrame(columns=["participant_id", "age", "sex", "handedness"])

    logger.info(f"Reading room conditions file: {questionnaire_file}")
    pheno_df = pd.read_excel(questionnaire_file)
    pheno_df.columns = pheno_df.columns.str.strip()
    pheno_df.rename(
        columns={
            "PPN": "participant_id",
            "Time_of_day": "recorded_at",

        },
        inplace=True
    )

    required_columns = ["participant_id", "Age", "Sex"]
    for col in required_columns:
        if col not in pheno_df.columns:
            print(
                f"Missing required column '{col}' in questionnaire file: {questionnaire_file}"
            )
            raise ValueError(
                f"Missing required column '{col}' in questionnaire file: {questionnaire_file}"
            )

    pheno_df["participant_id"] = pheno_df["participant_id"].apply(lambda x: f"sub-{int(x):03d}")

    # rename columns
    rename_mapping = {
        "Time_of_day": "recorded_at",
        "Room_temperature": "room_temperature",
        "Room_humidity": "humidity",
        "Age": "age",
        "Sex": "sex"
    }

    pheno_df.rename(columns=rename_mapping, inplace=True)

    # Map each participant to the date of their physiology acquisition file
    raw_path = os.path.dirname(questionnaire_file)
    pheno_df = append_acq_date_to_df(
        pheno_df,
        raw_path,
    )

    # clean time column (important!)
    time_str = pheno_df["recorded_at"].astype(str).str.strip()

    # fix malformed times like "14:00:0" → "14:00:00"
    time_str = time_str.str.replace(
        r"^(\d{2}:\d{2}:\d{1})$", r"\g<1>0", regex=True
    )

    # parse time
    times = pd.to_datetime(time_str, format="%H:%M:%S", errors="coerce")

    # combine date + time
    pheno_df["recorded_at"] = (
        pd.to_datetime(pheno_df["acq_date"]) +
        (times - times.dt.normalize())
    ).dt.strftime("%Y-%m-%dT%H:%M:%S")

    # enforce percentage
    pheno_df["humidity"] = pd.to_numeric(pheno_df["humidity"], errors="coerce")

    pheno_df["humidity"] = pheno_df["humidity"].where(
        pheno_df["humidity"] > 1,
        pheno_df["humidity"] * 100
    )
    
    return pheno_df


def parse_questionnaire_file(
        raw_data_dir: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse Reading questionnaire data and standardize participant fields.

    Unlike the Bonn implementation, which parses a single questionnaire
    export, the Reading pipeline combines questionnaire responses from
    multiple text log files with participant metadata from a separate Excel
    file.

    Parameters
    ----------
    raw_data_dir : str
        Root directory containing Reading questionnaire text files and the
        participant Excel file.

    Returns
    -------
    info_df : pandas.DataFrame
        Participant-level table containing ``"participant_id"``, ``"age"``,
        ``"sex"``, and ``"handedness"`` for use in ``participants.tsv``.
    pheno_df : pandas.DataFrame
        Full parsed phenotype table combining questionnaire responses and
        participant metadata.

    Notes
    -----
    This function reads multiple text and Excel files but does not write
    output files.

    Reading-specific parsing merges questionnaire data extracted from
    ``"quest1"`` and ``"quest2"`` text files with participant metadata from
    ``"Calinet_Hormonal questionnaire_roomconditions.xlsx"``. :contentReference[oaicite:3]{index=3}
    """

    # read subject-wise questionnaire txt files
    df_pheno = parse_all_quest_txt_files(raw_data_dir)

    # read general info
    df_part = read_participant_file(raw_data_dir)

    pheno_df = pd.merge(
        df_part,
        df_pheno,
        on="participant_id",
        how="inner",
        indicator=True
    )

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
    Aggregate Reading BFI questionnaire data and write the output TSV.

    Compared with the Bonn implementation, the Reading pipeline matches BFI
    item columns using the exact source prefix ``"bfi_"`` extracted from log
    files.

    Parameters
    ----------
    pheno_df : pandas.DataFrame
        Phenotype table containing ``"participant_id"`` and Reading BFI item
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

    Reading-specific matching uses the exact prefix ``"bfi_"`` and renames
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
    Aggregate Reading GAD questionnaire data and write the output TSV.

    Compared with the Bonn implementation, the Reading pipeline matches GAD
    item columns using the exact source prefix ``"gad_"`` and excludes
    columns ending with ``"_total"`` before aggregation.

    Parameters
    ----------
    pheno_df : pandas.DataFrame
        Phenotype table containing ``"participant_id"`` and Reading GAD item
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

    Reading-specific matching uses the exact prefix ``"gad_"`` and excludes
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
    Aggregate Reading IUS questionnaire data and write the output TSV.

    Compared with the Bonn implementation, the Reading pipeline matches IUS
    item columns using the exact source prefix ``"ius_"`` and excludes
    columns ending with ``"_total"`` before aggregation.

    Parameters
    ----------
    pheno_df : pandas.DataFrame
        Phenotype table containing ``"participant_id"`` and Reading IUS item
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

    Reading-specific matching uses the exact prefix ``"ius_"`` and excludes
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
    Aggregate Reading PHQ questionnaire data and write the output TSV.

    Compared with the Bonn implementation, the Reading pipeline matches PHQ
    item columns using the exact source prefix ``"phq_"`` and excludes
    columns ending with ``"_total"`` before aggregation.

    Parameters
    ----------
    pheno_df : pandas.DataFrame
        Phenotype table containing ``"participant_id"`` and Reading PHQ item
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

    Reading-specific matching uses the exact prefix ``"phq_"`` and excludes
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
    Aggregate Reading SOC questionnaire data and write the output TSV.

    Unlike the Bonn implementation, which may use different SOC naming,
    the Reading pipeline matches SOC item columns using the exact source
    prefix ``"midi_"`` extracted from log files.

    Parameters
    ----------
    pheno_df : pandas.DataFrame
        Phenotype table containing ``"participant_id"`` and Reading SOC item
        columns matched by the prefix ``"midi_"``.
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

    Reading-specific matching uses the exact prefix ``"midi_"`` and renames
    it to ``"soc{n_items}_"`` before writing. :contentReference[oaicite:8]{index=8}
    """

    # Remove columns that end with '_total'
    pheno_df = pheno_df[
        [col for col in pheno_df.columns if not col.endswith("_total")]
    ]

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
    Aggregate Reading STAI questionnaire data and write the output TSV.

    Unlike the Bonn implementation, which uses Bonn-specific STAI naming,
    the Reading pipeline matches STAI item columns using the exact source
    prefix ``"stai-t_"``.

    Parameters
    ----------
    pheno_df : pandas.DataFrame
        Phenotype table containing ``"participant_id"`` and Reading STAI
        item columns matched by the prefix ``"stai-t_"``.
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

    Reading-specific matching uses the exact prefix ``"stai-t_"`` and
    renames it to ``"stai{n_items}_"`` before writing. :contentReference[oaicite:9]{index=9}
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

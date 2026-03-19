# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import re
import os
import pandas as pd

import pathlib
from io import StringIO

from calinet import utils
from calinet.config import available_labs
from calinet.core.io import save_json, load_json
from calinet.core.metadata import map_participants_tsv
from calinet.templates.common import (
    get_questionnaire_spec,
    PARTICIPANT_INFO_SPEC
)

from typing import Union, Tuple, Optional

import logging
logger = logging.getLogger(__name__)


def gather_all_participant_pheno(
        raw_data_dir: Union[str, pathlib.Path],
        lab_name: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and process participant and phenotype data from a questionnaire file.

    This function uses a lab-specific module to locate and parse a questionnaire
    file within the given raw data directory. It returns cleaned participant
    metadata and associated phenotype data.

    Parameters
    ----------
    raw_data_dir : str or pathlib.Path
        Path to the directory containing raw data files.
    lab_name : str
        Name of the lab, used to dynamically fetch the appropriate parsing module.

    Returns
    -------
    df_participant : pd.DataFrame
        Cleaned participant metadata, with standardized fields (e.g., gender).
    df_pheno : pd.DataFrame
        Phenotype data extracted from the questionnaire file.

    Raises
    ------
    FileNotFoundError
        If no questionnaire file is found in `raw_data_dir`.
    Exception
        If an error occurs while parsing the questionnaire file.

    Notes
    -----
    - The function relies on a lab-specific module retrieved via
      `utils.fetch_lab_module`.
    - Participant data is post-processed using `map_participants_tsv`.

    Examples
    --------
    >>> df_participant, df_pheno = gather_all_participant_pheno(
    ...     "/data/raw", "lab_xyz"
    ... )
    >>> df_participant.head()
    >>> df_pheno.head()
    """
    
    # lab-specific module
    module = utils.fetch_lab_module(lab_name)
    questionnaire_file = module.find_questionnaire_file(raw_data_dir)
    if not questionnaire_file:
        raise FileNotFoundError(f"No questionnaire file found in {raw_data_dir}")
    
    logger.info(f"Questionnaire file: {questionnaire_file}")
    try:
        df_participant, df_pheno = module.parse_questionnaire_file(questionnaire_file)
    except Exception as e:
        raise Exception (f"Error processing questionnaire file: {e}") from e
    
    # clean gender stuff
    df_participant = map_participants_tsv(df_participant)

    return df_participant, df_pheno


def create_spec_json_aggregated(
        measure: str,
        phenotype_dir: str,
        df: Optional[pd.DataFrame]=None,
        lab_name: Optional[str]=None,
        overwrite: bool=False
    ) -> None:
    """
    Create and optionally update a questionnaire specification JSON file.

    This function builds a questionnaire specification dictionary for a given
    measure, annotates reverse-scored items based on dataframe column names and
    lab metadata, verifies the presence of the corresponding TSV file, and writes
    the specification to disk as a JSON file. If a dataframe is provided, the TSV
    file is also rewritten using canonical column names.

    Parameters
    ----------
    measure : str
        Questionnaire or measure name used to retrieve the corresponding
        specification and phenotype metadata.
    phenotype_dir : str
        Path to the directory containing phenotype TSV and JSON files.
    df : pd.DataFrame, optional
        Dataframe containing questionnaire item columns. When provided, reversed
        item suffixes are detected from column names, canonical column names are
        restored, and the TSV file is rewritten.
    lab_name : str, optional
        Name of the lab whose phenotype metadata should be used.
    overwrite : bool, default=False
        Whether to overwrite an existing JSON specification file.

    Returns
    -------
    None
        This function writes files to disk and does not return a value.

    Raises
    ------
    FileNotFoundError
        If the expected TSV file corresponding to the JSON specification does not
        exist.

    Notes
    -----
    Reverse-scored items are inferred from dataframe column names ending in
    ``_r`` or ``_rev``. If the lab metadata contains
    ``items_already_corrected=True``, all items are marked as not reverse-scored
    in the JSON metadata.

    Examples
    --------
    >>> create_spec_json_aggregated(
    ...     measure="ius",
    ...     phenotype_dir="/data/phenotypes",
    ...     df=df_questionnaire,
    ...     lab_name="austin",
    ...     overwrite=True
    ... )
    """

    lab_pheno = available_labs.get(lab_name).get("Phenotype")

    lang = lab_pheno.get("Language")
    n_items = lab_pheno.get(measure)

    # Austin has full IUS where we select subset from, so it's a list
    if isinstance(n_items, list):
        n_items = len(n_items)

    SPEC = get_questionnaire_spec(
        lang,
        measure.upper(),
        n_items
    )

    fname = f"{measure}{n_items}_{lang}"

    # detect reversed items from dataframe
    reverse_map = {}

    if df is not None:
        for col in df.columns:
            if isinstance(col, str) and re.search(r"_(r|rev)$", col):

                canonical = re.sub(r"_(r|rev)$", "", col)
                canonical = re.sub(r'_(0+)(\d+)\b', r'_\2', canonical)

                reverse_map[col] = canonical

    reverse_items = set(reverse_map.values())

    if reverse_map:
        logger.debug(f"{measure}: reverse items detected: {reverse_map}")
    else:
        logger.debug(f"{measure}: no reverse-scored items detected")

    # annotate spec
    already_corrected = lab_pheno.get("items_already_corrected", False)
    if already_corrected:
        logger.warning(
            f"{measure}: 'items_already_corrected=True' → assuming values are already reversed; "
            "metadata will not mark ReverseScored items"
        )

    for key, meta in SPEC.items():
        if not isinstance(meta, dict):
            continue

        if already_corrected:
            # values already corrected → nothing to mark
            meta["ReverseScored"] = False
        else:
            meta["ReverseScored"] = key in reverse_items

    # paths
    json_path = os.path.join(phenotype_dir, f"{fname}.json")
    tsv_path = json_path.replace(".json", ".tsv")

    if not os.path.exists(tsv_path):
        raise FileNotFoundError(
            f"Mismatch between json and tsv file: json='{json_path}', but corresponding '{tsv_path}' does not exist"
        )

    # rewrite TSV (only for questionnaires)
    if df is not None:

        df_tsv = df.copy()

        renamed_cols = {
            col: re.sub(r'_(r|rev)$', '', re.sub(r'_(0+)(\d+)\b', r'_\2', col))
            for col in df_tsv.columns if isinstance(col, str)
        }

        changed_cols = {
            old: new for old, new in renamed_cols.items()
            if old != new
        }

        if changed_cols:
            df_tsv = df_tsv.rename(columns=changed_cols)

            logger.debug(f"{measure}: rewriting TSV with canonical column names")
            for old, new in changed_cols.items():
                logger.debug(f"{measure}: renamed column '{old}' → '{new}'")

            # use same function to ensure Int64 with n/a; otherwise floats..
            _ = utils.common_write_tsv(
                df_tsv,
                id_key=f"{measure}{n_items}_",
                language=lang,
                phenotype_dir=phenotype_dir
            )

    # write JSON
    if not os.path.exists(json_path) or overwrite:
        save_json(json_path, SPEC)
        logger.info(f"{measure} specification JSON saved to: {json_path}")


def handle_pheno(
        all_data: pd.DataFrame,
        phenotype_dir: str,
        lab_name: Optional[str]=None,
        overwrite: bool=False
    ) -> None:
    """
    Create aggregated phenotype TSV and JSON specification files.

    This function generates participant-level and questionnaire-level phenotype
    outputs for all subjects. It writes a participant TSV file, applies a set of
    lab-specific aggregation functions to produce questionnaire data, and creates
    corresponding JSON specification files with reverse-scoring metadata.

    Parameters
    ----------
    all_data : pd.DataFrame
        Input dataframe containing all subject-level raw or preprocessed data
        required for phenotype aggregation.
    phenotype_dir : str
        Path to the directory where aggregated TSV and JSON files will be saved.
    lab_name : str, optional
        Name of the lab used to retrieve lab-specific aggregation functions.
    overwrite : bool, default=False
        Whether to overwrite existing JSON specification files.

    Returns
    -------
    None
        This function writes multiple TSV and JSON files to disk and does not
        return a value.

    Raises
    ------
    Exception
        If any questionnaire aggregation function fails.
    Exception
        If writing the JSON specification for a questionnaire fails.

    Notes
    -----
    - Participant information is written using `participant_write_tsv`.
    - Questionnaire aggregation is performed via lab-specific functions
      (e.g., `aggr_bfi_data`, `aggr_gad_data`).
    - For each successfully aggregated questionnaire, metadata is generated
      using `create_spec_json_aggregated`.

    Examples
    --------
    >>> handle_pheno(
    ...     all_data=df_all,
    ...     phenotype_dir="/data/phenotypes",
    ...     lab_name="austin",
    ...     overwrite=True
    ... )
    """

    os.makedirs(phenotype_dir, exist_ok=True)

    # write participant info
    _ = participant_write_tsv(
        all_data,
        phenotype_dir=phenotype_dir
    )

    # fetch lab-specific functions
    module = utils.fetch_lab_module(lab_name)
    jobs = [
        ("bfi", module.aggr_bfi_data),
        ("gad", module.aggr_gad_data),
        ("ius", module.aggr_ius_data),
        ("phq", module.aggr_phq_data),
        ("soc", module.aggr_soc_data),
        ("stai", module.aggr_stai_data),
    ]

    for name, aggregator_fn in jobs:

        # call site-specific aggregation functions
        df_quest = None
        try:
            df_quest = aggregator_fn(all_data, phenotype_dir)
        except Exception as e:
            raise Exception(f"Aggregation of {name} failed: {e}") from e
        
        if df_quest is not None:
            # write json collectively -> Assign ReverseScored in meta
            try:
                create_spec_json_aggregated(
                    name,
                    phenotype_dir,
                    df=df_quest,
                    lab_name=lab_name,
                    overwrite=overwrite
                )
            except Exception as e:
                raise Exception(f"Failed to write metadata for '{name}': {e}") from e

    logger.info("Aggregated phenotype files created in %s", phenotype_dir)


def pad_missing_columns(
        subset: pd.DataFrame,
        n_items: int,
        id_key: str
    ) -> pd.DataFrame:
    """
    Ensure a dataframe contains the expected number of questionnaire columns.

    This function checks whether the input dataframe has the expected set of
    columns based on the number of items (`n_items`) and an identifier prefix
    (`id_key`). If columns are missing, they are added and filled with ``None``.
    If extra columns are present, a warning is logged.

    Parameters
    ----------
    subset : pd.DataFrame
        Dataframe containing participant responses for a specific questionnaire.
    n_items : int
        Expected number of questionnaire items.
    id_key : str
        Prefix used for item column names (e.g., "bfi_", "gad_").

    Returns
    -------
    subset : pd.DataFrame
        Dataframe with missing columns added (if necessary) to match the expected
        structure.

    Notes
    -----
    - The expected columns are ``participant_id`` plus item columns in the form
      ``{id_key}{i}`` for ``i = 1..n_items``.
    - Columns with reverse suffixes (e.g., ``_r``) are considered valid substitutes
      and will not be duplicated.
    - Missing columns are filled with ``None``.

    Examples
    --------
    >>> df = pad_missing_columns(df_subset, n_items=10, id_key="bfi_")
    >>> df.shape
    """

    expected_base_cols = ["participant_id"] + [f"{id_key}{i}" for i in range(1, n_items + 1)]

    if subset.shape[1] < (n_items + 1):
        logger.warning(
            f"Reformatted '{id_key}' data has {subset.shape[1]} columns, "
            f"but {n_items+1} were expected -> padding with None"
        )

        existing_cols = set(subset.columns)

        for col in expected_base_cols:
            if col in existing_cols:
                continue
            if col != "participant_id" and f"{col}_r" in existing_cols:
                continue

            subset[col] = None
            logger.debug(f"{id_key}: added missing column '{col}' with None")
    elif subset.shape[1] > (n_items + 1):
        logger.warning(f"'{id_key}' dataframe has {subset.shape[1]} items, but only {n_items+1} were expected. Unwanted columns may be present..")

    return subset


def participant_write_tsv(
        df: pd.DataFrame,
        phenotype_dir: Optional[str]=None
    ) -> pd.DataFrame:
    """
    Create a standardized participant TSV file and corresponding JSON metadata.

    This function selects a predefined set of participant-related columns,
    ensures missing columns are added with ``None``, enforces consistent data
    formatting (timestamps, humidity, categorical values), and optionally writes
    the result to disk along with a JSON specification file.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing participant-level information.
    phenotype_dir : str, optional
        Directory where the TSV and JSON files will be saved. If ``None``, no
        files are written.

    Returns
    -------
    subset : pd.DataFrame
        Dataframe containing standardized participant information.

    Notes
    -----
    - Missing required columns are added and filled with ``None``.
    - The ``recorded_at`` column is converted to ISO-like format
      (``YYYY-MM-DDTHH:MM:SS``).
    - The ``humidity`` column is coerced to numeric and scaled to percentage
      if values are less than or equal to 1.
    - The ``sex`` column is normalized via `map_participants_tsv`.
    - Output files:
        - TSV: ``participant_info.tsv``
        - JSON: ``participant_info.json``

    Examples
    --------
    >>> df_participants = participant_write_tsv(
    ...     df_raw,
    ...     phenotype_dir="/data/phenotypes"
    ... )
    >>> df_participants.head()
    """

    # participant_info.json items
    columns_to_keep = [
        "participant_id",
        "recorded_at",
        "room_temperature",
        "humidity",
        "age",
        "sex",
        "handedness",
    ]

    # fill with None if columns do not exist
    add_na_cols = [col for col in columns_to_keep if col not in df.columns]
    if add_na_cols:
        logger.debug(f"Setting following columns to None: {add_na_cols}")
        for col in add_na_cols:
            df[col] = None

    subset = df[columns_to_keep].copy()

    # enfore consistent timestamp -> may add 1900-01-01, but you can parse from 'T'
    s = subset["recorded_at"].astype(str).str.strip()
    parsed = pd.to_datetime(s, errors="coerce")
    subset["recorded_at"] = parsed.dt.strftime("%Y-%m-%dT%H:%M:%S")

    # enforce percentage
    subset["humidity"] = pd.to_numeric(subset["humidity"], errors="coerce")

    subset["humidity"] = subset["humidity"].where(
        subset["humidity"] > 1,
        subset["humidity"] * 100
    )

    # reformat sex column
    subset = map_participants_tsv(subset)

    if phenotype_dir is not None:
        filename = os.path.join(phenotype_dir, "participant_info.tsv")
        subset.to_csv(filename, sep="\t", index=False, na_rep="n/a")

        # write JSON
        json_path = filename.replace(".tsv", ".json")
        save_json(json_path, PARTICIPANT_INFO_SPEC)
        logger.info(f"PARTICIPANT_INFO specification JSON saved to: {json_path}")

    # write
    return subset


def _load_sidecar(sidecar):
    if isinstance(sidecar, dict):
        return sidecar

    if isinstance(sidecar, str):
        return load_json(sidecar)

    raise ValueError("sidecar must be dict, JSON string, or path to JSON file")


def _derive_sidecar_path(df, sidecar):
    if sidecar is not None:
        return sidecar
    if isinstance(df, str) and os.path.exists(df):
        base, _ = os.path.splitext(df)
        return base + ".json"
    return sidecar


def _load_dataframe(data):
    """
    Accept:
      - pd.DataFrame
      - path to CSV/TSV
      - raw CSV/TSV string
    """

    if isinstance(data, pd.DataFrame):
        return data

    if isinstance(data, str):
        s = data.strip()

        # Case 1: file path
        if os.path.exists(s):
            if s.endswith(".tsv"):
                return pd.read_csv(s, sep="\t")
            else:
                return pd.read_csv(s)

        # Case 2: raw string → try TSV first, fallback CSV
        try:
            return pd.read_csv(StringIO(s), sep="\t")
        except Exception:
            return pd.read_csv(StringIO(s))

    raise ValueError("df must be DataFrame, file path, or raw CSV/TSV string")


def _collect_scale_items(df, sidecar, prefix_regex, scale_name):
    """
    Find questionnaire item columns from sidecar + df.

    prefix_regex examples:
    - r"gad7"
    - r"phq(?:\\d+)?"
    - r"ius(?:\\d+)?"
    - r"stai(?:\\d+)?"
    """
    item_pattern = re.compile(rf"^({prefix_regex})_(\d+)$")
    item_meta = {}

    for col in sidecar.keys():
        m = item_pattern.match(col)
        if m and col in df.columns:
            prefix, item_num = m.group(1), int(m.group(2))
            item_meta[col] = {
                "prefix": prefix,
                "item_num": item_num,
            }

    if not item_meta:
        available = [c for c in df.columns if "_" in c][:20]
        raise ValueError(
            f"No matching {scale_name} item columns found. "
            f"Example columns: {available}"
        )

    prefixes = {meta["prefix"] for meta in item_meta.values()}
    if len(prefixes) != 1:
        raise ValueError(f"Expected one {scale_name} prefix, found: {prefixes}")

    prefix = next(iter(prefixes))
    item_cols = sorted(item_meta, key=lambda c: item_meta[c]["item_num"])
    return prefix, item_cols, item_meta


BIG_FIVE_ORDER = [
    "Extraversion",
    "Agreeableness",
    "Conscientiousness",
    "NegativeEmotionality",
    "OpenMindedness",
]

def _score_sum_scale(
    df,
    sidecar=None,
    *,
    prefix_regex,
    scale_name,
    output_prefix=None,
    expected_n_items=None,
    add_n_items=True,
):
    """
    Generic total-score scorer for simple sum scales.
    """

    sidecar = _derive_sidecar_path(df, sidecar)
    df = _load_dataframe(df)
    sidecar = _load_sidecar(sidecar)

    detected_prefix, item_cols, item_meta = _collect_scale_items(
        df=df,
        sidecar=sidecar,
        prefix_regex=prefix_regex,
        scale_name=scale_name,
    )

    if expected_n_items is not None and len(item_cols) != expected_n_items:
        raise ValueError(
            f"Expected {expected_n_items} {scale_name} items, found {len(item_cols)}"
        )

    scored = df.copy()
    for col in item_cols:
        scored[col] = pd.to_numeric(scored[col], errors="coerce")

    prefix = output_prefix or detected_prefix

    out = pd.DataFrame(index=scored.index)
    if "participant_id" in scored.columns:
        out["participant_id"] = scored["participant_id"]

    out[f"{prefix}_total"] = scored[item_cols].sum(axis=1, skipna=True)
    out[f"{prefix}_mean"] = scored[item_cols].mean(axis=1, skipna=True)
    out[f"{prefix}_n_answered"] = scored[item_cols].notna().sum(axis=1)

    if add_n_items:
        out[f"{prefix}_n_items"] = len(item_cols)

    return out


def score_bfi(df, sidecar=None):
    """
    BFI is special because items roll up into 5 traits.
    """
    sidecar = _derive_sidecar_path(df, sidecar)
    df = _load_dataframe(df)
    sidecar = _load_sidecar(sidecar)

    item_pattern = re.compile(r"^(bfi\d+)_(\d+)$")
    item_meta = {}

    for col, meta in sidecar.items():
        m = item_pattern.match(col)
        if m and col in df.columns:
            prefix, item_num = m.group(1), int(m.group(2))
            item_meta[col] = {
                "prefix": prefix,
                "item_num": item_num,
            }

    if not item_meta:
        raise ValueError("No matching BFI item columns found.")

    prefixes = {item_meta[c]["prefix"] for c in item_meta}
    if len(prefixes) != 1:
        raise ValueError(f"Expected one questionnaire prefix, found: {prefixes}")

    item_cols = sorted(item_meta, key=lambda c: item_meta[c]["item_num"])

    scored = df.copy()
    for col in item_cols:
        scored[col] = pd.to_numeric(scored[col], errors="coerce")

    domain_to_items = {d: [] for d in BIG_FIVE_ORDER}
    for col in item_cols:
        item_num = item_meta[col]["item_num"]
        domain = BIG_FIVE_ORDER[(item_num - 1) % 5]
        domain_to_items[domain].append(col)

    out = pd.DataFrame(index=scored.index)
    if "participant_id" in scored.columns:
        out["participant_id"] = scored["participant_id"]

    for domain, cols in domain_to_items.items():
        out[f"{domain}_total"] = scored[cols].sum(axis=1, skipna=True)
        out[f"{domain}_mean"] = scored[cols].mean(axis=1, skipna=True)
        out[f"{domain}_n_answered"] = scored[cols].notna().sum(axis=1)

    return out


def score_gad(df, sidecar=None):
    out = _score_sum_scale(
        df,
        sidecar,
        prefix_regex=r"gad7",
        scale_name="GAD",
        output_prefix="gad",
        expected_n_items=7,
    )

    def categorize(score):
        if pd.isna(score):
            return None
        if score <= 4:
            return "minimal"
        if score <= 9:
            return "mild"
        if score <= 14:
            return "moderate"
        return "severe"

    out["gad_severity"] = out["gad_total"].apply(categorize)
    return out


def score_phq(df, sidecar=None):
    out = _score_sum_scale(
        df,
        sidecar,
        prefix_regex=r"phq(?:\d+)?",
        scale_name="PHQ",
        output_prefix="phq",
    )

    if out["phq_n_items"].iloc[0] == 9:
        def categorize(score):
            if pd.isna(score):
                return None
            if score <= 4:
                return "minimal"
            if score <= 9:
                return "mild"
            if score <= 14:
                return "moderate"
            if score <= 19:
                return "moderately severe"
            return "severe"

        out["phq_severity"] = out["phq_total"].apply(categorize)

    return out


def score_ius(df, sidecar=None):
    return _score_sum_scale(
        df,
        sidecar,
        prefix_regex=r"ius(?:\d+)?",
        scale_name="IUS",
        output_prefix="ius",
    )


def score_soc(df, sidecar=None):
    return _score_sum_scale(
        df,
        sidecar,
        prefix_regex=r"soc(?:\d+)?",
        scale_name="SOC",
        output_prefix="soc",
    )


def score_stai(df, sidecar=None):
    return _score_sum_scale(
        df,
        sidecar,
        prefix_regex=r"stai(?:\d+)?",
        scale_name="STAI",
        output_prefix="stai",
    )

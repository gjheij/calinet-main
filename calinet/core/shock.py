# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import pandas as pd
from calinet.templates.common import get_questionnaire_spec
from calinet.core.io import save_json
from calinet.core.pheno import (
    common_write_tsv,
    convert_questionnaire_columns_to_int
)

import logging
logger = logging.getLogger(__name__)


def extract_task_ratings_from_events_df(
        task_name: str,
        events_df: pd.DataFrame
    ) -> dict:
    """
    Extract and organize task-specific ratings from an events DataFrame.

    This function parses rating responses from an events DataFrame and assigns
    them to task-specific categories (e.g., pre-/post-acquisition, post-extinction).
    It handles incomplete data gracefully by issuing warnings and returning
    partially filled rating structures when necessary.

    Parameters
    ----------
    task_name : {"acquisition", "extinction"}
        Name of the task for which ratings should be extracted.
    events_df : pandas.DataFrame
        DataFrame containing event data, expected to include a column
        ``"rating_slider.response"`` with rating values.

    Returns
    -------
    this_task_ratings : dict
        Dictionary containing extracted ratings. Structure depends on `task_name`:

        - If ``task_name == "acquisition"``:
          - ``"pre_acquisition_ratings"`` : list
          - ``"post_acquisition_ratings"`` : list

        - If ``task_name == "extinction"``:
          - ``"post_extinction_ratings"`` : list

        If ratings are missing or incomplete, a default structure (from
        `get_blank_task_ratings`) is returned with available values filled in.

    Raises
    ------
    ValueError
        If `task_name` is not one of {"acquisition", "extinction"}.

    Notes
    -----
    Processing steps:

    1. Column check:
       - Verifies that ``"rating_slider.response"`` exists in `events_df`
       - Returns default ratings if missing

    2. Data extraction:
       - Drops NaNs and converts ratings to integers

    3. Task-specific handling:

       Acquisition:
        - Expected: 12 ratings (6 pre + 6 post)
        - <6 ratings: insufficient → return defaults
        - ==6 ratings: treated as pre-acquisition only
        - 6–11 ratings: partial post-acquisition data
        - ≥12 ratings: full split into pre/post

       Extinction:
        - Expected: up to 8 ratings (post-extinction)
        - <1 rating: return defaults
        - <8 ratings: partial fill
        - ≥8 ratings: full assignment

    The function prioritizes robustness by returning valid structures even when
    data are incomplete.

    Examples
    --------
    >>> extract_task_ratings_from_events_df("acquisition", df)
    {
        "pre_acquisition_ratings": [...],
        "post_acquisition_ratings": [...]
    }

    >>> extract_task_ratings_from_events_df("extinction", df)
    {
        "post_extinction_ratings": [...]
    }
    """

    # get blank dictionary
    this_task_ratings = get_blank_task_ratings(task_name)

    if not "rating_slider.response" in events_df.columns:
        logger.warning(f"'rating_slider.response' column not found in the events dataframe for task {task_name}; returning 'n/a' dictionary")
        return this_task_ratings
    
    # get ratings and convert to float
    ratings = events_df["rating_slider.response"].dropna()
    ratings_list = ratings.tolist()
    ratings_list = [int(float(r)) for r in ratings_list]

    # split based on task and availability
    if task_name == "acquisition":
        
        # only pre-acq ratings available
        if len(ratings_list)<6:
            logger.warning(f"Only {len(ratings_list)} scores are present in rating list; need at least 6 for this split [{task_name}]")
            return this_task_ratings
        elif len(ratings_list)==6:
            # if len(ratings_list)==6, assume pre-acq
            logger.warning(f"Only {len(ratings_list)} scores available; assuming 'pre-acquisition' ratings!")
            this_task_ratings["pre_acquisition_ratings"] = ratings
            return this_task_ratings
        elif 6<len(ratings_list)<12:
            # pre-acq ratings available, limited number of post-acq
            logger.warning(f"Only {len(ratings_list)} out of 12 scores are present in rating list; assuming first 6 are 'pre-acq', the remaining ({len(ratings_list[6:])}) items are assigned to 'post-acq'")

            this_task_ratings["pre_acquisition_ratings"][:6] = ratings_list[:6]
            this_task_ratings["post_acquisition_ratings"][0:len(ratings_list[6:])] = ratings_list[0:len(ratings_list[6:])]

            return this_task_ratings
        
        # split full ratings_list
        this_task_ratings = {
            "pre_acquisition_ratings": ratings_list[:6],
            "post_acquisition_ratings": ratings_list[6:12],
        }

        return this_task_ratings
    elif task_name == "extinction":

        if len(ratings_list)<1:
            logger.warning(f"Zero scores available for 'post-extinction' phase!")
            return this_task_ratings
        elif len(ratings_list)<8:
            logger.warning(f"Only {len(ratings_list)} scores are present in rating list; assuming first {len(ratings_list)} out of 8 items")
            
            this_task_ratings["post_extinction_ratings"][:len(ratings_list)] = ratings_list
        else:
            this_task_ratings["post_extinction_ratings"] = ratings_list

        return this_task_ratings
    else:
        raise ValueError(f"task must be one of 'acquisition' or 'extinction', not '{task_name}'")
    

def get_blank_task_ratings(
        task_name: str,
        na_rep=None
    ) -> dict:
    """
    Generate a blank ratings dictionary for a given task.

    This function creates a standardized dictionary structure for storing
    task-specific ratings, initializing all entries with a specified placeholder
    value (e.g., ``None`` or ``"n/a"``).

    Parameters
    ----------
    task_name : {"acquisition", "extinction"}
        Name of the task for which to generate the ratings structure.
    na_rep : any, optional
        Placeholder value used to fill the rating entries (default is ``None``).

    Returns
    -------
    this_task_ratings : dict or None
        Dictionary containing placeholder ratings:

        - If ``task_name == "acquisition"``:
          - ``"pre_acquisition_ratings"`` : list of length 6
          - ``"post_acquisition_ratings"`` : list of length 6

        - If ``task_name == "extinction"``:
          - ``"post_extinction_ratings"`` : list of length 8

        Returns ``None`` if `task_name` is not recognized.

    Notes
    -----
    The function ensures a consistent structure for downstream processing,
    even when actual rating data are missing.

    Examples
    --------
    >>> get_blank_task_ratings("acquisition")
    {
        "pre_acquisition_ratings": [None, None, None, None, None, None],
        "post_acquisition_ratings": [None, None, None, None, None, None]
    }

    >>> get_blank_task_ratings("extinction", na_rep="n/a")
    {
        "post_extinction_ratings": ["n/a", ..., "n/a"]
    }
    """

    if task_name == "acquisition":
        this_task_ratings = {
            "pre_acquisition_ratings": [na_rep for i in range(6)],
            "post_acquisition_ratings": [na_rep for i in range(6)],
        }
        return this_task_ratings
    elif task_name == "extinction":
        this_task_ratings = {"post_extinction_ratings": [na_rep for i in range(8)]}
        return this_task_ratings
    return None


def get_json_template(ratings_type, language=None):
    if ratings_type == "pre_acquisition_ratings":
        return get_questionnaire_spec(language, "ratings", "pre-acq")
    elif ratings_type == "post_acquisition_ratings":
        return get_questionnaire_spec(language, "ratings", "post-acq")
    elif ratings_type == "post_extinction_ratings":
        return get_questionnaire_spec(language, "ratings", "post-ext")
    else:
        logger.warning(f"Unknown ratings type: {ratings_type}")
        return None


def accumulate_shock_ratings(
        aggregated_dict: dict,
        subject_name: str,
        task_ratings: dict,
        language: str="german"
    ) -> dict:
    """
    Accumulate subject-level shock ratings into an aggregated structure.

    This function appends task-specific rating data for a subject to a shared
    aggregation dictionary. It ensures that ratings are aligned with a predefined
    JSON template and stored in a structured, tabular-like format.

    Parameters
    ----------
    aggregated_dict : dict
        Dictionary used to collect ratings across subjects. Keys correspond to
        rating types (e.g., "pre_acquisition_ratings"), and values are lists of
        records (dictionaries).
    subject_name : str
        Identifier for the subject. This is inserted as the first column in each
        record.
    task_ratings : dict
        Dictionary containing task-specific ratings. Keys correspond to rating
        types, and values are sequences of ratings.
    language : str, default="german"
        Language used to retrieve the appropriate JSON template via
        `get_json_template`.

    Returns
    -------
    aggregated_dict : dict
        Updated aggregation dictionary with appended subject-level records.

    Raises
    ------
    Exception
        If no JSON template is found for a given rating type and language.
    ValueError
        If the number of values does not match the expected number of columns
        defined in the template.

    Notes
    -----
    The function performs the following steps for each rating type:

    1. Template retrieval:
       - Loads a JSON template using `get_json_template`
       - Extracts column names (excluding metadata fields)

    2. Row construction:
       - Prepends `subject_name` to the rating values
       - Ensures the number of values matches the expected columns

    3. Record creation:
       - Combines column names and values into a dictionary

    4. Aggregation:
       - Appends the record to `aggregated_dict[rating_type]`
       - Initializes the list if the key does not yet exist

    Assumptions:
        - Templates returned by `get_json_template` have consistent structure
        - The first column corresponds to subject identifiers

    Examples
    --------
    >>> agg = {}
    >>> accumulate_shock_ratings(
    ...     agg,
    ...     subject_name="sub-01",
    ...     task_ratings={"pre_acquisition_ratings": [1, 2, 3, 4, 5, 6]}
    ... )
    >>> agg["pre_acquisition_ratings"][0]["subject"]
    'sub-01'
    """

    for ratings_type, ratings in task_ratings.items():
        ratings_json_template = get_json_template(
            ratings_type,
            language=language
        )

        if ratings_json_template is None:
            raise Exception(
                f"Could not find template for rating='{ratings_type}' and language='{language}'"
            )

        columns = list(ratings_json_template.keys())[1:]  # includes subject column + rating columns
        
        row_values = [subject_name] + list(ratings)

        if len(row_values) != len(columns):
            raise ValueError(
                f"{ratings_type}: expected {len(columns)} columns, got {len(row_values)} values"
            )

        record = dict(zip(columns, row_values))
        aggregated_dict.setdefault(ratings_type, []).append(record)


def write_aggregated_shock_ratings(
    aggregated_dict: dict,
    phenotype_dir: str,
    language: str = "german"
) -> dict:
    """
    Write aggregated shock ratings to TSV and JSON files.

    This function converts aggregated subject-level rating records into
    tabular format, ensures proper column ordering and data types, and writes
    both TSV files and corresponding JSON sidecar metadata files for each
    rating type.

    Parameters
    ----------
    aggregated_dict : dict
        Dictionary containing aggregated rating records. Keys correspond to
        rating types (e.g., "pre_acquisition_ratings"), and values are lists
        of dictionaries representing subject-level records.
    phenotype_dir : str
        Directory where output TSV and JSON files will be written.
    language : str, default="german"
        Language used to retrieve JSON templates via `get_json_template`.

    Returns
    -------
    ddict : dict
        Dictionary mapping rating types to pandas DataFrames containing the
        processed and written data.

    Raises
    ------
    Exception
        If no JSON template is found for a given rating type and language.

    Notes
    -----
    The function performs the following steps for each rating type:

    1. Template retrieval:
       - Loads a JSON template using `get_json_template`
       - Extracts column names (excluding metadata fields)

    2. DataFrame creation:
       - Converts list of records into a pandas DataFrame
       - Reorders columns to match the JSON template

    3. Data type enforcement:
       - Converts rating columns (excluding subject column) to integers using
         `convert_questionnaire_columns_to_int`

    4. File writing:
       - Writes TSV file using `common_write_tsv`
       - Writes corresponding JSON sidecar file using `save_json`

    5. Output collection:
       - Stores the resulting DataFrame in the output dictionary

    Naming convention:
        - TSV and JSON files are named using a CamelCase version of the rating type
          (e.g., "pre_acquisition_ratings" → "PreAcquisitionRatings.tsv/json")

    Assumptions:
        - Records in `aggregated_dict` follow a consistent structure
        - Templates from `get_json_template` define the correct column order
        - `common_write_tsv` handles file naming and saving logic

    Examples
    --------
    >>> dfs = write_aggregated_shock_ratings(
    ...     aggregated_dict=agg,
    ...     phenotype_dir="derivatives/phenotype/"
    ... )

    >>> dfs["pre_acquisition_ratings"].head()
    """
    
    os.makedirs(phenotype_dir, exist_ok=True)

    ddict = {}
    for ratings_type, records in aggregated_dict.items():
        
        ratings_json_template = get_json_template(
            ratings_type,
            language=language
        )

        if ratings_json_template is None:
            raise Exception(
                f"Could not find template for rating='{ratings_type}' and language='{language}'"
            )

        columns = list(ratings_json_template.keys())[1:]
        ratings_name = "".join(word.title() for word in ratings_type.split("_"))

        df = pd.DataFrame.from_records(records)

        # Ensure column order matches JSON template
        df = df.reindex(columns=columns)

        # ensure integers
        rating_cols = df.columns[1:]
        df = convert_questionnaire_columns_to_int(df, rating_cols)
        df = common_write_tsv(
            df,
            ratings_name,
            language="",
            phenotype_dir=phenotype_dir
        )

        # write sidecar
        json_file_name = f"{ratings_name}.json"
        json_file_path = os.path.join(phenotype_dir, json_file_name)
        save_json(json_file_path, ratings_json_template)

        # compile output
        ddict[ratings_type] = df

    return ddict

# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from calinet.config import (
    config,
    available_labs
)

from calinet.core.io import (
    save_json
)

from calinet.data import run_pspm_trim_directory

from calinet.core.metadata import (
    create_readme,
    create_dataset_description,
    create_bidsignore
)

from calinet.templates.common import (
    EVENTS_JSON_TEMPLATE,
    PARTICIPANTS_JSON_TEMPLATE
)

from calinet.utils import (
    clean_output_directory,
    find_sub_dirs,
    extract_subject_name
)

from calinet.core.shock import (
    accumulate_shock_ratings,
    write_aggregated_shock_ratings
)

from calinet.core.events import handle_events
from calinet.core.pheno import (
    handle_pheno,
    gather_all_participant_pheno
)
from calinet.core.pupil import handle_eyetracking

from calinet.core.anonymize import anonymize_converted_data
from calinet.core.physio import handle_physio

from calinet.plotting import _generate_single_qa_plot
from calinet.logger import worker_init, current_subject

import logging
logger = logging.getLogger(__name__)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

from typing import List, Tuple, Optional


def _process_single_subject(
        raw_dir: str,
        conv_dir: str,
        dataset_name: str,
        lab_meta_name: str,
        trim_window: List[float],
    ) -> Tuple[str, pd.DataFrame]:
    """
    Process a single subject and return subject-local task ratings.

    This function runs the full per-subject processing pipeline, including
    physiological signal processing, event extraction, eyetracking handling,
    and trimming of physiological data. It is defined at module level to remain
    compatible with Windows multiprocessing.

    Parameters
    ----------
    raw_dir : str
        Path to the subject's raw data directory.
    conv_dir : str
        Path to the root directory for converted output data.
    dataset_name : str
        Name of the dataset or lab configuration used for physiological
        processing.
    lab_meta_name : str
        Name of the lab metadata configuration used for eyetracking
        processing.
    trim_window : list of float
        Two-element list defining the trimming interval relative to the marker,
        in the form ``[from_, to]``.

    Returns
    -------
    subject_name : str
        Subject identifier extracted from the raw directory name.
    task_ratings : pd.DataFrame
        Subject-level task ratings returned by `handle_events`.

    Raises
    ------
    RuntimeError
        If `handle_physio` fails for the subject.
    RuntimeError
        If `handle_events` fails for the subject.
    RuntimeError
        If `handle_eyetracking` fails for the subject.
    RuntimeError
        If `run_pspm_trim_directory` fails for the subject.

    Notes
    -----
    - This function must be defined at top level for Windows multiprocessing.
    - Logging context is managed with `current_subject`.
    - The converted subject directory is constructed from `conv_dir` and the
      extracted subject name.

    Examples
    --------
    >>> subject_name, task_ratings = _process_single_subject(
    ...     raw_dir="/data/raw/sub-01",
    ...     conv_dir="/data/converted",
    ...     dataset_name="austin",
    ...     lab_meta_name="austin_meta",
    ...     trim_window=[-5.0, 20.0],
    ... )
    """

    subj_dir = os.path.basename(raw_dir)
    subject_name = extract_subject_name(subj_dir)
    conv_subj_dir = os.path.join(conv_dir, subject_name)

    token = current_subject.set(subject_name)
    logger = logging.getLogger(__name__)

    try:
        logger.info("Processing subject")

        # physio
        try:
            event_onsets = handle_physio(
                subject_name=subject_name,
                subject_new_dir=conv_subj_dir,
                subject_raw_data_path=raw_dir,
                lab_name=dataset_name,
            )
        except Exception as e:
            raise RuntimeError(f"handle_physio failed for '{subject_name}': {e}") from e

        # events
        try:
            task_ratings, task_events = handle_events(
                raw_path=raw_dir,
                conv_path=conv_subj_dir,
                subject_name=subject_name,
                events_dict=event_onsets,
                events_tpl=EVENTS_JSON_TEMPLATE,
            )
        except Exception as e:
            raise RuntimeError(f"handle_events failed for '{subject_name}': {e}") from e

        # eyetracking
        try:
            handle_eyetracking(
                raw_path=raw_dir,
                conv_path=conv_subj_dir,
                subject_name=subject_name,
                onsets_dict=task_events,
                lab_name=lab_meta_name,
            )
        except Exception as e:
            raise RuntimeError(f"handle_eyetracking failed for '{subject_name}': {e}") from e

        # trim physio
        try:
            run_pspm_trim_directory(
                root_dir=conv_subj_dir,
                from_=trim_window[0],
                to=trim_window[1],
                reference="marker",
                event_time_col="onset",
                overwrite=True,
            )
        except Exception as e:
            raise RuntimeError(f"run_pspm_trim_directory failed for '{subject_name}': {e}") from e

        return subject_name, task_ratings

    finally:
        logger.info(f"Finished: {subject_name}")
        current_subject.reset(token)


def convert_data(
        input_dir: str,
        conv_dir: str,
        clean: bool=False,
        overwrite: bool=False,
        log_file: Optional[str]=None,
        n_workers: int=1,
    ) -> None:
    """
    Convert a full dataset into the standardized output structure.

    This function processes all subject folders within an input dataset,
    generates dataset-level descriptors, aggregates participant and phenotype
    information, converts subject-level data, combines shock ratings across
    subjects, anonymizes converted outputs, and generates QA plots.

    Parameters
    ----------
    input_dir : str
        Path to the input dataset directory containing subject subfolders.
    conv_dir : str
        Path to the root output directory for converted data.
    clean : bool, default=False
        Whether to clear the output directory before processing.
    overwrite : bool, default=False
        Whether to overwrite existing output files where applicable.
    log_file : str, optional
        Path to a log file used when cleaning the output directory.
    n_workers : int, default=1
        Number of worker processes to use for subject processing and QA plot
        generation. Values less than 1 are treated as 1.

    Returns
    -------
    None
        This function writes converted dataset outputs to disk and does not
        return a value.

    Raises
    ------
    Exception
        If lab settings cannot be retrieved for the dataset.
    Exception
        If the output directory cannot be cleaned.
    Exception
        If participant or phenotype information cannot be gathered.
    Exception
        If no subject directories are found in `input_dir`.
    Exception
        If aggregated phenotype processing fails.
    RuntimeError
        If subject-level processing fails.
    Exception
        If aggregated shock ratings cannot be written.
    Exception
        If anonymization fails.
    RuntimeError
        If QA plot generation fails.

    Notes
    -----
    - The dataset name is inferred from the basename of `input_dir`.
    - Subject directories are discovered using `find_sub_dirs`.
    - Aggregated phenotype files are written to ``<conv_dir>/phenotype``.
    - Shock ratings are accumulated across subjects and written after all
      subjects have been processed.
    - When `n_workers > 1`, subject conversion and QA plot generation are
      parallelized with `ProcessPoolExecutor`.
    - Subject IDs are anonymized after conversion and before QA plot generation.

    Examples
    --------
    >>> convert_data(
    ...     input_dir="/data/raw/austin",
    ...     conv_dir="/data/converted/austin",
    ...     clean=True,
    ...     overwrite=True,
    ...     n_workers=4,
    ... )
    """

    log_level = logging.getLogger().level

    dataset_name = os.path.basename(input_dir)
    logger.info(f"Processing files in {input_dir} [lab={dataset_name}]")

    # get lab_settings
    try:
        lab_settings = available_labs.get(dataset_name)
    except Exception as e:
        raise Exception(f"Could not extract lab settings from 'available_labs' in config.py [lab='{dataset_name}']")
    if clean:
        logger.warning(f"Clearing out '{conv_dir}' before processing")
        
        try:
            _ = clean_output_directory(
                conv_dir,
                log_file=log_file
            )
        except Exception as e:
            raise Exception(f"Could not clear {conv_dir} before processing: {e}") from e

    # Create dataset_description.json and readme
    create_dataset_descriptors(
        conv_dir,
        dataset_name,
    )

    # Gather info for all participants
    logger.info("Gathering participant/phenotype info...")
    try:
        (_, df_pheno) = handle_participant_info(
            input_dir=input_dir,
            output_dir=conv_dir,
            lab_name=dataset_name,
            overwrite=overwrite
        )
    except Exception as e:
        raise Exception(f"handle_participant_info failed for '{dataset_name}': {e}") from e    

    # Gather all subject folders - folders starting with prefix "sub"
    logger.info(f"Discovering subjects in {input_dir}")
    subject_dirs = find_sub_dirs(input_dir)

    if len(subject_dirs)==0:
        raise Exception(f"Found 0 subjects in '{input_dir}'")
    
    logger.info(f"Found {len(subject_dirs)} subjects in dataset.")

    # Create a common phenotype folder in the output root
    pheno_dir = os.path.join(
        conv_dir,
        "phenotype"
    )
    os.makedirs(pheno_dir, exist_ok=True)

    # Process aggregated phenotype (questionnaire) data for all subjects at once.
    try:
        handle_pheno(
            df_pheno,
            pheno_dir,
            lab_name=dataset_name,
            overwrite=overwrite
        )
    except Exception as e:
        raise Exception(f"handle_pheno failed for '{dataset_name}': {e}") from e

    # Initialize aggregated shock ratings dictionary.
    aggregated_shock_ratings = {}

    # read window from config, but default to -10, 30 before/after marker
    window = config.get("trim_window", [-10, 30])

    if n_workers is None or n_workers < 1:
        n_workers = 1

    logger.info(f"Processing {len(subject_dirs)} subjects with n_workers={n_workers}")

    if n_workers == 1:
        for raw_dir in subject_dirs:
            subj_dir = os.path.basename(raw_dir)
            subject_name = extract_subject_name(subj_dir)

            subject_name, task_ratings = _process_single_subject(
                raw_dir=raw_dir,
                conv_dir=conv_dir,
                dataset_name=dataset_name,
                lab_meta_name=lab_settings.get("MetaName"),
                trim_window=window,
            )

            accumulate_shock_ratings(
                aggregated_shock_ratings,
                subject_name,
                task_ratings
            )
    else:
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=worker_init,
            initargs=(conv_dir, log_level),
        ) as ex:

            futures = {
                ex.submit(
                    _process_single_subject,
                    raw_dir,
                    conv_dir,
                    dataset_name,
                    lab_settings.get("MetaName"),
                    window,
                ): raw_dir
                for raw_dir in subject_dirs
            }

            for fut in as_completed(futures):
                raw_dir = futures[fut]
                subj_dir = os.path.basename(raw_dir)
                subject_name = extract_subject_name(subj_dir)

                try:
                    finished_subject, task_ratings = fut.result()

                    accumulate_shock_ratings(
                        aggregated_shock_ratings,
                        finished_subject,
                        task_ratings
                    )
                except Exception as e:
                    raise RuntimeError(f"Subject processing failed for '{subject_name}': {e}") from e


    # Write aggregated shock ratings to the common phenotype folder.
    try:
        logger.info(f"Writing aggregated shock ratings")
        _ = write_aggregated_shock_ratings(
            aggregated_shock_ratings,
            pheno_dir,
            language=lab_settings.get("Phenotype").get("Language")
        )
    except Exception as e:
        raise Exception(f"Aggregating rating scores failed: {e}") from e

    # anonymize subject IDs
    try:
        logger.info("Anonymizing...")
        mapper_file, _ = anonymize_converted_data(
            conv_dir,
            dataset_name
        )
        logger.info(f"Mapping old IDs → new IDs saved in: '{mapper_file}'")
    except Exception as e:
        raise Exception(f"Anonymization for {dataset_name} failed: {e}") from e
    
    # generate QA plot
    participant_tsv = os.path.join(conv_dir, "participants.tsv")
    subject_ids = pd.read_csv(
        participant_tsv,
        sep="\t",
        na_values=["n/a"]
    )["participant_id"].tolist()

    qa_dir = os.path.join(conv_dir, "derivatives", "qa")
    os.makedirs(qa_dir, exist_ok=True)

    logger.info(f"Generating QA-plots for {len(subject_ids)} subjects in '{conv_dir}'")

    if n_workers == 1:
        for subject in subject_ids:
            finished_subject, fname = _generate_single_qa_plot(
                dataset_name=dataset_name,
                subject=subject,
                qa_dir=qa_dir,
            )
    else:
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=worker_init,
            initargs=(conv_dir, log_level),
        ) as ex:
            
            futures = {
                ex.submit(
                    _generate_single_qa_plot,
                    dataset_name,
                    subject,
                    qa_dir,
                ): subject
                for subject in subject_ids
            }

            for fut in as_completed(futures):
                subject = futures[fut]
                try:
                    finished_subject, fname = fut.result()
                except Exception as e:
                    raise RuntimeError(f"QA plot generation failed for '{subject}': {e}") from e

    logger.info("Dataset conversion complete.")


def create_dataset_descriptors(
        converted_dataset_root: str,
        lab_name: str,
    ) -> None:
    """
    Create dataset-level descriptor files in the converted dataset directory.

    This function generates standard descriptor files at the root of a converted
    dataset, including a README, dataset description JSON, and a ``.bidsignore``
    file with predefined patterns.

    Parameters
    ----------
    converted_dataset_root : str
        Path to the root directory of the converted dataset.
    lab_name : str
        Name of the lab or dataset used to populate descriptor content.

    Returns
    -------
    None
        This function writes descriptor files to disk and does not return a value.

    Notes
    -----
    - The following files are created:
        - ``dataset_description.json``
        - ``README`` (or ``readme.txt`` depending on implementation)
        - ``.bidsignore``
    - The ``.bidsignore`` file excludes directories such as ``phenotype/``,
      ``physio/``, and ``derivatives/``, as well as log and mapping files.

    Examples
    --------
    >>> create_dataset_descriptors(
    ...     converted_dataset_root="/data/converted/austin",
    ...     lab_name="austin"
    ... )
    """
    
    # readme
    create_readme(converted_dataset_root, lab_name)

    # dataset description
    create_dataset_description(converted_dataset_root, lab_name)

    # bidsignore for phenotype
    create_bidsignore(
        converted_dataset_root,
        patterns=[
            "phenotype/",
            "log.log",
            "physio/",
            "mapper.json",
            "derivatives/"
        ]
    )


def handle_participant_info(
        input_dir: Optional[str]=None,
        output_dir: Optional[str]=None,
        lab_name: Optional[str]=None,
        write_file: bool=True,
        **kwargs
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load participant and phenotype data and optionally write participant metadata.

    This function retrieves participant-level information and phenotype data
    using `gather_all_participant_pheno`. Optionally, it writes a
    ``participants.json`` file at the dataset root using the extracted
    participant information.

    Parameters
    ----------
    input_dir : str, optional
        Path to the input dataset directory containing raw data.
    output_dir : str, optional
        Path to the output directory where participant metadata will be written.
    lab_name : str, optional
        Name of the lab used to determine parsing behavior.
    write_file : bool, default=True
        Whether to write the ``participants.json`` file.
    **kwargs
        Additional keyword arguments passed to
        `create_root_level_participant_info`.

    Returns
    -------
    df_info : pd.DataFrame
        Dataframe containing participant-level metadata.
    df_pheno : pd.DataFrame
        Dataframe containing phenotype data extracted from questionnaires.

    Notes
    -----
    - Participant and phenotype data are loaded via
      `gather_all_participant_pheno`.
    - If `write_file=True`, a ``participants.json`` file is created in
      `output_dir`.

    Examples
    --------
    >>> df_info, df_pheno = handle_participant_info(
    ...     input_dir="/data/raw",
    ...     output_dir="/data/converted",
    ...     lab_name="austin"
    ... )
    """
    
    # read phenotype data and parse participants info
    df_info, df_pheno = gather_all_participant_pheno(
        input_dir,
        lab_name
    )

    # write participants.json
    if write_file:
        create_root_level_participant_info(
            df_info,
            output_dir,
            PARTICIPANTS_JSON_TEMPLATE,
            **kwargs
        )

    return (df_info, df_pheno)


def create_root_level_participant_info(
        df_info: pd.DataFrame,
        output_dir: str,
        json_tpl: dict,
        overwrite: bool=False
    ) -> None:
    """
    Create root-level participant metadata files (TSV and JSON).

    This function generates the ``participants.json`` and ``participants.tsv``
    files in the dataset root directory. Files are only written if they do not
    already exist or if `overwrite=True`.

    Parameters
    ----------
    df_info : pd.DataFrame
        Dataframe containing participant-level metadata.
    output_dir : str
        Path to the dataset root directory where files will be written.
    json_tpl : dict
        Template dictionary used to populate the ``participants.json`` file.
    overwrite : bool, default=False
        Whether to overwrite existing ``participants.json`` and
        ``participants.tsv`` files.

    Returns
    -------
    None
        This function writes files to disk and does not return a value.

    Notes
    -----
    - Missing values in the TSV file are written as ``"n/a"``.
    - The JSON file is created using `save_json`.
    - Existing files are preserved unless `overwrite=True`.

    Examples
    --------
    >>> create_root_level_participant_info(
    ...     df_info=df_participants,
    ...     output_dir="/data/converted",
    ...     json_tpl=PARTICIPANTS_JSON_TEMPLATE,
    ...     overwrite=True
    ... )
    """
    
    # Create the participants.json file
    json_path = os.path.join(output_dir, "participants.json")
    if not os.path.exists(json_path) or overwrite:
        save_json(
            json_path,
            json_tpl
        )

    # Create the participants.tsv file
    tsv_path = os.path.join(output_dir, "participants.tsv")
    if not os.path.exists(tsv_path) or overwrite:
        df_info.fillna("n/a").infer_objects(copy=False).to_csv(
            tsv_path,
            sep="\t",
            index=False
        )

    logger.info(
        f"Files created: {json_path}, {tsv_path} at {output_dir}"
    )

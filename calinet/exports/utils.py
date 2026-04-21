# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import shutil
import logging
import numpy as np
import pandas as pd
from copy import deepcopy

from pathlib import Path
from datetime import date
from typing import Optional, Iterable

import calinet.core.io as cio
from calinet.core.metadata import config
from calinet.utils import (
    _get_units,
    _extract_task,
    _extract_recording,
)
from calinet.templates.common import PARTICIPANT_INFO_SPEC

logger = logging.getLogger(__name__)


def normalize_modalities(
        modalities: Optional[Iterable[str]]
    ) -> Optional[set[str]]:
    """
    Normalize modality identifiers into a canonical lowercase set.

    Parameters
    ----------
    modalities : Optional[Iterable[str]]
        Iterable of modality names.

    Returns
    -------
    Optional[set[str]]
        Set of normalized modality names, or None if input is None.

    Notes
    -----
    - Empty or whitespace-only entries are ignored.
    - All values are converted to lowercase and stripped.
    """
    if modalities is None:
        return None
    return {m.strip().lower() for m in modalities if str(m).strip()}


def should_keep_file(
        fname: str,
        modalities: Optional[set[str]]
    ) -> bool:
    """
    Determine whether a file should be retained based on modality filtering.

    Parameters
    ----------
    fname : str
        Filename to evaluate.
    modalities : Optional[set[str]]
        Allowed modalities. If None, all files are retained.

    Returns
    -------
    bool
        True if the file should be kept, False otherwise.

    Notes
    -----
    - Files without identifiable modality are always retained.
    """
    if modalities is None:
        return True

    modality = _extract_recording(fname)
    if modality is None:
        return True

    return modality in modalities


def create_derivative_dataset_description(
        bids_root: str,
        derivative_name: str,
        source_dataset: str=None,
        dataset_type: str="derivative",
        generated_by_name: str=None,
        generated_by_version: str="1.0.0",
        generated_by_description: str=None,
        code_url: str="https://github.com/gjheij/calinet-main",
        source_datasets: list[dict]=None,
        extra_fields: dict=None
    ) -> None:
    """
    Create a reusable ``dataset_description.json`` for CALINET-derived datasets.

    This function is intended for derivative outputs generated from an existing
    BIDS/CALINET dataset, such as:

    - blinded datasets
    - Autonomate exports
    - EzySCR exports

    It writes a ``dataset_description.json`` file into ``bids_root`` and
    provides a consistent metadata scaffold that can be customized per
    derivative type through the function arguments.

    The function is deliberately more generic than a lab-specific raw-dataset
    description. It emphasizes provenance, derivative intent, and generation
    metadata so the same helper can be reused across multiple export or
    conversion workflows.

    Parameters
    ----------
    bids_root : str
        Path to the root folder of the derivative dataset.
    derivative_name : str
        Human-readable name of the derivative dataset. This becomes the value
        of the ``Name`` field in ``dataset_description.json``.
    source_dataset : str, default=None
        Optional human-readable name of the upstream source dataset. When
        provided, it is incorporated into the ``Description`` field and may
        also be stored in provenance metadata.
    lab_name : str, default=None
        Optional CALINET lab identifier. When provided and found in
        ``available_labs``, lab-specific metadata such as the consortium-facing
        lab name, authors, and available modalities are incorporated where
        appropriate.
    dataset_type : str, default="derivative"
        BIDS dataset type. For blinding, Autonomate export, and EzySCR export,
        this should generally remain ``"derivative"``.
    generated_by_name : str, default=None
        Name of the software or workflow that generated the derivative. If not
        provided, a generic CALINET-derived label is used.
    generated_by_version : str, default="1.0.0"
        Version string recorded in the ``GeneratedBy`` block.
    generated_by_description : str, default=None
        Free-text description of the derivative-generation workflow. If not
        provided, a generic description is generated based on
        ``derivative_name``.
    code_url : str, default="https://github.com/gjheij/calinet-main"
        Code repository URL for the generating workflow.
    source_datasets : list[dict], default=None
        Optional list of BIDS-style source dataset provenance records. If
        provided, it is written as ``SourceDatasets``. Each entry should
        already be structured as a JSON-serializable dictionary.
    extra_fields : dict, default=None
        Optional mapping of additional metadata fields to merge into the final
        dataset description. This can be used to add tool-specific fields such
        as event-coding notes, export format details, or processing-specific
        provenance.

    Returns
    -------
    None
        The function writes ``dataset_description.json`` to disk and returns
        nothing.

    Notes
    -----
    The generated metadata is designed to be broadly suitable for derivative
    exports that preserve or transform BIDS-organized CALINET data.

    Typical use cases include:

    Blinding
        Use a derivative name such as ``"CALINET Blinded Dataset"`` and provide
        a description explaining that identifying stimulus information and
        sensitive metadata have been removed or recoded.

    Autonomate export
        Use a derivative name such as ``"CALINET Autonomate Export"`` and add
        extra fields describing the two-column ASCII event-coding format.

    EzySCR export
        Use a derivative name such as ``"CALINET EzySCR Export"`` and add
        extra fields describing the MATLAB export format and event-vector
        coding scheme.

    Lab metadata
        If ``lab_name`` is found in ``available_labs``, this function attempts
        to populate fields such as:

        - ``Authors``
        - ``InstitutionName``
        - ``Modalities``

        If ``lab_name`` is missing or unknown, those fields are omitted unless
        explicitly provided through ``extra_fields``.

    Merge behavior
        ``extra_fields`` is applied last and therefore overrides any default
        fields generated by this function. This makes it easy to share a common
        base description while tailoring individual derivative types.

    See Also
    --------
    calinet.core.io.save_json
        Function used to write the JSON sidecar to disk.

    Examples
    --------
    Create a blinded derivative description

    >>> create_derivative_dataset_description(
    ...     bids_root="derivatives/blinded/austin",
    ...     derivative_name="Austin CALINET Blinded Dataset",
    ...     source_dataset="Austin CALINET Fear-Conditioning Dataset",
    ...     lab_name="austin",
    ...     generated_by_name="CALINET Blinder",
    ...     generated_by_description=(
    ...         "Creates a blinded derivative by recoding event labels and "
    ...         "removing identifying metadata fields."
    ...     ),
    ... )

    Create an Autonomate export description

    >>> create_derivative_dataset_description(
    ...     bids_root="derivatives/autonomate",
    ...     derivative_name="CALINET Autonomate Export",
    ...     generated_by_name="CALINET Autonomate Exporter",
    ...     generated_by_description=(
    ...         "Exports electrodermal recordings to two-column ASCII text "
    ...         "files for Autonomate."
    ...     ),
    ...     extra_fields={
    ...         "ExportFormat": {
    ...             "Name": "Autonomate ASCII",
    ...             "Columns": ["electrodermal_data", "event_code"],
    ...         }
    ...     },
    ... )

    Create an EzySCR export description

    >>> create_derivative_dataset_description(
    ...     bids_root="derivatives/ezyscr",
    ...     derivative_name="CALINET EzySCR Export",
    ...     generated_by_name="CALINET EzySCR Exporter",
    ...     generated_by_description=(
    ...         "Exports electrodermal recordings to MATLAB format for EzySCR."
    ...     ),
    ...     extra_fields={
    ...         "ExportFormat": {
    ...             "Name": "EzySCR MAT",
    ...             "Columns": ["Skin conductance", "Event"],
    ...         }
    ...     },
    ... )

    Raises
    ------
    OSError
        If the output directory cannot be created or the JSON file cannot be
        written.
    """
    bids_root = Path(bids_root)
    bids_root.mkdir(parents=True, exist_ok=True)

    generated_by_name = generated_by_name or f"{derivative_name} Generator"
    generated_by_description = generated_by_description or (
        f"Generated derivative dataset for '{derivative_name}' within the "
        f"CALINET workflow."
    )

    dataset_description = {
        "Name": derivative_name,
        "BIDSVersion": str(config.get("BIDS_Version")),
        "DatasetType": dataset_type,
        "GeneratedBy": [
            {
                "Name": generated_by_name,
                "Version": generated_by_version,
                "CodeURL": code_url,
                "Description": generated_by_description,
            }
        ],
        "GeneratedDate": str(date.today()),
        "License": "CC-BY-4.0",
        "Consortium": "CALINET2",
        "HowToAcknowledge": (
            "Please cite the original source dataset, the CALINET/CALINET2 "
            "consortium work, and the derivative-generation workflow used to "
            "produce these files."
        ),
    }

    if source_dataset is not None:
        dataset_description["Description"] = (
            f"Derivative dataset generated from '{source_dataset}'. "
            f"This export corresponds to the derivative type '{derivative_name}'."
        )
    else:
        dataset_description["Description"] = (
            f"Derivative dataset generated within the CALINET workflow. "
            f"This export corresponds to the derivative type '{derivative_name}'."
        )

    if source_datasets is not None:
        dataset_description["SourceDatasets"] = source_datasets

    if extra_fields is not None:
        dataset_description.update(extra_fields)

    output_file = bids_root / "dataset_description.json"
    cio.save_json(output_file, dataset_description)
    logger.info(f"Created: {output_file}")


def maybe_copy_participant_files(
        subjects_tsv: Optional[str],
        output_dir: str
    ) -> None:
    """
    Copy participant selection files into an output dataset directory.

    This helper copies a subject-selection TSV file into ``output_dir`` as
    ``participants.tsv`` and, if present alongside the TSV, also copies the
    matching JSON sidecar as ``participants.json``. Copying is skipped when
    no TSV path is provided or when the destination directory already matches
    the source directory.

    Parameters
    ----------
    subjects_tsv : Optional[str]
        Path to the source TSV file containing participant or subject
        selection information. If None, no files are copied.
    output_dir : str
        Destination directory where participant files should be written.

    Returns
    -------
    None

    Notes
    -----
    - If ``subjects_tsv`` is None, the function returns immediately.
    - If the parent directory of ``subjects_tsv`` is the same as
      ``output_dir``, copying is skipped.
    - The TSV is copied and renamed to ``participants.tsv``.
    - If a JSON file with the same stem exists next to ``subjects_tsv``,
      it is copied and renamed to ``participants.json``.
    - Informational messages are logged for copied and skipped files.

    Raises
    ------
    FileNotFoundError
        If ``subjects_tsv`` is provided but does not exist.
    OSError
        If the destination directory cannot be created or a file copy fails.
    """

    if subjects_tsv is None:
        return

    src_tsv = Path(subjects_tsv).resolve()
    src_dir = src_tsv.parent
    dst_dir = Path(output_dir).resolve()

    if src_dir == dst_dir:
        logger.info(
            "Output directory matches subjects TSV parent; not copying participant files."
        )
        return

    if not src_tsv.exists():
        raise FileNotFoundError(f"Subjects TSV does not exist: '{src_tsv}'")

    os.makedirs(dst_dir, exist_ok=True)

    dst_tsv = dst_dir / "participants.tsv"
    shutil.copy2(src_tsv, dst_tsv)
    logger.info(f"Copied participants TSV: '{src_tsv}' -> '{dst_tsv}'")

    src_json = src_tsv.with_suffix(".json")
    dst_json = dst_dir / "participants.json"
    if src_json.exists():
        shutil.copy2(src_json, dst_json)
        logger.info(f"Copied participants JSON: '{src_json}' -> '{dst_json}'")
    else:
        logger.info(
            f"No participants JSON found next to subjects TSV: '{src_json}'; generating it."
        )

        df = pd.read_csv(dst_tsv, delimiter="\t")
        cio.save_json(dst_json, build_participants_sidecar(df))


def is_multisite_export_mode(
        input_dir: str,
        subjects_tsv: Optional[str]
    ) -> bool:
    """
    Determine whether the dataset should be treated as a multi-site export.

    This function inspects the directory structure of the input dataset and
    the presence of a subjects export file to decide whether the dataset
    represents a multi-site container (i.e., sites containing subjects)
    rather than a flat subject-level dataset.

    Parameters
    ----------
    input_dir : str
        Root directory of the dataset.
    subjects_tsv : Optional[str]
        Path to a TSV file specifying selected subjects. If ``None``, multi-site
        mode is disabled.

    Returns
    -------
    bool
        True if the dataset appears to be a multi-site container and a
        subjects TSV is provided, otherwise False.

    Notes
    -----
    - Multi-site mode is inferred when no ``sub-*`` directories are found
      directly under ``input_dir``.
    """
    if subjects_tsv is None:
        return False

    root = Path(input_dir)
    direct_subjects = any(
        p.is_dir() and p.name.startswith("sub-")
        for p in root.iterdir()
    )
    return not direct_subjects


def load_subjects_from_export(
        export_tsv: str
    ) -> list[str]:
    """
    Load and normalize subject identifiers from an export TSV file.

    This function reads a TSV file generated by an external export or
    randomization process and extracts subject identifiers from known column
    names. Identifiers are normalized to the ``sub-*`` format.

    Parameters
    ----------
    export_tsv : str
        Path to the TSV file containing subject selections.

    Returns
    -------
    list of str
        Sorted list of unique subject identifiers in normalized format.

    Raises
    ------
    ValueError
        If no valid subject column is found or no valid subjects are extracted.

    Notes
    -----
    - Recognized column names include:
      ``participant_id``, ``subject_id``, ``subject``, ``sub``.
    - Empty or malformed entries are ignored.
    """

    if not os.path.isfile(export_tsv):
        raise TypeError(f"subject_tsv must be a file!")
    elif not os.path.exists(export_tsv):
        raise FileNotFoundError(f"Specified file '{export_tsv}' does not exist")
    else:
        pass
    
    df = pd.read_csv(export_tsv, sep="\t")

    for col in ["participant_id", "subject_id", "subject", "sub"]:
        if col in df.columns:
            subjects = df[col].dropna().astype(str).tolist()
            break
    else:
        raise ValueError(
            "Could not find a subject ID column in export TSV. "
            "Expected one of: participant_id, subject_id, subject, sub"
        )

    normalized = []
    for subject in subjects:
        subject = subject.strip()
        if not subject:
            continue
        if not subject.startswith("sub-"):
            subject = f"sub-{subject}"
        normalized.append(subject)

    out = sorted(set(normalized))
    if not out:
        raise ValueError(f"No valid subjects found in export TSV: '{export_tsv}'")

    return out


def discover_subjects(
        input_dir: str
    ) -> list[dict]:
    """
    Discover subject directories within a dataset.

    This function inspects the input directory to identify subject-level
    directories in one of two supported layouts:

    1. Flat layout:
       ``input_dir/sub-*``
    2. Multi-site layout:
       ``input_dir/<site>/sub-*``

    It returns a list of metadata dictionaries describing each discovered
    subject.

    Parameters
    ----------
    input_dir : str
        Root directory of the dataset.

    Returns
    -------
    list of dict
        Each dictionary contains:
        - ``site`` : str or None
            Site identifier if present, otherwise None.
        - ``subject_id`` : str
            Subject directory name (e.g., "sub-01").
        - ``subject_path`` : str
            Absolute path to the subject directory.
        - ``dataset_root`` : str
            Root directory of the dataset or site.

    Notes
    -----
    - The function prefers flat layout if both structures are present.
    - Returned records are sorted deterministically.
    """
    root = Path(input_dir)

    records = []

    # Case 1: site-level dataset
    direct_subjects = [
        p for p in sorted(root.iterdir())
        if p.is_dir() and p.name.startswith("sub-")
    ]
    if direct_subjects:
        for sub_dir in direct_subjects:
            records.append(
                {
                    "site": None,
                    "subject_id": sub_dir.name,
                    "subject_path": str(sub_dir),
                    "dataset_root": str(root),
                }
            )
        return records

    # Case 2: top-level container with site folders
    for site_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for sub_dir in sorted(site_dir.iterdir()):
            if sub_dir.is_dir() and sub_dir.name.startswith("sub-"):
                records.append(
                    {
                        "site": site_dir.name,
                        "subject_id": sub_dir.name,
                        "subject_path": str(sub_dir),
                        "dataset_root": str(site_dir),
                    }
                )

    return records


def filter_subjects(
        subjects: list[str],
        include_n: Optional[int]
    ) -> list[str]:
    """
    Restrict a subject list to at most the first ``N`` entries.

    This helper is used to support partial processing of a dataset by returning
    either the full input list or a prefix of that list, preserving the
    original order.

    Parameters
    ----------
    subjects : list[str]
        Ordered list of subject identifiers or subject directory names.
    include_n : Optional[int]
        Number of subjects to retain from the beginning of ``subjects``. If
        None, the full list is returned unchanged.

    Returns
    -------
    list[str]
        The full input list when ``include_n`` is None, otherwise the first
        ``include_n`` entries.

    Notes
    -----
    - Ordering is preserved exactly as given.
    - Standard Python slicing semantics apply, so values larger than the list
      length simply return the whole list.
    - Negative values are not explicitly validated and therefore follow Python
      slice behavior.
    """
    if include_n is None:
        return subjects
    return subjects[:include_n]


def find_physio_dir(
        subject_dir: str
    ) -> Optional[str]:
    """
    Locate the physiology directory for a subject.

    This function supports two expected directory layouts beneath a subject
    directory:

    1. ``sub-XX/physio``
    2. ``sub-XX/ses-01/physio``

    The direct ``physio`` directory is preferred when both layouts exist.

    Parameters
    ----------
    subject_dir : str
        Path to the root directory for a single subject.

    Returns
    -------
    Optional[str]
        Path to the discovered physiology directory, or None if no supported
        physiology directory is found.

    Notes
    -----
    - Only the exact ``ses-01`` session path is checked for session-based
      layouts.
    - This function does not recurse beyond the supported patterns.
    - The returned value is a string path suitable for downstream file-system
      operations.
    """
    direct = os.path.join(subject_dir, "physio")
    ses01 = os.path.join(subject_dir, "ses-01", "physio")

    if os.path.isdir(direct):
        return direct
    if os.path.isdir(ses01):
        return ses01
    return None


def read_table(
        path: str,
        **kwargs
    ) -> pd.DataFrame:
    """
    Read a physiology-related tabular file into a dataframe.

    This helper transparently supports both plain-text TSV files and gzipped
    TSV files. Gzipped inputs are delegated to
    :func:`calinet.core.io.read_physio_tsv_headerless`, while non-gzipped files
    are read with :func:`pandas.read_csv` using tab separation.

    Parameters
    ----------
    path : str
        Path to the input table. Supported extensions include ``.tsv`` and
        ``.tsv.gz``.
    **kwargs
        Additional keyword arguments forwarded to :func:`pandas.read_csv` when
        reading non-gzipped files.

    Returns
    -------
    pandas.DataFrame
        Parsed tabular data.

    Notes
    -----
    - Files ending in ``".gz"`` are treated as compressed physiology tables and
      are read with the CALINET I/O helper.
    - Non-gzipped files are assumed to be tab-separated.
    - Keyword arguments are ignored for compressed files because those are read
      through the specialized helper instead of pandas directly.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    pandas.errors.ParserError
        If the file cannot be parsed as tabular data.
    OSError
        If the file cannot be opened.
    """
    if path.endswith(".gz"):
        return cio.read_physio_tsv_headerless(path)
    return pd.read_csv(path, sep="\t", **kwargs)


def find_matching_files(
        physio_dir: str,
        modality: str,
        task_name: Optional[str]=None
    ) -> list[dict]:
    """
    Build per-task file bundles for physiology conversion.

    This function inspects a physiology directory, identifies event files, and
    attempts to pair each events TSV with the corresponding physiology signal
    data file and physiology JSON sidecar for a given recording modality. Each
    successfully matched task is represented as a dictionary describing the
    bundle of files needed for conversion.

    Parameters
    ----------
    physio_dir : str
        Path to a physiology directory containing task-specific events files,
        physiology signal files, and physiology JSON sidecars.
    modality : str
        Recording modality to match, such as ``"scr"``.
    task_name : str
        Task name to export, such as ``"acquisition"``. If None, all tasks ['acquisition', 'extinction'] will be used.

    Returns
    -------
    list[dict]
        A list of dictionaries, one per matched task. Each dictionary contains
        the keys:

        ``"task"``
            Extracted task label.
        ``"events_tsv"``
            Full path to the matching events TSV.
        ``"physio_tsv"``
            Full path to the matching physiology TSV or TSV.GZ file.
        ``"physio_json"``
            Full path to the matching physiology JSON sidecar.

    Notes
    -----
    - Task matching relies on ``utils._extract_task``.
    - Modality matching relies on ``utils._extract_recording``.
    - If multiple candidate physiology files exist for a task, the first
      lexicographically sorted candidate is chosen.
    - Missing signal files or JSON sidecars are logged as warnings and cause
      that task bundle to be skipped.
    - Events files for which no task can be extracted are also skipped.

    Raises
    ------
    FileNotFoundError
        If ``physio_dir`` does not exist.
    NotADirectoryError
        If ``physio_dir`` is not a directory.
    PermissionError
        If directory contents cannot be listed.
    """
    files = os.listdir(physio_dir)
    bundles = []

    event_files = [
        f for f in files
        if "events" in f and f.endswith(".tsv")
    ]

    for event_fname in sorted(event_files):
        task = _extract_task(event_fname)
        if task is None:
            logger.warning(f"Could not extract task from events file '{event_fname}'")
            continue
        
        if task_name is not None:
            if task != task_name:
                continue
            
        physio_data_candidates = [
            f for f in files
            if _extract_task(f) == task
            and _extract_recording(f) == modality
            and (
                f.endswith("_physio.tsv")
                or f.endswith("_physio.tsv.gz")
            )
        ]

        physio_json_candidates = [
            f for f in files
            if _extract_task(f) == task
            and _extract_recording(f) == modality
            and f.endswith("_physio.json")
        ]

        if not physio_data_candidates:
            logger.warning(
                f"No physio data file found for task='{task}', modality='{modality}' in '{physio_dir}'"
            )
            continue

        if not physio_json_candidates:
            logger.warning(
                f"No physio JSON found for task='{task}', modality='{modality}' in '{physio_dir}'"
            )
            continue

        bundles.append(
            {
                "task": task,
                "events_tsv": os.path.join(physio_dir, event_fname),
                "physio_tsv": os.path.join(physio_dir, sorted(physio_data_candidates)[0]),
                "physio_json": os.path.join(physio_dir, sorted(physio_json_candidates)[0]),
            }
        )

    return bundles


def load_sampling_info(
        physio_json_path: str,
        modality: str='scr'
    ) -> tuple[float, str]:
    """
    Load sampling frequency and signal units from a physiology JSON sidecar.

    This function reads metadata from a physiology JSON file, validates that
    required sampling information is present, and derives the units for the
    requested modality.

    Parameters
    ----------
    physio_json_path : str
        Path to the physiology JSON sidecar file.
    modality : str='scr'
        Recording modality whose units should be extracted from the metadata.

    Returns
    -------
    tuple[float, str]
        Two-element tuple containing:

        - sampling frequency in Hz as ``float``
        - units string for the requested modality

    Raises
    ------
    ValueError
        If the JSON metadata does not contain ``"SamplingFrequency"``.
    ValueError
        If the JSON metadata does not contain a non-empty ``"Columns"`` entry.
    OSError
        If the file cannot be read.
    KeyError
        If modality-specific unit lookup performed by downstream utilities
        fails unexpectedly.

    Notes
    -----
    - Metadata is loaded using :func:`calinet.core.io.load_json`.
    - Unit extraction is delegated to ``utils._get_units``.
    - Sampling frequency is coerced to ``float`` before being returned.
    """
    
    meta = cio.load_json(physio_json_path)

    sampling_freq = meta.get("SamplingFrequency")
    if sampling_freq is None:
        raise ValueError(f"No 'SamplingFrequency' in '{physio_json_path}'")

    columns = meta.get("Columns")
    if not columns:
        raise ValueError(f"No 'Columns' entry in '{physio_json_path}'")

    units = _get_units(meta, modality)

    return float(sampling_freq), units


def find_signal_column(
        df: pd.DataFrame,
        modality: str
    ) -> str:
    """
    Identify the dataframe column containing the requested physiology signal.

    The lookup first prefers an exact column-name match to ``modality``. If no
    exact match is found, the function performs a case-insensitive search over
    dataframe column names. This makes the function tolerant of capitalization
    differences while still favoring explicit exact matches.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing physiology signal columns.
    modality : str
        Name of the desired modality column, such as ``"scr"``.

    Returns
    -------
    str
        Name of the matching dataframe column.

    Raises
    ------
    ValueError
        If no exact or case-insensitive match can be found for ``modality``.

    Notes
    -----
    - Exact matching is attempted before case-insensitive matching.
    - In case-insensitive mode, later duplicate lowercase column names would
      overwrite earlier ones in the temporary lookup map.
    - The returned value is the original dataframe column label, preserving its
      original capitalization.
    """
    if modality in df.columns:
        return modality

    lower_map = {str(c).lower(): c for c in df.columns}
    if modality.lower() in lower_map:
        return lower_map[modality.lower()]

    raise ValueError(
        f"Could not find modality column '{modality}' in physio data columns: {list(df.columns)}"
    )


def build_event_column(
        events_df: pd.DataFrame,
        n_samples: int,
        sampling_freq: float,
        default_value: int=5,
        cs_value: int=0
    ) -> np.ndarray:
    """
    Construct an EzySCR-compatible event vector from an events table.

    This function creates a one-dimensional integer array aligned to the
    physiology sample stream. All samples are initialized to ``default_value``,
    and onsets of conditioned-stimulus-like events are marked with
    ``cs_value``. Events are identified from rows whose ``event_type`` begins
    with ``"CS"``.

    Parameters
    ----------
    events_df : pandas.DataFrame
        Events table containing at least ``event_type`` and ``onset`` columns.
        If ``duration`` is absent, a default duration of 1.0 seconds is
        assumed.
    n_samples : int
        Total number of samples in the physiology recording.
    sampling_freq : float
        Sampling frequency of the physiology data in Hz.
    default_value : int = 5
        Baseline code assigned to all samples before event mapping.
    cs_value : int = 0
        Code assigned at detected CS event onset samples.

    Returns
    -------
    numpy.ndarray
        One-dimensional ``int32`` array of length ``n_samples`` containing the
        generated event codes.

    Notes
    -----
    - Only the onset sample of each qualifying CS event is explicitly marked.
      Although event end indices are computed internally, the implementation
      does not fill the full event duration window.
    - If required columns are missing, the returned vector remains entirely at
      ``default_value`` and a warning is logged.
    - ``onset`` and ``duration`` are coerced to numeric values with invalid
      entries converted to missing values.
    - Events with missing onset values are skipped.
    - Events whose onset falls outside the valid sample range are ignored.

    Raises
    ------
    ValueError
        If invalid upstream inputs lead to array construction issues.
    """
    event_col = np.full(n_samples, default_value, dtype=np.int32)

    if "event_type" not in events_df.columns or "onset" not in events_df.columns:
        logger.warning("events.tsv missing required columns; leaving Event column at default value")
        return event_col

    if "duration" not in events_df.columns:
        events_df = events_df.copy()
        events_df["duration"] = 1.0

    events_df = events_df.copy()
    events_df["onset"] = pd.to_numeric(events_df["onset"], errors="coerce")
    events_df["duration"] = pd.to_numeric(events_df["duration"], errors="coerce")
    event_type = events_df["event_type"].astype(str).str.strip()

    cs_df = events_df.loc[event_type.str.startswith("CS", na=False)]

    logger.info(f"Total CS-like events: {len(cs_df)}")

    valid_count = 0

    for _, row in cs_df.iterrows():
        onset = row["onset"]
        duration = row["duration"]

        if pd.isna(onset):
            continue

        start = int(round(float(onset) * sampling_freq))
        end = int((float(onset)+duration)*sampling_freq)

        if 0 <= start < n_samples:
            event_col[start:end] = cs_value
            valid_count += 1

    logger.info(f"Valid mapped events: {valid_count}")
    
    return event_col


def build_participants_sidecar(
        df: pd.DataFrame
    ) -> dict:
    """
    Build a BIDS-style ``participants.json`` sidecar from the participant table.

    This function starts from the CALINET participant metadata template and
    constructs a sidecar dictionary containing metadata entries relevant to the
    columns present in ``df``. Template-defined metadata blocks are preserved
    only when the corresponding dataframe columns exist, with the exception of
    top-level metadata blocks that should always be retained.

    For dataframe columns that are not represented in the template, the
    function adds a fallback metadata entry containing a simple default
    description.

    Parameters
    ----------
    df : pandas.DataFrame
        Participant-level table that will be written as ``participants.tsv``.
        Column names are used to determine which metadata fields should be
        retained from the template and which fallback entries should be added.

    Returns
    -------
    dict
        A dictionary suitable for serialization as a BIDS-style
        ``participants.json`` sidecar. Keys correspond to dataframe columns
        and preserved metadata blocks, and values are metadata dictionaries.

    Notes
    -----
    - Metadata is derived from ``PARTICIPANT_INFO_SPEC``.
    - The ``MeasurementToolMetadata`` block is always preserved when present
      in the template.
    - Any dataframe column not covered by the template is given a fallback
      entry of the form ``{"Description": "<column> field"}``.
    - This function does not validate BIDS compliance beyond constructing a
      metadata mapping aligned to the dataframe columns.

    See Also
    --------
    write_participants_files : Write participant TSV and JSON files together.
    """
    spec = deepcopy(PARTICIPANT_INFO_SPEC)

    keep = {"MeasurementToolMetadata"}
    keep.update([col for col in df.columns if col in spec])

    sidecar = {}
    for key, value in spec.items():
        if key in keep:
            sidecar[key] = value

    # Add fallback descriptions for columns not covered by template
    for col in df.columns:
        if col not in sidecar:
            sidecar[col] = {"Description": f"{col} field"}

    return sidecar


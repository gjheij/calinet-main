# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import os
import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.io import savemat, loadmat

import calinet.core.io as cio
from calinet.config import config

from calinet.exports.utils import (
    read_table,
    find_physio_dir,
    filter_subjects,
    discover_subjects,
    build_event_column,
    find_signal_column,
    load_sampling_info,
    find_matching_files,
    is_multisite_export_mode,
    load_subjects_from_export,
    maybe_copy_participant_files,
    create_derivative_dataset_description
)

logger = logging.getLogger(__name__)


def save_ezyscr_mat(
        out_path,
        scr_signal,
        event_signal,
        sampling_freq,
        scr_units="uS"
    ) -> None:
    """
    Save physiology and event data in EzySCR-compatible MATLAB format.

    This function packages a skin conductance signal and an aligned event
    signal into a two-column MATLAB ``.mat`` file containing the metadata
    fields expected by downstream EzySCR workflows.

    Parameters
    ----------
    out_path : Any
        Output path for the MAT file. The value is passed directly to
        :func:`scipy.io.savemat` and therefore may be a path-like object or a
        string.
    scr_signal : Any
        One-dimensional or array-like skin conductance signal. The data are
        coerced to a flattened ``float64`` NumPy array.
    event_signal : Any
        One-dimensional or array-like event code signal aligned to
        ``scr_signal``. The data are coerced to a flattened ``float64`` NumPy
        array.
    sampling_freq : Any
        Sampling frequency in Hz. Used to derive the inter-sample interval in
        milliseconds.
    scr_units : str="uS"
        Units label to store for the skin conductance signal.

    Returns
    -------
    None
        The MAT file is written to disk as a side effect.

    Raises
    ------
    ValueError
        If ``scr_signal`` and ``event_signal`` do not have the same length.
    OSError
        If the MAT file cannot be written.

    Notes
    -----
    The saved MAT structure contains the following keys:

    ``data``
        Two-column array with skin conductance in the first column and event
        codes in the second.
    ``labels``
        String array containing ``"Skin conductance"`` and ``"Event"``.
    ``units``
        String array containing the SCR units and ``"code"``.
    ``isi``
        Inter-sample interval in milliseconds, stored as a 1x1 float array.
    ``isi_units``
        String array containing ``"ms"``.
    ``start_sample``
        Unsigned integer array initialized to zero.

    - Compression is disabled when writing the MAT file.
    - Input arrays are flattened before saving.
    """

    scr_signal = np.asarray(scr_signal, dtype=np.float64).reshape(-1)
    event_signal = np.asarray(event_signal, dtype=np.float64).reshape(-1)

    if scr_signal.shape[0] != event_signal.shape[0]:
        raise ValueError("scr_signal and event_signal must have the same length")

    data = np.column_stack([scr_signal, event_signal]).astype(np.float64)

    mat_dict = {
        "data": data,
        "labels": np.array(["Skin conductance", "Event"], dtype="U"),
        "units": np.array([scr_units, "code"], dtype="U"),
        "isi": np.array([[1000.0 / sampling_freq]], dtype=np.float64),
        "isi_units": np.array(["ms"], dtype="U"),
        "start_sample": np.array([[0]], dtype=np.uint8),
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    savemat(out_path, mat_dict, do_compression=False)


def _convert_bundle_to_mat(
        subject: str,
        bundle: dict,
        output_dir: str,
        modality: str,
        overwrite: bool = False
    ) -> str:
    """
    Convert one task-specific physiology bundle into MAT and JSON outputs.

    This function reads the events table, physiology signal file, and
    physiology JSON sidecar for a single subject-task bundle; constructs an
    EzySCR-compatible event vector; writes a MATLAB file; reopens that MAT file
    to verify and summarize key metadata; and finally writes a JSON sidecar
    describing the converted output.

    Parameters
    ----------
    subject : str
        Subject identifier, typically in ``"sub-XX"`` form.
    bundle : dict
        Dictionary describing one matched task bundle. Expected keys are
        ``"task"``, ``"events_tsv"``, ``"physio_tsv"``, and ``"physio_json"``.
    output_dir : str
        Directory where the converted MAT and JSON files will be written.
    modality : str
        Recording modality to extract from the physiology table, such as
        ``"scr"``.
    overwrite : bool = False
        If False and the target MAT file already exists, conversion is skipped
        and the existing output path is returned.

    Returns
    -------
    str
        Path to the written or reused MAT file.

    Raises
    ------
    KeyError
        If required keys are missing from ``bundle``.
    ValueError
        If sampling metadata or the requested signal column cannot be resolved.
    FileNotFoundError
        If one of the bundle files does not exist.
    OSError
        If output files cannot be written.
    scipy.io.matlab.miobase.MatReadError
        If the written MAT file cannot be reloaded successfully.

    Notes
    -----
    Processing steps performed by this function include:

    1. Load sampling frequency and signal units from the physiology JSON.
    2. Load the events and physiology tables.
    3. Identify the appropriate modality column in the physiology data.
    4. Build the EzySCR event column.
    5. Write the MAT file.
    6. Reload the MAT file and derive summary metadata.
    7. Write a JSON sidecar adjacent to the MAT file.

    Output naming
        The MAT filename is generated as
        ``"{subject}_task-{task}_recording-{modality}_physio.mat"``.

    Sidecar contents
        The generated JSON includes sampling frequency, sampling interval,
        inferred CS duration, output column labels, event coding summary, and
        a recommended EzySCR settings block.

    - The current implementation loads sampling info without forwarding the
      provided ``modality`` argument to ``utils.load_sampling_info``.
    - The configuration value ``config["SOA"]`` is read and logged, but not
      otherwise used in output generation.
    """
    
    task = bundle["task"]
    events_tsv = bundle["events_tsv"]
    physio_tsv = bundle["physio_tsv"]
    physio_json = bundle["physio_json"]

    sampling_freq, signal_units = load_sampling_info(physio_json)
    soa = float(config["SOA"])

    logger.info(f"[{subject} | {task}] SamplingFrequency={sampling_freq}")
    logger.info(f"[{subject} | {task}] SOA={soa}")

    events_df = read_table(events_tsv)
    physio_df = read_table(physio_tsv)

    signal_col = find_signal_column(physio_df, modality)
    signal = pd.to_numeric(physio_df[signal_col], errors="coerce").to_numpy(dtype=float)

    event_col = build_event_column(
        events_df=events_df,
        n_samples=len(physio_df),
        sampling_freq=sampling_freq,
        default_value=5,
        cs_value=0,
    ).astype(float)

    if signal_units:
        signal_units = str(signal_units)
    else:
        signal_units = "uS"

    out_name = f"{subject}_task-{task}_recording-{modality}_physio.mat"
    out_path = os.path.join(output_dir, "physio", out_name)

    if os.path.exists(out_path) and not overwrite:
        logger.info(f"MAT exists, skipping: '{out_path}'")
        return out_path

    save_ezyscr_mat(
        out_path=out_path,
        scr_signal=signal,
        event_signal=event_col,
        sampling_freq=sampling_freq,
        scr_units=signal_units,
    )
    logger.info(f"Saved MAT: '{out_path}'")

    # Load MAT to ensure consistency
    m = loadmat(out_path)

    # Sampling frequency
    isi = float(m["isi"].squeeze())
    sampling_freq = 1000.0 / isi

    # Labels (preserve whitespace)
    labels = [_unwrap_mat_string(x) for x in np.array(m["labels"]).ravel()]

    # CS duration
    cs_df = events_df.loc[
        events_df["event_type"].astype(str).str.startswith("CS", na=False)
    ]
    cs_duration = (
        float(cs_df["duration"].dropna().iloc[0])
        if not cs_df.empty else None
    )

    # Event stats
    event_values, _ = np.unique(event_col, return_counts=True)

    # Build sidecar
    ezyscr_meta = {
        "SamplingFrequency": sampling_freq,
        "SamplingInterval_ms": isi,
        "CS_Duration_s": cs_duration,

        "Columns": {
            "SCR": labels[0] if len(labels) > 0 else None,
            "Event": labels[1] if len(labels) > 1 else None,
        },

        "EventCoding": {
            "BaselineValue": int(np.max(event_values)),
            "OnsetValue": int(np.min(event_values)),
        },

        "EzySCR_Settings_Recommendation": {
            "ProgramType": "Acknowledge_Lipp",
            "OldSampleRate": sampling_freq,
            "SCR_var_id": labels[0] if len(labels) > 0 else None,
            "Resp_var_id": None,
            "Event_var_id": labels[1] if len(labels) > 1 else None,
            "CS_Duration": cs_duration,
            "OnsetVar": int(np.min(event_values)),
        }
    }

    out_json = out_path.replace(".mat", ".json")
    cio.save_json(out_json, ezyscr_meta)

    return out_path


def _unwrap_mat_string(
        x
    ):
    """
    Unwrap MATLAB-loaded scalar string-like values into a Python string.

    MATLAB files loaded through SciPy often represent strings as nested NumPy
    arrays or object arrays. This helper repeatedly unwraps single-element
    arrays until a scalar-like value is reached, then converts the result to
    ``str``.

    Parameters
    ----------
    x : Any
        Value loaded from a MAT file. This may be a scalar, a NumPy array, or
        a nested array structure.

    Returns
    -------
    str
        String representation of the unwrapped value.

    Notes
    -----
    - If ``x`` is a NumPy array with exactly one element, that element is
      extracted with ``item()``.
    - If ``x`` is a NumPy array with more than one element, only the first
      flattened element is used.
    - Unwrapping continues until ``x`` is no longer a NumPy array.
    """
    while isinstance(x, np.ndarray):
        if x.size == 1:
            x = x.item()
        else:
            x = x.ravel()[0]
    return str(x)


def print_ezyscr_summary(
        mat_file,
        bundle
    ):
    """
    Log a human-readable summary of an EzySCR MAT file and its source events.

    This function loads a converted MAT file, extracts key metadata needed for
    EzySCR configuration, inspects the original events TSV for event names and
    CS duration, and writes a structured summary to the module logger.

    Parameters
    ----------
    mat_file : Any
        Path to the MAT file to inspect. The value is passed to
        :func:`scipy.io.loadmat`.
    bundle : Any
        Bundle dictionary containing at least an ``"events_tsv"`` entry
        pointing to the original events table used during conversion.

    Returns
    -------
    None
        This function produces logging side effects only.

    Notes
    -----
    The logged summary includes:

    - file name
    - inferred sampling frequency
    - CS duration
    - column labels to copy into EzySCR
    - suggested EzySCR configuration values

    - Labels are extracted from the MAT file using ``_unwrap_mat_string`` so
      whitespace is preserved as closely as possible.
    - Event names are read from the original events TSV and collected, although
      they are not currently emitted in the log output.
    - If no CS-like events are present, the reported CS duration is ``None``.

    Raises
    ------
    KeyError
        If ``bundle`` does not contain ``"events_tsv"``.
    FileNotFoundError
        If the MAT file or events TSV cannot be opened.
    pandas.errors.ParserError
        If the events TSV cannot be parsed.
    """
    m = loadmat(mat_file)

    # Sampling frequency
    isi = float(m["isi"].squeeze())  # ms
    sampling_freq = 1000.0 / isi

    # Labels (preserve whitespace EXACTLY) 
    raw_labels = m["labels"]
    labels = [_unwrap_mat_string(x) for x in np.array(raw_labels).ravel()]

    # Event names from events.tsv
    events_df = pd.read_csv(bundle["events_tsv"], sep="\t")
    event_names = sorted(events_df["event_type"].astype(str).unique())

    # CS duration
    cs_df = events_df.loc[events_df["event_type"].astype(str).str.startswith("CS", na=False)]
    cs_duration = float(cs_df["duration"].dropna().iloc[0]) if not cs_df.empty else None

    # Log output
    logger.info("=" * 60)
    logger.info(f"EzySCR summary: {os.path.basename(mat_file)}")
    logger.info("-" * 60)

    logger.info(f"SamplingFrequency (Hz): {sampling_freq:.2f}")
    logger.info(f"CS Duration (s):        {cs_duration}")

    logger.info("Column labels (COPY EXACTLY into EzySCR):")
    for lab in labels:
        logger.info(f"  -> '{lab}'")

    # Optional: ready-to-use config hint
    logger.info("Suggested EzySCR inputs:")
    logger.info("  Program Type: Acknowledge_Lipp")
    logger.info(f"  Sampling:     {sampling_freq:.0f} Hz")
    logger.info(f"  SCR var id:   '{labels[0]}'")
    logger.info(f"  Resp var id:  ")
    if len(labels) > 1:
        logger.info(f"  Event var id: '{labels[1]}'")
    logger.info(f"  CS Duration:  {cs_duration}")
    logger.info(f"  Onset var:    0")

    logger.info("=" * 60)


def convert_dataset_to_ezyscr(
        input_dir: str,
        output_dir: str,
        modality: str="scr",
        task_name: Optional[str]=None,
        overwrite: bool=False,
        include_n: Optional[int]=None,
        subjects_tsv: Optional[str]=None,
    ) -> None:
    """
    Convert a blinded dataset into EzySCR-compatible MAT files.

    This function iterates over discovered subjects in a blinded CALINET-style
    dataset, locates physiology directories, matches task-specific files for
    the requested modality, converts each matched bundle into a MAT file plus
    JSON sidecar, and finally logs a summary for the last converted bundle.

    Parameters
    ----------
    input_dir : str
        Path to the input BIDS dataset.
    output_dir : str
        Directory where converted files will be written.
    modality : str, default="scr"
        Signal modality to export.
    task_name : str
        Task name to export, such as ``"acquisition"``. If None, all tasks ['acquisition', 'extinction'] will be used.
    overwrite : bool, default=False
        If True, overwrite existing ASCII files.
    include_n : Optional[int], default=None
        Maximum number of subjects to include (ignored if ``subjects_tsv`` is provided).
    subjects_tsv : Optional[str], default=None
        Path to a TSV file specifying selected subjects.

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If the input directory does not exist.
    ValueError
        If no subjects are found or no matching data is available.

    Notes
    -----
    - Supports both standard and multi-site export modes.
    - Generates JSON sidecar metadata for each ASCII file.
    - Logs progress, warnings, and summary information.
    - A final summary is printed for the last processed file.
    """

    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)
    modality = modality.lower()

    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory does not exist: '{input_dir}'")

    os.makedirs(output_dir, exist_ok=True)

    logger.info("Converting BIDS physio data to EzySCR-mat format")
    logger.info(f"Input directory: '{input_dir}'")
    logger.info(f"Output directory: '{output_dir}'")
    logger.info(f"Requested modality: '{modality}'")

    subject_records = discover_subjects(input_dir)
    if not subject_records:
        raise ValueError(f"No subjects found in '{input_dir}'")

    all_subjects = sorted({rec["subject_id"] for rec in subject_records})

    if subjects_tsv is not None:
        selected_subjects = load_subjects_from_export(str(subjects_tsv))
        logger.info(
            f"Loaded {len(selected_subjects)} subjects from export file: '{subjects_tsv}'"
        )
    else:
        selected_subjects = filter_subjects(all_subjects, include_n)

    maybe_copy_participant_files(
        subjects_tsv=subjects_tsv,
        output_dir=output_dir,
    )

    multisite_export_mode = is_multisite_export_mode(input_dir, subjects_tsv)

    last_mat_file = None
    last_bundle = None

    # ------------------------------------------------------------------
    # MULTI-SITE EXPORT MODE → FLATTENED SUBJECT OUTPUT
    if multisite_export_mode:
        logger.info("Running in multi-site export mode with flattened subject output")

        selected_set = set(selected_subjects)

        selected_records = [
            rec for rec in subject_records
            if rec["subject_id"] in selected_set
        ]

        found_ids = {rec["subject_id"] for rec in selected_records}
        missing_subjects = sorted(selected_set - found_ids)
        if missing_subjects:
            logger.warning(
                f"Subjects requested in export file but not found in dataset: {missing_subjects}"
            )

        for rec in selected_records:
            subject = rec["subject_id"]
            subject_dir = rec["subject_path"]
            dest_subject_dir = os.path.join(output_dir, subject)

            physio_dir = find_physio_dir(subject_dir)
            if physio_dir is None:
                logger.warning(
                    f"No physio directory found for '{subject}' from site "
                    f"'{rec['site']}', skipping"
                )
                continue

            bundles = find_matching_files(
                physio_dir,
                modality,
                task_name=task_name
            )

            if not bundles:
                logger.warning(
                    f"No matching task bundles found for subject '{subject}' "
                    f"and modality '{modality}'"
                )
                continue

            logger.info(
                f"Converting subject '{subject}' from site "
                f"'{rec['site']}' → '{dest_subject_dir}'"
            )

            for bundle in bundles:
                last_mat_file = _convert_bundle_to_mat(
                    subject=subject,
                    bundle=bundle,
                    output_dir=dest_subject_dir,
                    modality=modality,
                    overwrite=overwrite,
                )
                last_bundle = bundle

        create_derivative_dataset_description(
            bids_root=output_dir,
            derivative_name="CALINET EzySCR Export",
            source_dataset="CALINET Fear-Conditioning Dataset",
            generated_by_name="CALINET Autonomate Exporter",
            generated_by_description=(
                "Exports electrodermal activity (EDA) recordings from BIDS-formatted "
                "CALINET datasets into EzySCR mat-files compatible with "
                "EzySCR"
            ),
            extra_fields={
                "ExportFormat": {
                    "Name": "EzySCR",
                    "FileExtension": ".mat",
                },
                "EventCoding": {
                    "DefaultCode": 5,
                    "Description": (
                        "Each event_type is mapped to 0."
                    ),
                },
                "Notes": (
                    "This derivative dataset is intended for import into EzySCR"
                ),
            },
        )
        
    # ------------------------------------------------------------------
    # NORMAL MODE → SITE-AWARE SUBJECT OUTPUT
    else:
        logger.info("Running in standard mode")

        selected_set = set(selected_subjects)

        selected_records = [
            rec for rec in subject_records
            if rec["subject_id"] in selected_set
        ]

        if not selected_records:
            raise ValueError("No selected subjects found in dataset")

        for rec in selected_records:
            subject = rec["subject_id"]
            subject_dir = rec["subject_path"]

            if rec["site"] is not None:
                dest_subject_dir = os.path.join(
                    output_dir,
                    rec["site"],
                    rec["subject_id"],
                )
            else:
                dest_subject_dir = os.path.join(
                    output_dir,
                    rec["subject_id"],
                )

            physio_dir = find_physio_dir(subject_dir)
            if physio_dir is None:
                logger.warning(f"No physio directory found for '{subject}', skipping")
                continue

            bundles = find_matching_files(
                physio_dir,
                modality,
                task_name=task_name
            )

            if not bundles:
                logger.warning(
                    f"No matching task bundles found for subject '{subject}' "
                    f"and modality '{modality}'"
                )
                continue

            logger.info(
                f"Converting subject '{subject}' → '{dest_subject_dir}'"
            )

            for bundle in bundles:
                last_mat_file = _convert_bundle_to_mat(
                    subject=subject,
                    bundle=bundle,
                    output_dir=dest_subject_dir,
                    modality=modality,
                    overwrite=overwrite,
                )
                last_bundle = bundle

    # ------------------------------------------------------------------
    # FINAL SUMMARY
    if last_mat_file is not None and last_bundle is not None:
        print_ezyscr_summary(last_mat_file, last_bundle)
    else:
        logger.warning("No EzySCR files were written.")


def convert_dataset_to_mat(
        input_dir: str,
        output_dir: str,
        modality: str="scr",
        task_name: Optional[str]=None,
        overwrite: bool=False,
        include_n: Optional[int]=None,
        subjects_tsv: Optional[str]=None
    ) -> None:
    """
    Convert a dataset into EzySCR-compatible MAT files.

    This function iterates over discovered subjects in a CALINET-style
    dataset, locates physiology directories, matches task-specific files for
    the requested modality, converts each matched bundle into a MAT file plus
    JSON sidecar, and finally logs a summary for the last converted bundle.

    Parameters
    ----------
    input_dir : str
        Root directory of the blinded input dataset.
    output_dir : str
        Directory where converted MAT and JSON files will be written.
    modality : str = "scr"
        Recording modality to convert, such as ``"scr"``. The value is
        lowercased before matching.
    task_name : str
        Task name to export, such as ``"acquisition"``. If None, all tasks ['acquisition', 'extinction'] will be used.        
    overwrite : bool = False
        If True, existing MAT outputs are overwritten. If False, existing MAT
        files are reused and skipped.
    include_n : Optional[int] = None
        If provided, only the first ``include_n`` discovered subjects are
        processed.
    subjects_tsv : Optional[str], default=None
        Path to a TSV file specifying selected subjects.

    Returns
    -------
    None
        Converted files are written to disk as side effects.

    Raises
    ------
    FileNotFoundError
        If ``input_dir`` does not exist.
    OSError
        If ``output_dir`` cannot be created.
    ValueError
        Propagated from downstream helpers when metadata, signal columns, or
        conversion assumptions are invalid.

    Notes
    -----
    Processing workflow
        The function performs the following high-level steps:

        1. Normalize and validate input and output paths.
        2. Discover subject directories.
        3. Optionally restrict processing to the first ``N`` subjects.
        4. Locate each subject's physiology directory.
        5. Build matched file bundles for the requested modality.
        6. Convert each bundle to MAT and JSON outputs.
        7. Print a summary for the final processed MAT file.

    Dataset assumptions
        Subject directories are expected directly beneath ``input_dir`` and
        should be named with the ``"sub-"`` prefix. Physiology data are
        expected under either ``physio`` or ``ses-01/physio`` within each
        subject.

    Logging behavior
        Informational messages are logged for subject discovery, modality
        selection, bundle conversion, and missing physiology/task bundles.

    Caution
    -------
    The summary step at the end uses the variables ``mat_file`` and ``bundle``
    from the most recent successful loop iteration. If no bundles are
    converted, the current implementation would fail before summary logging.
    """
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)
    modality = modality.lower()

    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory does not exist: '{input_dir}'")

    os.makedirs(output_dir, exist_ok=True)

    subjects = discover_subjects(input_dir)
    subjects = filter_subjects(subjects, include_n)

    logger.info(f"Found {len(subjects)} subject(s)")
    logger.info(f"Converting modality: '{modality}'")

    if subjects_tsv is not None:
        selected_subjects = load_subjects_from_export(str(subjects_tsv))
        logger.info(
            f"Loaded {len(selected_subjects)} subjects from export file: '{subjects_tsv}'"
        )
    else:
        selected_subjects = filter_subjects(subjects, include_n)

    maybe_copy_participant_files(
        subjects_tsv=subjects_tsv,
        output_dir=output_dir,
    )

    for subject in subjects:
        subject_dir = os.path.join(input_dir, subject)
        physio_dir = find_physio_dir(subject_dir)

        if physio_dir is None:
            logger.warning(f"No physio directory found for '{subject}'")
            continue

        bundles = find_matching_files(
            physio_dir,
            modality=modality,
            task_name=task_name
        )

        if not bundles:
            logger.warning(f"No matching task bundles found for '{subject}' and modality '{modality}'")
            continue

        for bundle in bundles:
            mat_file = _convert_bundle_to_mat(
                subject=subject,
                bundle=bundle,
                output_dir=output_dir,
                modality=modality,
                overwrite=overwrite,
            )

    print_ezyscr_summary(mat_file, bundle)

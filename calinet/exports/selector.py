# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from pathlib import Path
import pandas as pd
import calinet.core.io as cio
from calinet.exports.utils import build_participants_sidecar

EXPORT_TRACKING_COLUMNS = [
    "eligible_for_export",
    "ever_exported",
    "export_count",
    "last_export_batch",
    "last_export_date",
]


def write_participants_files(
        df: pd.DataFrame,
        out_dir: str | Path
    ) -> None:
    """
    Write participant export files to an output directory.

    This function writes a tab-separated ``participants.tsv`` file and its
    accompanying ``participants.json`` sidecar into ``out_dir``. The JSON
    sidecar is generated from the dataframe using
    :func:`build_participants_sidecar`.

    Parameters
    ----------
    df : pandas.DataFrame
        Participant-level table to write.
    out_dir : str or pathlib.Path
        Destination directory where ``participants.tsv`` and
        ``participants.json`` will be created. The directory is created if it
        does not already exist.

    Returns
    -------
    None
        This function performs file-system side effects only.

    Notes
    -----
    - The output directory is created with ``parents=True`` and
      ``exist_ok=True``.
    - The TSV file is written without the dataframe index.
    - The JSON sidecar reflects the current dataframe columns and may include
      fallback descriptions for non-template fields.

    See Also
    --------
    build_participants_sidecar : Construct metadata for ``participants.json``.

    Raises
    ------
    OSError
        If the output directory cannot be created or one of the files cannot
        be written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    participants_tsv = out_dir / "participants.tsv"
    participants_json = out_dir / "participants.json"

    df.to_csv(participants_tsv, sep="\t", index=False)
    cio.save_json(participants_json, build_participants_sidecar(df))


@dataclass(frozen=True)
class ExportSummary:
    batch_id: str
    output_dir: Path
    export_log_path: Path
    updated_registry_path: Path
    metadata_path: Path
    n_selected: int


def build_participants_registry(
        raw_path: Path | str
    ) -> pd.DataFrame:
    """
    Build a combined participant registry from lab-level phenotype tables.

    This function searches each immediate subdirectory of ``raw_path`` for a
    phenotype file located at ``phenotype/participant_info.tsv``. All such
    files are read, annotated with the lab name as a ``site`` column, and
    concatenated into a single participants registry. Export-tracking columns
    are then added or normalized using
    :func:`initialize_registry_columns`.

    Parameters
    ----------
    raw_path : pathlib.Path or str
        Root directory containing one subdirectory per lab or site. Each lab
        directory is expected to contain a phenotype table at
        ``phenotype/participant_info.tsv``.

    Returns
    -------
    pandas.DataFrame
        Combined participant registry containing rows from all discovered
        phenotype tables, along with normalized export-tracking columns.

    Raises
    ------
    FileNotFoundError
        If ``raw_path`` does not exist.
    ValueError
        If no ``participant_info.tsv`` files are found under the provided
        directory.

    Notes
    -----
    - Missing phenotype tables for individual labs are silently skipped.
    - The string ``"n/a"`` is interpreted as missing data when reading the TSV
      files.
    - The site name is taken from the lab directory name.
    - The returned dataframe is passed through
      :func:`initialize_registry_columns` before being returned.

    See Also
    --------
    initialize_registry_columns : Add and normalize export-tracking columns.
    load_registry : Load an existing saved registry from disk.
    """
    raw_path = Path(raw_path)
    if not raw_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {raw_path}")

    dfs: list[pd.DataFrame] = []
    for lab_dir in sorted([p for p in raw_path.iterdir() if p.is_dir()]):
        pheno_file = lab_dir / "phenotype" / "participant_info.tsv"
        if not pheno_file.exists():
            continue

        df = pd.read_csv(pheno_file, delimiter="\t", na_values=["n/a"])
        df["site"] = lab_dir.name
        dfs.append(df)

    if not dfs:
        raise ValueError(f"No participant_info.tsv files found under: {raw_path}")

    participants = pd.concat(dfs, ignore_index=True)
    participants = initialize_registry_columns(participants)
    return participants


def initialize_registry_columns(
        df: pd.DataFrame
    ) -> pd.DataFrame:
    """
    Ensure that a participant registry contains stable export-tracking columns.

    This function validates that the dataframe contains the minimum required
    structural columns and then adds any missing export-tracking columns with
    default values. Existing tracking columns are normalized to predictable
    types so downstream selection and export bookkeeping behave consistently.

    Parameters
    ----------
    df : pandas.DataFrame
        Participant registry dataframe. It must contain at least
        ``participant_id`` and ``site`` columns.

    Returns
    -------
    pandas.DataFrame
        A copy of the input dataframe with all required export-tracking columns
        present and normalized.

    Raises
    ------
    ValueError
        If ``participant_id`` is missing.
    ValueError
        If ``site`` is missing.

    Notes
    -----
    The following tracking columns are ensured:

    ``eligible_for_export``
        Boolean flag indicating whether the participant is currently eligible
        for export.
    ``ever_exported``
        Boolean flag indicating whether the participant has ever been included
        in an export package.
    ``export_count``
        Integer count of how many times the participant has been exported.
    ``last_export_batch``
        Identifier of the most recent export batch, or missing if never
        exported.
    ``last_export_date``
        Date string for the most recent export, or missing if never exported.

    Existing values are normalized as follows:

    - ``eligible_for_export`` is filled with ``True`` where missing and cast to
      ``bool``.
    - ``ever_exported`` is filled with ``False`` where missing and cast to
      ``bool``.
    - ``export_count`` is coerced to numeric, missing or invalid values are
      replaced with ``0``, and the column is cast to ``int``.

    See Also
    --------
    build_participants_registry : Create a registry from phenotype tables.
    load_registry : Read a registry file and normalize its columns.
    """
    df = df.copy()

    if "participant_id" not in df.columns:
        raise ValueError("participants table must contain a 'participant_id' column")
    if "site" not in df.columns:
        raise ValueError("participants table must contain a 'site' column")

    defaults = {
        "eligible_for_export": True,
        "ever_exported": False,
        "export_count": 0,
        "last_export_batch": pd.NA,
        "last_export_date": pd.NA,
    }

    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    df["eligible_for_export"] = df["eligible_for_export"].fillna(True).astype(bool)
    df["ever_exported"] = df["ever_exported"].fillna(False).astype(bool)
    df["export_count"] = pd.to_numeric(df["export_count"], errors="coerce").fillna(0).astype(int)
    return df


def load_registry(
        registry_path: Path | str
    ) -> pd.DataFrame:
    """
    Load a participant registry from disk and normalize tracking columns.

    This function reads a registry file from ``registry_path`` and infers the
    delimiter from the file extension. Tab-separated files are assumed for
    ``.tsv`` and ``.tab`` extensions, while all other extensions are treated
    as comma-separated. The loaded dataframe is then passed through
    :func:`initialize_registry_columns`.

    Parameters
    ----------
    registry_path : pathlib.Path or str
        Path to the registry file to load.

    Returns
    -------
    pandas.DataFrame
        Loaded participant registry with normalized export-tracking columns.

    Notes
    -----
    - File extension is used only to choose the delimiter.
    - Missing values are preserved using pandas default NA handling.
    - The returned dataframe is always normalized with
      :func:`initialize_registry_columns`.

    See Also
    --------
    initialize_registry_columns : Ensure registry columns and defaults.
    save_registry : Persist a registry to disk.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    pandas.errors.ParserError
        If the file cannot be parsed with the inferred delimiter.
    """
    registry_path = Path(registry_path)
    sep = "\t" if registry_path.suffix.lower() in {".tsv", ".tab"} else ","
    df = pd.read_csv(registry_path, sep=sep, keep_default_na=True)
    return initialize_registry_columns(df)


def save_registry(
        df: pd.DataFrame,
        output_path: Path | str
    ) -> Path:
    """
    Save a participant registry to a tab-separated file.

    This function writes the provided dataframe to ``output_path`` as a TSV
    file, creating parent directories as needed.

    Parameters
    ----------
    df : pandas.DataFrame
        Registry dataframe to write.
    output_path : pathlib.Path or str
        Destination path for the TSV file.

    Returns
    -------
    pathlib.Path
        The resolved output path as a ``Path`` object.

    Notes
    -----
    - The file is always written as tab-separated text regardless of the file
      extension.
    - The dataframe index is not written.
    - Parent directories are created automatically.

    See Also
    --------
    load_registry : Load a registry from disk.
    write_export_package : Save the updated registry as part of an export.

    Raises
    ------
    OSError
        If the parent directory cannot be created or the file cannot be
        written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, sep="\t", index=False)
    return output_path


def allocate_counts(
        total_n: int,
        counts_by_site: pd.Series,
        mode: str
    ) -> dict[str, int]:
    """
    Allocate a target number of selections across sites.

    This function computes per-site selection counts for a requested total
    sample size under one of two allocation strategies:

    ``"equal"``
        Distribute selections as evenly as possible across sites with eligible
        participants, while respecting per-site capacity constraints.

    ``"proportional"``
        Distribute selections in proportion to the number of eligible
        participants at each site, again respecting site capacities.

    In both modes, if initial assignment leaves unallocated quota because some
    sites hit capacity, the remaining quota is reassigned greedily to sites
    that still have available capacity.

    Parameters
    ----------
    total_n : int
        Total number of participants to allocate across sites. Must be greater
        than zero.
    counts_by_site : pandas.Series
        Series indexed by site name with integer counts of eligible
        participants available at each site.
    mode : str
        Allocation strategy. Must be one of ``"equal"`` or
        ``"proportional"``.

    Returns
    -------
    dict[str, int]
        Mapping from site name to the number of participants to sample from
        that site.

    Raises
    ------
    ValueError
        If ``total_n`` is not greater than zero.
    ValueError
        If ``counts_by_site`` is empty.
    ValueError
        If ``mode`` is not one of ``"equal"`` or ``"proportional"``.
    ValueError
        If ``mode="equal"`` and no sites have positive capacity.

    Notes
    -----
    Equal mode
        Eligible sites are those with capacity greater than zero. The requested
        total is split into a base allocation plus remainder. The remainder is
        distributed across sites in index order, then any still-unassigned
        quota is filled greedily.

    Proportional mode
        Raw proportional allocations are computed from
        ``counts_by_site / counts_by_site.sum() * total_n``. The floor of each
        raw allocation is taken first, then the remaining quota is assigned in
        descending order of fractional remainder.

    Capacity handling
        In both modes, no site is assigned more than its available count in
        ``counts_by_site``.

    Examples
    --------
    >>> counts = pd.Series({"site_a": 10, "site_b": 20, "site_c": 5})
    >>> allocate_counts(9, counts, mode="equal")
    {'site_a': 3, 'site_b': 3, 'site_c': 3}

    >>> allocate_counts(9, counts, mode="proportional")
    {'site_a': 3, 'site_b': 5, 'site_c': 1}

    See Also
    --------
    select_subjects : Use the allocation to draw participant samples.
    """
    if total_n <= 0:
        raise ValueError("total_n must be > 0")
    if counts_by_site.empty:
        raise ValueError("No eligible subjects available after filtering")

    site_names = list(counts_by_site.index)
    capacities = counts_by_site.to_dict()

    if mode == "equal":
        active_sites = [s for s in site_names if capacities[s] > 0]
        if not active_sites:
            raise ValueError("No sites have eligible subjects")

        base = total_n // len(active_sites)
        remainder = total_n % len(active_sites)
        allocation = {s: min(base, capacities[s]) for s in active_sites}

        # spread the remainder and then fill any unused quota greedily
        for s in active_sites[:remainder]:
            if allocation[s] < capacities[s]:
                allocation[s] += 1

        assigned = sum(allocation.values())
        while assigned < total_n:
            progressed = False
            for s in active_sites:
                if allocation[s] < capacities[s]:
                    allocation[s] += 1
                    assigned += 1
                    progressed = True
                    if assigned == total_n:
                        break
            if not progressed:
                break
        return allocation

    if mode == "proportional":
        proportions = counts_by_site / counts_by_site.sum()
        raw = proportions * total_n
        floored = np.floor(raw).astype(int)
        allocation = floored.to_dict()

        for s in allocation:
            allocation[s] = min(allocation[s], capacities[s])

        assigned = sum(allocation.values())
        remainders = (raw - floored).sort_values(ascending=False)
        while assigned < total_n:
            progressed = False
            for s in remainders.index:
                if allocation[s] < capacities[s]:
                    allocation[s] += 1
                    assigned += 1
                    progressed = True
                    if assigned == total_n:
                        break
            if not progressed:
                break
        return allocation

    raise ValueError("mode must be one of: 'equal', 'proportional'")


def select_subjects(
        registry_df: pd.DataFrame,
        n: int,
        mode: str="equal",
        seed: int=1234,
        exclude_previously_exported: bool=True,
        extra_filters: Optional[dict[str, Iterable]]=None
    ) -> pd.DataFrame:
    """
    Select participants using site-stratified random sampling.

    This function filters a participant registry down to eligible rows, applies
    optional exclusion of previously exported participants and any additional
    caller-provided filters, computes a per-site allocation, and then samples
    participants without replacement within each site. The final selected set
    is shuffled and returned sorted by ``participant_id``.

    Parameters
    ----------
    registry_df : pandas.DataFrame
        Participant registry containing at least ``participant_id``, ``site``,
        and the export-tracking columns required by
        :func:`initialize_registry_columns`. Missing tracking columns are added
        automatically.
    n : int
        Number of participants to select. Must be greater than zero.
    mode : str, default="equal"
        Site allocation strategy passed to :func:`allocate_counts`. Supported
        values are ``"equal"`` and ``"proportional"``.
    seed : int, default=1234
        Seed for the NumPy random number generator used for reproducible
        sampling.
    exclude_previously_exported : bool, default=True
        If True, rows with ``ever_exported == True`` are excluded before
        sampling.
    extra_filters : Optional[dict[str, Iterable]], default=None
        Optional mapping from column name to an iterable of accepted values.
        Rows are retained only if their value in each specified column is a
        member of the corresponding accepted set.

    Returns
    -------
    pandas.DataFrame
        Dataframe containing the selected participants.

    Raises
    ------
    ValueError
        If ``n`` is not greater than zero.
    ValueError
        If a filter column in ``extra_filters`` is not present in the
        dataframe.
    ValueError
        If no eligible subjects remain after filtering.
    ValueError
        If ``n`` exceeds the number of eligible subjects available.
    ValueError
        Propagated from :func:`allocate_counts` if the allocation cannot be
        computed.

    Notes
    -----
    Filtering proceeds in the following order:

    1. Normalize registry columns with :func:`initialize_registry_columns`.
    2. Keep only rows where ``eligible_for_export`` is True.
    3. Optionally exclude rows already exported.
    4. Apply each entry in ``extra_filters``.

    Sampling behavior
        Sampling is performed independently within each site using
        ``DataFrame.sample(..., replace=False)``. A master NumPy random number
        generator is seeded with ``seed`` and used to derive per-site random
        states, which improves reproducibility while avoiding identical sampling
        streams across sites.

    Output ordering
        After concatenating selected site-specific samples, the dataframe is
        shuffled once more and then sorted by ``participant_id`` before being
        returned.

    See Also
    --------
    allocate_counts : Compute per-site target counts.
    update_registry_with_export : Mark selected participants as exported.
    """
    if n <= 0:
        raise ValueError("n must be > 0")

    df = initialize_registry_columns(registry_df)
    eligible = df[df["eligible_for_export"]].copy()

    if exclude_previously_exported:
        eligible = eligible[~eligible["ever_exported"]].copy()

    if extra_filters:
        for column, accepted_values in extra_filters.items():
            if column not in eligible.columns:
                raise ValueError(f"Filter column not found: {column}")
            accepted_values = set(accepted_values)
            eligible = eligible[eligible[column].isin(accepted_values)].copy()

    if eligible.empty:
        raise ValueError("No eligible subjects remain after filtering")
    if n > len(eligible):
        raise ValueError(
            f"Requested n={n}, but only {len(eligible)} eligible subjects are available"
        )

    counts_by_site = eligible.groupby("site")["participant_id"].count().sort_index()
    allocation = allocate_counts(n, counts_by_site, mode=mode)

    rng = np.random.default_rng(seed)
    selected_chunks = []
    for site, site_n in allocation.items():
        if site_n <= 0:
            continue
        site_df = eligible.loc[eligible["site"] == site].copy()
        random_state = int(rng.integers(0, np.iinfo(np.int32).max))
        sampled = site_df.sample(n=site_n, replace=False, random_state=random_state)
        selected_chunks.append(sampled)

    selected = pd.concat(selected_chunks, ignore_index=True)
    random_state = int(rng.integers(0, np.iinfo(np.int32).max))
    selected = selected.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    selected = selected.sort_values("participant_id").reset_index(drop=True)
    return selected


def make_batch_id(
        recipient_lab: str,
        n: int,
        seed: int
    ) -> str:
    """
    Construct a human-readable export batch identifier.

    The batch identifier includes the current date, a sanitized recipient lab
    name, the requested sample size, and the random seed used for selection.

    Parameters
    ----------
    recipient_lab : str
        Name of the recipient lab or organization. Spaces are replaced with
        underscores in the generated identifier.
    n : int
        Requested number of participants in the export batch.
    seed : int
        Random seed associated with participant selection.

    Returns
    -------
    str
        Batch identifier of the form
        ``EXP_<YYYY-MM-DD>_<recipient>_N<n>_seed<seed>``.

    Notes
    -----
    - The date component is generated using the local current date from
      ``datetime.now()``.
    - This function does not validate characters beyond replacing spaces with
      underscores.

    See Also
    --------
    write_export_package : Use the batch identifier when creating an export.
    """
    stamp = datetime.now().strftime("%Y-%m-%d")
    safe_recipient = recipient_lab.replace(" ", "_")
    return f"EXP_{stamp}_{safe_recipient}_N{n}_seed{seed}"


def append_export_log(
        selected_df: pd.DataFrame,
        export_log_path: Path | str,
        *,
        batch_id: str,
        recipient_lab: str,
        sampling_method: str,
        seed: int
    ) -> Path:
    """
    Append selected participants to the cumulative export log.

    This function creates a log dataframe from the selected participants,
    annotates it with export metadata, and appends it to an existing export
    log if present. The resulting log is written as a TSV file.

    Parameters
    ----------
    selected_df : pandas.DataFrame
        Dataframe of selected participants. It must contain at least
        ``participant_id`` and ``site`` columns.
    export_log_path : pathlib.Path or str
        Path to the export log TSV file.
    batch_id : str
        Identifier of the export batch.
    recipient_lab : str
        Name of the recipient lab or organization.
    sampling_method : str
        Description of the sampling method used to generate the batch, such as
        ``"equal"`` or ``"proportional"``.
    seed : int
        Random seed used during participant selection.

    Returns
    -------
    pathlib.Path
        Path to the written export log file.

    Notes
    -----
    - The current date is recorded in ISO format in the ``export_date`` column.
    - If an export log already exists, its contents are read and the new rows
      are appended.
    - No deduplication is performed; repeated exports of the same participant
      will create multiple log entries.

    See Also
    --------
    update_registry_with_export : Update participant-level export history.
    write_export_package : High-level export writer that calls this function.

    Raises
    ------
    KeyError
        If ``selected_df`` does not contain required columns.
    OSError
        If the log directory cannot be created or the file cannot be written.
    """
    export_log_path = Path(export_log_path)
    export_log_path.parent.mkdir(parents=True, exist_ok=True)

    export_date = datetime.now().date().isoformat()
    log_df = selected_df[["participant_id", "site"]].copy()
    log_df.insert(0, "batch_id", batch_id)
    log_df.insert(1, "export_date", export_date)
    log_df["recipient_lab"] = recipient_lab
    log_df["sampling_method"] = sampling_method
    log_df["seed"] = seed

    if export_log_path.exists():
        existing = pd.read_csv(export_log_path, sep="\t")
        log_df = pd.concat([existing, log_df], ignore_index=True)

    log_df.to_csv(export_log_path, sep="\t", index=False)
    return export_log_path


def update_registry_with_export(
        registry_df: pd.DataFrame,
        selected_df: pd.DataFrame,
        *,
        batch_id: str,
        export_date: Optional[str]=None
    ) -> pd.DataFrame:
    """
    Update registry export-tracking fields for selected participants.

    This function marks participants present in ``selected_df`` as exported in
    the registry, increments their export count, and records the most recent
    batch identifier and export date.

    Parameters
    ----------
    registry_df : pandas.DataFrame
        Full participant registry.
    selected_df : pandas.DataFrame
        Dataframe containing the subset of selected participants. The
        ``participant_id`` column is used to identify which rows to update in
        the registry.
    batch_id : str
        Identifier of the export batch to record.
    export_date : Optional[str], default=None
        Export date to record, typically in ISO format. If None, the current
        date is generated automatically.

    Returns
    -------
    pandas.DataFrame
        Updated copy of the participant registry.

    Notes
    -----
    The following columns are updated for matched participants:

    - ``ever_exported`` is set to True.
    - ``export_count`` is incremented by 1.
    - ``last_export_batch`` is set to ``batch_id``.
    - ``last_export_date`` is set to ``export_date``.

    Matching behavior
        Participant identifiers are converted to strings before comparison to
        reduce type-mismatch issues between registries and selected subsets.

    See Also
    --------
    initialize_registry_columns : Normalize registry structure before updates.
    append_export_log : Record batch membership in a cumulative log.
    write_export_package : High-level function that performs these updates.

    Raises
    ------
    KeyError
        If ``selected_df`` does not contain ``participant_id``.
    """
    df = initialize_registry_columns(registry_df)
    export_date = export_date or datetime.now().date().isoformat()
    selected_ids = set(selected_df["participant_id"].astype(str))

    mask = df["participant_id"].astype(str).isin(selected_ids)
    df.loc[mask, "ever_exported"] = True
    df.loc[mask, "export_count"] = df.loc[mask, "export_count"].fillna(0).astype(int) + 1
    df.loc[mask, "last_export_batch"] = batch_id
    df.loc[mask, "last_export_date"] = export_date
    return df


def write_export_package(
        registry_df: pd.DataFrame,
        selected_df: pd.DataFrame,
        output_dir: Path | str,
        *,
        batch_id: str,
        recipient_lab: str,
        sampling_method: str,
        seed: int,
        save_updated_registry: bool=True
    ) -> ExportSummary:
    """
    Write a complete export package and associated bookkeeping files.

    This is a high-level orchestration function that creates a batch-specific
    output directory, writes participant export files, saves metadata for the
    batch, appends the cumulative export log, updates the participant registry,
    and optionally persists the updated registry.

    Parameters
    ----------
    registry_df : pandas.DataFrame
        Full participant registry prior to applying export updates.
    selected_df : pandas.DataFrame
        Dataframe containing the participants included in the current export
        batch.
    output_dir : pathlib.Path or str
        Root directory where export artifacts will be written. A batch-specific
        subdirectory named ``batch_id`` is created beneath this directory.
    batch_id : str
        Identifier for the export batch.
    recipient_lab : str
        Name of the recipient lab or organization.
    sampling_method : str
        Description of the sampling method used for participant selection.
    seed : int
        Random seed used during participant selection.
    save_updated_registry : bool, default=True
        If True, save the updated registry to
        ``participants_registry.tsv`` in ``output_dir``. If False, the updated
        registry is computed but not written to disk.

    Returns
    -------
    ExportSummary
        Dataclass describing key paths and counts associated with the written
        export package.

    Notes
    -----
    Files written by this function include:

    - ``<output_dir>/<batch_id>/participants.tsv``
    - ``<output_dir>/<batch_id>/participants.json``
    - ``<output_dir>/<batch_id>/metadata.json``
    - ``<output_dir>/export_log.tsv``
    - ``<output_dir>/participants_registry.tsv`` (optional)

    Metadata contents
        The batch metadata JSON records the batch identifier, UTC creation
        timestamp, recipient lab, selected count, sampling method, and random
        seed.

    Registry handling
        The updated registry is always computed via
        :func:`update_registry_with_export`. Persisting it to disk is
        controlled by ``save_updated_registry``.

    See Also
    --------
    write_participants_files : Write batch participant TSV/JSON files.
    append_export_log : Update the cumulative export log.
    update_registry_with_export : Update participant export history.

    Raises
    ------
    OSError
        If required directories or files cannot be created or written.
    KeyError
        If required columns are missing from ``selected_df`` or ``registry_df``.
    """

    output_dir = Path(output_dir)
    batch_dir = output_dir / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = batch_dir / "metadata.json"
    export_log_path = output_dir / "export_log.tsv"
    updated_registry_path = output_dir / "participants_registry.tsv"

    write_participants_files(selected_df, batch_dir)

    metadata = {
        "batch_id": batch_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "recipient_lab": recipient_lab,
        "n_requested": int(len(selected_df)),
        "n_selected": int(len(selected_df)),
        "sampling_method": sampling_method,
        "random_seed": int(seed),
    }

    cio.save_json(metadata_path, metadata)

    append_export_log(
        selected_df,
        export_log_path,
        batch_id=batch_id,
        recipient_lab=recipient_lab,
        sampling_method=sampling_method,
        seed=seed,
    )

    updated_registry = update_registry_with_export(
        registry_df,
        selected_df,
        batch_id=batch_id,
    )
    if save_updated_registry:
        save_registry(updated_registry, updated_registry_path)

    return ExportSummary(
        batch_id=batch_id,
        output_dir=batch_dir,
        export_log_path=export_log_path,
        updated_registry_path=updated_registry_path,
        metadata_path=metadata_path,
        n_selected=len(selected_df),
    )

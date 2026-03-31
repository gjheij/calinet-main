# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import math
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd


# Math helpers
def fisher_z(r: float) -> float:
    if r >= 1.0:
        return float("inf")
    if r <= -1.0:
        return float("-inf")
    return 0.5 * math.log((1.0 + r) / (1.0 - r))


def inv_fisher_z(z: float) -> float:
    return math.tanh(z)


def pearson_correlation(x, y) -> float:
    x = np.asarray(list(x), dtype=float)
    y = np.asarray(list(y), dtype=float)

    n = len(x)
    if n < 2 or len(y) != n:
        return float("nan")

    mean_x = x.mean()
    mean_y = y.mean()

    dx = x - mean_x
    dy = y - mean_y

    num = np.sum(dx * dy)
    den_x = np.sum(dx * dx)
    den_y = np.sum(dy * dy)

    den = np.sqrt(den_x * den_y)
    return float("nan") if den == 0.0 else float(num / den)


def compute_correlation_ci(r: float, n: int):
    if n < 4 or pd.isna(r):
        return None, None

    z = fisher_z(r)
    se = 1.0 / math.sqrt(n - 3)
    z_crit = 1.96

    low = inv_fisher_z(z - z_crit * se)
    high = inv_fisher_z(z + z_crit * se)

    low = max(-1.0, low)
    high = min(1.0, high)

    return low, high


def compute_mean_ci(values):
    values = np.asarray(list(values), dtype=float)
    n = len(values)

    if n < 3:
        return None, None

    mean = values.mean()
    variance = np.sum((values - mean) ** 2) / (n - 1)
    sd = math.sqrt(variance)
    se = sd / math.sqrt(n)

    if n >= 6:
        t = 1.96
    elif n == 3:
        t = 4.303
    elif n == 4:
        t = 3.182
    elif n == 5:
        t = 2.776
    else:
        t = 1.96

    return float(mean - t * se), float(mean + t * se)


def count_distinct_datasets(rows: pd.DataFrame) -> int:
    if rows is None or rows.empty:
        return 0
    return int(rows["data_id"].dropna().nunique())


def first_non_null(rows: pd.DataFrame, col: str):
    if col not in rows.columns:
        return None
    s = rows[col].dropna()
    return None if s.empty else s.iloc[0]



# Core score logic
def compute_retrodictive_validity(rows: pd.DataFrame) -> Optional[float]:
    """
    Python equivalent of Java computeRetrodictiveValidity(List<DatasetTaskOutputDTO> rows)

    Required columns:
        data_id, parsed_value, intended_value
    """
    if rows is None or len(rows) < 3:
        return None

    filtered = rows.dropna(subset=["parsed_value", "intended_value"])
    if filtered.empty:
        return None

    intended = []
    parsed_centered = []

    for _, subject_rows in filtered.groupby("data_id", sort=False):
        if len(subject_rows) < 2:
            continue

        mean_parsed = subject_rows["parsed_value"].mean()
        if pd.isna(mean_parsed):
            continue

        centered = subject_rows["parsed_value"] - mean_parsed
        parsed_centered.extend(centered.tolist())
        intended.extend(subject_rows["intended_value"].tolist())

    if len(parsed_centered) < 3 or len(parsed_centered) != len(intended):
        return None

    r = pearson_correlation(intended, parsed_centered)
    return None if np.isnan(r) else float(r)


def compute_dataset_contrast(rows: pd.DataFrame) -> Optional[float]:
    """
    Java equivalent:
        ordered by output_order,
        contrast = first parsed_value - second parsed_value
    """
    if rows is None or len(rows) < 2:
        return None

    ordered = (
        rows.dropna(subset=["output_order", "parsed_value"])
        .sort_values("output_order")
    )

    if len(ordered) < 2:
        return None

    return float(ordered.iloc[0]["parsed_value"] - ordered.iloc[1]["parsed_value"])


def build_score_row(
        method_id: int,
        task_id:
        int,
        rows: pd.DataFrame
    ) -> Dict[str, Any]:

    rv = None
    ci_low = None
    ci_high = None
    n_datasets = 0

    if rows is not None and not rows.empty:
        rv = compute_retrodictive_validity(rows)
        n_rows = len(rows)

        if rv is not None and n_rows >= 4:
            ci_low, ci_high = compute_correlation_ci(rv, n_rows)

        n_datasets = count_distinct_datasets(rows)

    return {
        "method_id": int(method_id),
        "task_id": int(task_id),
        "score": rv,
        "ci_low": ci_low,
        "score_ci_high": ci_high,
        "n_datasets": n_datasets,
    }



# DataFrame-returning service equivalents
def get_scores_by_method(
        df: pd.DataFrame,
        method_id: int
    ) -> pd.DataFrame:
    """
    Equivalent to getScoresByMethod(int methodId)
    Uses only the main dataframe.
    """
    method_rows = df[df["method_id"] == method_id]
    if method_rows.empty:
        return pd.DataFrame(
            columns=["method_id", "task_id", "score", "ci_low", "score_ci_high", "n_datasets"]
        )

    task_ids = sorted(method_rows["task_id"].dropna().unique().tolist())

    out = []
    for task_id in task_ids:
        rows = method_rows[method_rows["task_id"] == task_id]
        out.append(build_score_row(method_id=method_id, task_id=task_id, rows=rows))

    return pd.DataFrame(out).sort_values(["task_id", "method_id"]).reset_index(drop=True)


def get_best_scores_by_task_id(
        df: pd.DataFrame,
        task_id: int
    ) -> pd.DataFrame:
    """
    Equivalent to getBestScoresByTaskId(int taskId)
    """
    all_rows = df[df["task_id"] == task_id]
    if all_rows.empty:
        return pd.DataFrame(
            columns=[
                "method_id", "method_name", "task_name", "method_description",
                "n_datasets", "score", "ci_low", "ci_high",
                "lab_head", "observable_name"
            ]
        )

    rankings = []

    for method_id, rows in all_rows.groupby("method_id", sort=False):
        if rows.empty:
            continue

        rv = compute_retrodictive_validity(rows)
        if rv is None:
            continue

        n_rows = len(rows)
        ci_low, ci_high = (None, None)
        if n_rows >= 4:
            ci_low, ci_high = compute_correlation_ci(rv, n_rows)

        rankings.append({
            "method_id": int(method_id),
            "method_name": first_non_null(rows, "method_name"),
            "task_name": str(task_id),
            "method_description": None,
            "n_datasets": count_distinct_datasets(rows),
            "score": rv,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "lab_head": None,
            "observable_name": None,
        })

    result = pd.DataFrame(rankings)
    if result.empty:
        return result

    return (
        result.sort_values(["score", "method_id"], ascending=[False, True], na_position="last")
        .reset_index(drop=True)
    )


def get_best_scores_by_task_id_grouped_by_observable(
        df: pd.DataFrame,
        task_id: int
    ) -> Dict[str, pd.DataFrame]:
    """
    Equivalent to getBestScoresByTaskIdGroupedByObservable(int taskId)

    Returns:
        dict[observable_name] -> dataframe
    """
    all_rows = df[df["task_id"] == task_id].copy()
    if all_rows.empty:
        return {}

    if "observable_name" not in all_rows.columns:
        all_rows["observable_name"] = "Unknown"

    all_rows["observable_name"] = all_rows["observable_name"].fillna("Unknown")

    result = {}

    for observable, observable_rows in all_rows.groupby("observable_name", sort=False):
        rankings = []

        for method_id, rows in observable_rows.groupby("method_id", sort=False):
            rv = compute_retrodictive_validity(rows)
            if rv is None or rows.empty:
                continue

            n_rows = len(rows)
            ci_low, ci_high = (None, None)
            if n_rows >= 4:
                ci_low, ci_high = compute_correlation_ci(rv, n_rows)

            rankings.append({
                "method_id": int(method_id),
                "method_name": first_non_null(rows, "method_name"),
                "task_name": str(task_id),
                "method_description": None,
                "observable_name": observable,
                "n_datasets": count_distinct_datasets(rows),
                "score": rv,
                "ci_low": ci_low,
                "ci_high": ci_high,
            })

        obs_df = pd.DataFrame(rankings)
        if not obs_df.empty:
            obs_df = obs_df.sort_values(
                ["score", "method_id"], ascending=[False, True], na_position="last"
            ).reset_index(drop=True)

        result[observable] = obs_df

    return result


def get_rv_by_lab(
        df: pd.DataFrame,
        task_id: int
    ) -> pd.DataFrame:
    """
    Equivalent to getRvByLab(int taskId)

    Steps:
    1. find best method for the task by RV
    2. compute RV by lab_head within that best method
    """
    all_rows = df[df["task_id"] == task_id]
    if all_rows.empty or "lab_head" not in all_rows.columns:
        return pd.DataFrame(columns=["lab_head", "score", "ci_low", "ci_high", "n_datasets"])

    rows_by_method = {
        method_id: rows
        for method_id, rows in all_rows.dropna(subset=["method_id"]).groupby("method_id", sort=False)
    }

    best_method_id = None
    best_method_rv = None

    for method_id, rows in rows_by_method.items():
        rv = compute_retrodictive_validity(rows)
        if rv is None:
            continue
        if best_method_rv is None or rv > best_method_rv:
            best_method_rv = rv
            best_method_id = method_id

    if best_method_id is None:
        return pd.DataFrame(columns=["lab_head", "score", "ci_low", "ci_high", "n_datasets"])

    best_method_rows = rows_by_method[best_method_id]

    results = []
    for lab_head, rows in best_method_rows.dropna(subset=["lab_head"]).groupby("lab_head", sort=False):
        if str(lab_head).strip() == "":
            continue

        rv = compute_retrodictive_validity(rows)
        if rv is None:
            continue

        ci_low, ci_high = (None, None)
        if len(rows) >= 4:
            ci_low, ci_high = compute_correlation_ci(rv, len(rows))

        results.append({
            "lab_head": lab_head,
            "score": rv,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "n_datasets": count_distinct_datasets(rows),
        })

    result = pd.DataFrame(results)
    if result.empty:
        return result

    return result.sort_values(["score", "lab_head"], ascending=[False, True], na_position="last").reset_index(drop=True)


def get_cumulative_contrast_by_method(
        df: pd.DataFrame,
        task_id: int
    ) -> Dict[str, pd.DataFrame]:
    """
    Equivalent to getCumulativeContrastByMethod(int taskId)

    Returns:
        dict[method_name] -> dataframe with columns:
            step, data_id, filename, contrast, cumulative_mean_contrast

    Note:
        cumulative_mean_contrast is kept to mirror the Java field name,
        but the actual value is cumulative retrodictive validity.
    """
    all_rows = df[df["task_id"] == task_id].copy()
    if all_rows.empty:
        return {}

    result = {}

    for method_id, rows in all_rows.groupby("method_id", sort=False):
        if rows.empty:
            continue

        method_name = first_non_null(rows, "method_name")
        if method_name is None:
            method_name = str(method_id)

        ordered_datasets = sorted(rows.groupby("data_id", sort=False), key=lambda x: x[0])

        points = []
        running_parts = []
        step = 0

        for data_id, dataset_rows in ordered_datasets:
            if dataset_rows.empty:
                continue

            running_parts.append(dataset_rows)
            running_rows = pd.concat(running_parts, ignore_index=True)
            step += 1

            cumulative_rv = compute_retrodictive_validity(running_rows)

            points.append({
                "step": step,
                "data_id": int(data_id),
                "filename": first_non_null(dataset_rows, "filename"),
                "contrast": compute_dataset_contrast(dataset_rows),
                "cumulative_mean_contrast": cumulative_rv,
            })

        result[method_name] = pd.DataFrame(points)

    return result


def get_scores_by_lab_data(
        df: pd.DataFrame,
        method_id: int,
        task_id: int
    ) -> pd.DataFrame:
    """
    Main-dataframe equivalent to getScoresByLabData(int methodId, int taskId, int challengeId)

    Java version used Output.value.
    Here we use parsed_value from the main dataframe.

    Grouping:
        lab_head -> data_id
    For each dataset:
        contrast = first parsed_value by output_order - second parsed_value by output_order
    Then:
        mean contrast by lab
    """
    rows = df[
        (df["method_id"] == method_id) &
        (df["task_id"] == task_id)
    ].copy()

    required = {"lab_head", "data_id", "parsed_value", "output_order"}
    missing = required - set(rows.columns)
    if rows.empty or missing:
        return pd.DataFrame(columns=["lab_head", "score", "ci_low", "ci_high", "n_datasets"])

    rows = rows.dropna(subset=["lab_head", "data_id"])
    if rows.empty:
        return pd.DataFrame(columns=["lab_head", "score", "ci_low", "ci_high", "n_datasets"])

    out = []

    for lab_head, lab_df in rows.groupby("lab_head", sort=False):
        contrasts = []

        for _, dataset_rows in lab_df.groupby("data_id", sort=False):
            ordered = dataset_rows.sort_values("output_order")
            if len(ordered) < 2:
                continue

            cs_plus = ordered.iloc[0]["parsed_value"]
            cs_minus = ordered.iloc[1]["parsed_value"]
            contrasts.append(float(cs_plus - cs_minus))

        if not contrasts:
            continue

        n = len(contrasts)
        mean = float(np.mean(contrasts))

        ci_low, ci_high = (None, None)
        if n >= 3:
            ci_low, ci_high = compute_mean_ci(contrasts)

        out.append({
            "lab_head": lab_head,
            "score": mean,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "n_datasets": n,
        })

    result = pd.DataFrame(out)
    if result.empty:
        return result

    return result.sort_values(["score", "lab_head"], ascending=[False, True], na_position="last").reset_index(drop=True)


def get_last_updated_score_by_task_id(
        df: pd.DataFrame,
        task_id: int
    ) -> pd.DataFrame:
    """
    Equivalent to getLastUpdatedScoreByTaskId(int taskId)
    """
    rankings = get_best_scores_by_task_id(df, task_id)
    if rankings.empty:
        return pd.DataFrame(
            columns=[
                "method_id", "task_id", "score", "ci_low",
                "ci_high", "n_datasets", "last_updated"
            ]
        )

    best = rankings.iloc[0]

    rows = df[
        (df["task_id"] == task_id) &
        (df["method_id"] == best["method_id"])
    ]

    last_updated = None
    if "created_at" in rows.columns and not rows.empty:
        last_updated = pd.to_datetime(rows["created_at"], errors="coerce").max()

    return pd.DataFrame([{
        "method_id": int(best["method_id"]),
        "task_id": int(task_id),
        "score": best["score"],
        "ci_low": best["ci_low"],
        "ci_high": best["ci_high"],
        "n_datasets": int(best["n_datasets"]),
        "last_updated": last_updated,
    }])


# Optional convenience wrapper
class ScoreServiceDataFrame:
    """
    Thin wrapper around the standalone functions.
    All outputs are DataFrames, except methods that naturally return
    multiple tables, which return dict[str, DataFrame].
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def compute_retrodictive_validity(
            self,
            rows: Optional[pd.DataFrame] = None
        ) -> Optional[float]:
        return compute_retrodictive_validity(self.df if rows is None else rows)

    def get_scores_by_method(
            self,
            method_id: int
        ) -> pd.DataFrame:
        return get_scores_by_method(self.df, method_id)

    def get_best_scores_by_task_id(
            self,
            task_id: int
        ) -> pd.DataFrame:
        return get_best_scores_by_task_id(self.df, task_id)

    def get_best_scores_by_task_id_grouped_by_observable(
            self,
            task_id: int
        ) -> Dict[str, pd.DataFrame]:
        return get_best_scores_by_task_id_grouped_by_observable(self.df, task_id)

    def get_rv_by_lab(
            self,
            task_id: int
        ) -> pd.DataFrame:
        return get_rv_by_lab(self.df, task_id)

    def get_cumulative_contrast_by_method(
            self,
            task_id: int
        ) -> Dict[str, pd.DataFrame]:
        return get_cumulative_contrast_by_method(self.df, task_id)

    def get_scores_by_lab_data(
            self,
            method_id: int,
            task_id: int
        ) -> pd.DataFrame:
        return get_scores_by_lab_data(self.df, method_id, task_id)

    def get_last_updated_score_by_task_id(
            self,
            task_id: int
        ) -> pd.DataFrame:
        return get_last_updated_score_by_task_id(self.df, task_id)
    
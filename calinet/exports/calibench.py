# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import math
from typing import Optional, Dict, Any, Iterable, Tuple

import numpy as np
import pandas as pd


# Math helpers
def fisher_z(r: float) -> float:
    """
    Apply Fisher's z-transformation to a correlation coefficient.

    This function transforms a Pearson correlation coefficient ``r`` into
    Fisher's z-space, which stabilizes variance and makes the distribution
    more approximately normal for inferential statistics.

    Parameters
    ----------
    r : float
        Pearson correlation coefficient. Expected to be in the range
        ``[-1, 1]``.

    Returns
    -------
    z : float
        Fisher z-transformed value. Returns:

        - ``inf`` if ``r >= 1.0``
        - ``-inf`` if ``r <= -1.0``
        - otherwise ``0.5 * log((1 + r) / (1 - r))``

    Raises
    ------
    TypeError
        Raised if ``r`` is not a numeric value.

    Notes
    -----
    Fisher's z-transformation is defined as:

    - z = 0.5 * ln((1 + r) / (1 - r))

    It is commonly used when averaging correlations or performing hypothesis
    testing, because it converts correlation coefficients into a scale where
    standard statistical techniques are more appropriate.

    Values of ``r`` exactly equal to ``-1`` or ``1`` would lead to division by
    zero in the transformation. This implementation returns signed infinity
    for those boundary cases.

    Examples
    --------
    Transform a positive correlation.

    >>> fisher_z(0.5)
    0.5493061443340549

    Transform a negative correlation.

    >>> fisher_z(-0.5)
    -0.5493061443340549

    Handle boundary values.

    >>> fisher_z(1.0)
    inf
    >>> fisher_z(-1.0)
    -inf
    """
    if r >= 1.0:
        return float("inf")
    if r <= -1.0:
        return float("-inf")
    return 0.5 * math.log((1.0 + r) / (1.0 - r))


def inv_fisher_z(z: float) -> float:
    """
    Apply the inverse Fisher z-transformation to recover a correlation value.

    This function converts a Fisher z-transformed value back into a Pearson
    correlation coefficient using the hyperbolic tangent function.

    Parameters
    ----------
    z : float
        Fisher z-transformed value.

    Returns
    -------
    r : float
        Pearson correlation coefficient in the range ``[-1, 1]``.

    Raises
    ------
    TypeError
        Raised if ``z`` is not a numeric value.

    Notes
    -----
    The inverse Fisher transformation is defined as:

    - r = tanh(z)

    This operation maps values from the real line back to the bounded
    correlation interval ``[-1, 1]``.

    Extremely large positive or negative values of ``z`` will asymptotically
    approach ``1`` or ``-1`` due to the properties of the hyperbolic tangent
    function.

    Examples
    --------
    Recover a correlation from z-space.

    >>> inv_fisher_z(0.5493061443340549)
    0.5

    Recover a negative correlation.

    >>> inv_fisher_z(-0.5493061443340549)
    -0.5

    Handle large values.

    >>> inv_fisher_z(10)
    0.9999999958776927
    >>> inv_fisher_z(-10)
    -0.9999999958776927
    """
    return math.tanh(z)


def pearson_correlation(x: Iterable[Any], y: Iterable[Any]) -> float:
    """
    Compute the Pearson correlation coefficient between two numeric sequences.

    This function calculates the linear correlation (Pearson's r) between two
    equal-length iterables by converting them to NumPy arrays and applying the
    standard covariance normalization formula.

    Parameters
    ----------
    x : iterable
        First sequence of numeric values. Any iterable is accepted (e.g., list,
        NumPy array, pandas Series). Values are cast to ``float``.
    y : iterable
        Second sequence of numeric values. Must have the same length as ``x``.
        Values are cast to ``float``.

    Returns
    -------
    r : float
        Pearson correlation coefficient in the range ``[-1, 1]``. Returns
        ``nan`` if:

        - the input sequences have different lengths
        - the number of observations is less than 2
        - either sequence has zero variance (division by zero)

    Raises
    ------
    TypeError
        Raised if elements of ``x`` or ``y`` cannot be converted to float.

    Notes
    -----
    The Pearson correlation coefficient is defined as:

    - covariance of ``x`` and ``y`` divided by the product of their standard
      deviations

    This implementation performs the computation manually using NumPy
    operations:

    1. subtract means from both sequences
    2. compute the dot product of deviations (numerator)
    3. compute the product of squared deviations (denominator)

    If the denominator is zero (i.e., one of the inputs has zero variance),
    the function returns ``nan``.

    This function does not rely on ``scipy`` or ``pandas`` and is suitable for
    lightweight use cases where a dependency-free implementation is desired.

    Examples
    --------
    Compute correlation between two lists.

    >>> pearson_correlation([1, 2, 3], [1, 2, 3])
    1.0

    Compute negative correlation.

    >>> pearson_correlation([1, 2, 3], [3, 2, 1])
    -1.0

    Handle mismatched lengths.

    >>> pearson_correlation([1, 2], [1])
    nan

    Handle zero variance.

    >>> pearson_correlation([1, 1, 1], [2, 3, 4])
    nan

    Use with pandas Series.

    >>> s1 = pd.Series([1, 2, 3])
    >>> s2 = pd.Series([2, 4, 6])
    >>> pearson_correlation(s1, s2)
    1.0
    """
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


def compute_correlation_ci(
        r: float,
        n: int
    ) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute a 95% confidence interval for a Pearson correlation coefficient.

    This function estimates the confidence interval of a correlation
    coefficient using Fisher's z-transformation. The interval is computed in
    z-space and then transformed back to the correlation scale.

    Parameters
    ----------
    r : float
        Pearson correlation coefficient. Expected to be in the range
        ``[-1, 1]``. If ``NaN``, no interval is computed.
    n : int
        Sample size used to compute the correlation.

    Returns
    -------
    low : float or None
        Lower bound of the 95% confidence interval. Returns ``None`` if the
        interval cannot be computed.
    high : float or None
        Upper bound of the 95% confidence interval. Returns ``None`` if the
        interval cannot be computed.

    Raises
    ------
    TypeError
        Raised if ``r`` or ``n`` is not numeric.
    ValueError
        Raised if downstream transformations fail due to invalid values.

    Notes
    -----
    The confidence interval is computed as follows:

    1. transform ``r`` to Fisher z using ``fisher_z``
    2. compute the standard error: ``SE = 1 / sqrt(n - 3)``
    3. compute the z-interval using a critical value of ``1.96`` (approximate
       95% confidence)
    4. transform the bounds back using ``inv_fisher_z``

    The resulting interval is clipped to the valid correlation range
    ``[-1, 1]``.

    A minimum sample size of ``n >= 4`` is required because the standard error
    is undefined for smaller samples. If ``n < 4`` or ``r`` is ``NaN``, the
    function returns ``(None, None)``.

    Examples
    --------
    Compute a confidence interval for a moderate correlation.

    >>> compute_correlation_ci(0.5, 30)
    (0.190..., 0.728...)

    Handle small sample sizes.

    >>> compute_correlation_ci(0.5, 3)
    (None, None)

    Handle NaN input.

    >>> compute_correlation_ci(float("nan"), 30)
    (None, None)

    Strong correlation with larger sample.

    >>> compute_correlation_ci(0.9, 50)
    (0.83..., 0.94...)
    """
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


def compute_mean_ci(
        values: Iterable[Any]
    ) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute an approximate confidence interval for the mean of a sample.

    This function estimates a confidence interval around the sample mean using
    the standard error and an approximate critical value based on sample size.
    For small samples, predefined t-values are used; for larger samples, a
    normal approximation is applied.

    Parameters
    ----------
    values : iterable
        Sequence of numeric values (e.g., list, NumPy array, pandas Series).
        Values are cast to ``float``.

    Returns
    -------
    low : float or None
        Lower bound of the confidence interval. Returns ``None`` if the sample
        size is too small.
    high : float or None
        Upper bound of the confidence interval. Returns ``None`` if the sample
        size is too small.

    Raises
    ------
    TypeError
        Raised if elements of ``values`` cannot be converted to float.

    Notes
    -----
    The interval is computed as:

    - mean ± t * SE

    where:

    - mean is the sample mean
    - SE (standard error) = SD / sqrt(n)
    - SD is the sample standard deviation (using ``n - 1`` in the denominator)

    The critical value ``t`` depends on sample size:

    - n = 3 → 4.303
    - n = 4 → 3.182
    - n = 5 → 2.776
    - n ≥ 6 → 1.96 (normal approximation)

    If ``n < 3``, the function returns ``(None, None)`` because the variance
    estimate is unreliable.

    This implementation uses a simplified lookup rather than computing exact
    t-distribution quantiles.

    Examples
    --------
    Compute a confidence interval for a small sample.

    >>> compute_mean_ci([1, 2, 3])
    (approx. -0.48, 4.48)

    Larger sample with normal approximation.

    >>> compute_mean_ci([1, 2, 3, 4, 5, 6])
    (approx. 2.16, 4.84)

    Handle insufficient data.

    >>> compute_mean_ci([1, 2])
    (None, None)

    Use with a pandas Series.

    >>> s = pd.Series([10, 12, 14, 16, 18])
    >>> compute_mean_ci(s)
    (approx. 11.2, 16.8)
    """
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
    """
    Count the number of distinct dataset identifiers in a dataframe.

    Parameters
    ----------
    rows : pandas.DataFrame
        Input dataframe expected to contain a ``data_id`` column identifying
        datasets.

    Returns
    -------
    n_datasets : int
        Number of unique non-null dataset identifiers in ``rows``. Returns
        ``0`` if ``rows`` is ``None`` or empty.

    Raises
    ------
    KeyError
        Raised if ``rows`` is not empty and does not contain ``data_id``.

    Notes
    -----
    Missing values in ``data_id`` are ignored when counting unique datasets.

    This helper is used by scoring and ranking functions to report dataset
    counts alongside retrodictive validity metrics.

    Examples
    --------
    Count unique datasets.

    >>> rows = pd.DataFrame({"data_id": [1, 1, 2, 3, np.nan]})
    >>> count_distinct_datasets(rows)
    3

    Handle empty input.

    >>> count_distinct_datasets(pd.DataFrame())
    0
    """
    if rows is None or rows.empty:
        return 0
    return int(rows["data_id"].dropna().nunique())


def first_non_null(
        rows: pd.DataFrame,
        col: str
    ) -> Optional[Any]:
    """
    Return the first non-missing value from a dataframe column.

    Parameters
    ----------
    rows : pandas.DataFrame
        Input dataframe to search.
    col : str
        Column name from which to retrieve the first non-null value.

    Returns
    -------
    value : Any or None
        First non-null value in ``col``. Returns ``None`` if the column is
        missing or if all values in the column are null.

    Notes
    -----
    This helper preserves the original row order of ``rows`` and simply
    returns the first available non-null entry.

    It is commonly used to propagate representative metadata such as
    ``method_name`` or ``filename`` into summary outputs.

    Examples
    --------
    Return the first non-null value.

    >>> rows = pd.DataFrame({"method_name": [None, "Method A", "Method B"]})
    >>> first_non_null(rows, "method_name")
    'Method A'

    Return ``None`` if the column is absent.

    >>> first_non_null(rows, "missing_col") is None
    True
    """
    if col not in rows.columns:
        return None
    s = rows[col].dropna()
    return None if s.empty else s.iloc[0]


# Core score logic
def compute_retrodictive_validity(
        rows: pd.DataFrame,
        score_col: str="parsed_value",
        intended_col: str="intended_value",
        subject_col: str="data_id"
    ) -> Optional[float]:
    """
    Compute retrodictive validity from subject-wise centered scores.

    This function estimates retrodictive validity by mean-centering scores
    within each subject, pooling the centered values across subjects, and
    computing the Pearson correlation between those centered scores and the
    corresponding intended values.

    Parameters
    ----------
    rows : pandas.DataFrame
        Input dataframe containing one or more rows per subject. The dataframe
        must provide columns for the observed score, intended value, and
        subject identifier as specified by ``score_col``, ``intended_col``,
        and ``subject_col``.
    score_col : str, default="parsed_value"
        Column in ``rows`` containing the observed or parsed scores.
    intended_col : str, default="intended_value"
        Column in ``rows`` containing the intended target values against which
        retrodictive validity is computed.
    subject_col : str, default="data_id"
        Column in ``rows`` identifying subjects or datasets. Scores are
        centered separately within each subject group.

    Returns
    -------
    r : float or None
        Retrodictive validity coefficient. Returns ``None`` if the input is
        missing, too small, contains no usable rows after filtering, produces
        too few centered observations, or yields an undefined correlation.

    Raises
    ------
    KeyError
        Raised if ``score_col``, ``intended_col``, or ``subject_col`` is
        missing from ``rows``.
    TypeError
        Raised if downstream numeric operations or correlation computation
        reject the provided values.

    Notes
    -----
    This function is a Python equivalent of the Java method
    ``computeRetrodictiveValidity(List<DatasetTaskOutputDTO> rows)``.

    Processing proceeds as follows:

    1. return ``None`` if ``rows`` is ``None`` or has fewer than 3 rows
    2. drop rows with missing values in ``score_col`` or ``intended_col``
    3. group remaining rows by ``subject_col``
    4. ignore subject groups with fewer than 2 rows
    5. compute the mean score within each subject
    6. subtract the subject mean from each score to obtain centered values
    7. pool centered scores and intended values across all eligible subjects
    8. compute Pearson correlation between intended values and centered scores

    Subject-wise centering removes between-subject differences in absolute
    score level, so the resulting coefficient reflects whether within-subject
    score variation aligns with intended values.

    If the final pooled vectors have fewer than 3 elements, have unequal
    lengths, or the correlation is undefined, the function returns ``None``.

    Examples
    --------
    Compute retrodictive validity for two subjects with two conditions each.

    >>> rows = pd.DataFrame({
    ...     "data_id": ["sub-01", "sub-01", "sub-02", "sub-02"],
    ...     "parsed_value": [0.8, 0.2, 0.7, 0.1],
    ...     "intended_value": [1, 0, 1, 0]
    ... })
    >>> compute_retrodictive_validity(rows)
    1.0

    Ignore subjects with fewer than two rows.

    >>> rows = pd.DataFrame({
    ...     "data_id": ["sub-01", "sub-01", "sub-02"],
    ...     "parsed_value": [0.8, 0.2, 0.7],
    ...     "intended_value": [1, 0, 1]
    ... })
    >>> compute_retrodictive_validity(rows) is None
    True

    Use custom column names.

    >>> rows = pd.DataFrame({
    ...     "subject": ["s1", "s1", "s2", "s2"],
    ...     "score": [2.0, 1.0, 3.0, 1.5],
    ...     "target": [1, 0, 1, 0]
    ... })
    >>> compute_retrodictive_validity(
    ...     rows,
    ...     score_col="score",
    ...     intended_col="target",
    ...     subject_col="subject"
    ... )
    1.0

    Return ``None`` when all usable rows are missing.

    >>> rows = pd.DataFrame({
    ...     "data_id": ["sub-01", "sub-01", "sub-02"],
    ...     "parsed_value": [np.nan, np.nan, np.nan],
    ...     "intended_value": [1, 0, 1]
    ... })
    >>> compute_retrodictive_validity(rows) is None
    True
    """
    if rows is None or len(rows) < 3:
        return None

    filtered = rows.dropna(subset=[score_col, intended_col])
    if filtered.empty:
        return None

    intended = []
    parsed_centered = []

    for _, subject_rows in filtered.groupby(subject_col, sort=False):
        if len(subject_rows) < 2:
            continue

        mean_parsed = subject_rows[score_col].mean()
        if pd.isna(mean_parsed):
            continue

        centered = subject_rows[score_col] - mean_parsed
        parsed_centered.extend(centered.tolist())
        intended.extend(subject_rows[intended_col].tolist())

    if len(parsed_centered) < 3 or len(parsed_centered) != len(intended):
        return None

    r = pearson_correlation(intended, parsed_centered)
    return None if np.isnan(r) else float(r)


def compute_dataset_contrast(rows: pd.DataFrame) -> Optional[float]:
    """
    Compute a within-dataset contrast from ordered parsed values.

    This function sorts rows by ``output_order`` and computes the contrast as
    the first ``parsed_value`` minus the second ``parsed_value``.

    Parameters
    ----------
    rows : pandas.DataFrame
        Input dataframe for a single dataset. The dataframe must contain
        ``output_order`` and ``parsed_value`` columns.

    Returns
    -------
    contrast : float or None
        Contrast value computed as:

        - first ordered ``parsed_value`` minus second ordered ``parsed_value``

        Returns ``None`` if ``rows`` is ``None``, contains fewer than two rows,
        or does not provide at least two non-missing ordered values.

    Raises
    ------
    KeyError
        Raised if required columns are missing from ``rows``.

    Notes
    -----
    This function mirrors the Java logic:

    - order rows by ``output_order``
    - compute ``contrast = first parsed_value - second parsed_value``

    Rows with missing ``output_order`` or ``parsed_value`` are dropped before
    sorting and contrast calculation.

    Examples
    --------
    Compute a simple contrast.

    >>> rows = pd.DataFrame({
    ...     "output_order": [2, 1],
    ...     "parsed_value": [0.2, 0.8]
    ... })
    >>> compute_dataset_contrast(rows)
    0.6000000000000001

    Return ``None`` when fewer than two valid rows remain.

    >>> rows = pd.DataFrame({
    ...     "output_order": [1],
    ...     "parsed_value": [0.8]
    ... })
    >>> compute_dataset_contrast(rows) is None
    True
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
        task_id: int,
        rows: pd.DataFrame
    ) -> Dict[str, Any]:
    """
    Build a standardized score summary row for one method-task combination.

    Parameters
    ----------
    method_id : int
        Method identifier for the score row.
    task_id : int
        Task identifier for the score row.
    rows : pandas.DataFrame
        Input rows used to compute retrodictive validity and metadata counts.

    Returns
    -------
    result : dict
        Dictionary containing:

        - ``method_id``: integer method identifier
        - ``task_id``: integer task identifier
        - ``score``: retrodictive validity or ``None``
        - ``ci_low``: lower confidence bound or ``None``
        - ``score_ci_high``: upper confidence bound or ``None``
        - ``n_datasets``: number of distinct datasets

    Notes
    -----
    If ``rows`` is empty or ``None``, the score and confidence interval fields
    are returned as ``None`` and ``n_datasets`` is ``0``.

    Confidence intervals are computed only when retrodictive validity is
    defined and the number of rows is at least 4.

    This helper is primarily used by higher-level dataframe-returning service
    functions.

    Examples
    --------
    Build a score row from valid inputs.

    >>> rows = pd.DataFrame({
    ...     "data_id": [1, 1, 2, 2],
    ...     "parsed_value": [0.8, 0.2, 0.7, 0.1],
    ...     "intended_value": [1, 0, 1, 0]
    ... })
    >>> out = build_score_row(method_id=10, task_id=3, rows=rows)
    >>> out["method_id"], out["task_id"]
    (10, 3)

    Handle empty input.

    >>> build_score_row(10, 3, pd.DataFrame())["n_datasets"]
    0
    """

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
    Compute task-level scores for a single method.

    Parameters
    ----------
    df : pandas.DataFrame
        Main score dataframe containing at least ``method_id`` and ``task_id``
        columns, along with the fields required by ``build_score_row`` and
        ``compute_retrodictive_validity``.
    method_id : int
        Method identifier to filter on.

    Returns
    -------
    result : pandas.DataFrame
        Dataframe with one row per task for the selected method. Columns are:

        - ``method_id``
        - ``task_id``
        - ``score``
        - ``ci_low``
        - ``score_ci_high``
        - ``n_datasets``

        If no rows match ``method_id``, an empty dataframe with those columns
        is returned.

    Raises
    ------
    KeyError
        Raised if required columns such as ``method_id`` or ``task_id`` are
        missing from ``df``.

    Notes
    -----
    This function is the dataframe equivalent of
    ``getScoresByMethod(int methodId)``.

    For each unique task associated with the selected method, it builds a
    summary row using ``build_score_row``.

    Examples
    --------
    Get scores for one method across tasks.

    >>> df = pd.DataFrame({
    ...     "method_id": [1, 1, 1, 1],
    ...     "task_id": [10, 10, 20, 20],
    ...     "data_id": [1, 1, 2, 2],
    ...     "parsed_value": [0.8, 0.2, 0.7, 0.1],
    ...     "intended_value": [1, 0, 1, 0]
    ... })
    >>> out = get_scores_by_method(df, 1)
    >>> sorted(out["task_id"].tolist())
    [10, 20]
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
    Rank methods for a task by retrodictive validity.

    Parameters
    ----------
    df : pandas.DataFrame
        Main score dataframe containing at least ``task_id``, ``method_id``,
        and the columns required by ``compute_retrodictive_validity``.
    task_id : int
        Task identifier to evaluate.

    Returns
    -------
    result : pandas.DataFrame
        Ranking dataframe with one row per method and columns:

        - ``method_id``
        - ``method_name``
        - ``task_name``
        - ``method_description``
        - ``n_datasets``
        - ``score``
        - ``ci_low``
        - ``ci_high``
        - ``lab_head``
        - ``observable_name``

        Rows are sorted by descending ``score`` and ascending ``method_id``.
        If the task has no matching rows, an empty dataframe with the expected
        columns is returned.

    Raises
    ------
    KeyError
        Raised if required columns are missing from ``df``.

    Notes
    -----
    This function is the dataframe equivalent of
    ``getBestScoresByTaskId(int taskId)``.

    Retrodictive validity is computed separately for each method. Methods with
    undefined retrodictive validity are excluded from the ranking.

    Confidence intervals are computed only when a method has at least 4 rows.

    Examples
    --------
    Rank methods for a task.

    >>> df = pd.DataFrame({
    ...     "task_id": [1, 1, 1, 1, 1, 1, 1, 1],
    ...     "method_id": [10, 10, 10, 10, 20, 20, 20, 20],
    ...     "method_name": ["A", "A", "A", "A", "B", "B", "B", "B"],
    ...     "data_id": [1, 1, 2, 2, 1, 1, 2, 2],
    ...     "parsed_value": [0.8, 0.2, 0.7, 0.1, 0.6, 0.4, 0.55, 0.45],
    ...     "intended_value": [1, 0, 1, 0, 1, 0, 1, 0]
    ... })
    >>> out = get_best_scores_by_task_id(df, 1)
    >>> "score" in out.columns
    True
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
    Rank methods for a task separately within each observable.

    Parameters
    ----------
    df : pandas.DataFrame
        Main score dataframe containing task, method, and scoring columns. If
        ``observable_name`` is missing, it is created with the value
        ``"Unknown"``.
    task_id : int
        Task identifier to evaluate.

    Returns
    -------
    result : dict of {str: pandas.DataFrame}
        Dictionary mapping each observable name to a ranking dataframe. Each
        ranking dataframe contains one row per method with columns such as:

        - ``method_id``
        - ``method_name``
        - ``task_name``
        - ``method_description``
        - ``observable_name``
        - ``n_datasets``
        - ``score``
        - ``ci_low``
        - ``ci_high``

        The ranking within each observable is sorted by descending ``score``
        and ascending ``method_id``.

    Raises
    ------
    KeyError
        Raised if required scoring columns are missing from ``df``.

    Notes
    -----
    This function is the dataframe equivalent of
    ``getBestScoresByTaskIdGroupedByObservable(int taskId)``.

    Missing observable names are filled with ``"Unknown"`` before grouping.

    Methods with undefined retrodictive validity within an observable are
    omitted from that observable's ranking dataframe.

    Examples
    --------
    Group rankings by observable.

    >>> df = pd.DataFrame({
    ...     "task_id": [1, 1, 1, 1],
    ...     "method_id": [10, 10, 20, 20],
    ...     "method_name": ["A", "A", "B", "B"],
    ...     "observable_name": ["SCR", "SCR", "SCR", "SCR"],
    ...     "data_id": [1, 1, 2, 2],
    ...     "parsed_value": [0.8, 0.2, 0.7, 0.1],
    ...     "intended_value": [1, 0, 1, 0]
    ... })
    >>> out = get_best_scores_by_task_id_grouped_by_observable(df, 1)
    >>> list(out.keys())
    ['SCR']
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
    Compute retrodictive validity by lab for the best-performing method.

    Parameters
    ----------
    df : pandas.DataFrame
        Main score dataframe containing at least ``task_id``, ``method_id``,
        ``lab_head``, and the columns required for retrodictive validity.
    task_id : int
        Task identifier to evaluate.

    Returns
    -------
    result : pandas.DataFrame
        Dataframe with one row per lab and columns:

        - ``lab_head``
        - ``score``
        - ``ci_low``
        - ``ci_high``
        - ``n_datasets``

        Rows are sorted by descending ``score`` and ascending ``lab_head``.
        If no valid task or lab information is available, an empty dataframe
        with those columns is returned.

    Raises
    ------
    KeyError
        Raised if required columns are missing from ``df``.

    Notes
    -----
    This function is the dataframe equivalent of ``getRvByLab(int taskId)``.

    Processing proceeds as follows:

    1. filter rows for the requested task
    2. compute retrodictive validity for each method
    3. identify the best-performing method
    4. compute retrodictive validity separately for each non-empty ``lab_head``
       within that best method

    Confidence intervals are computed per lab only when the corresponding lab
    subset has at least 4 rows.

    Examples
    --------
    Compute RV by lab for a task.

    >>> out = get_rv_by_lab(df, task_id=1)
    >>> isinstance(out, pd.DataFrame)
    True
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
    Compute cumulative retrodictive validity trajectories by method.

    Parameters
    ----------
    df : pandas.DataFrame
        Main score dataframe containing rows for one or more methods and
        datasets. It must provide the columns required by
        ``compute_retrodictive_validity`` and ``compute_dataset_contrast``.
    task_id : int
        Task identifier to evaluate.

    Returns
    -------
    result : dict of {str: pandas.DataFrame}
        Dictionary mapping method name to a dataframe with columns:

        - ``step``
        - ``data_id``
        - ``filename``
        - ``contrast``
        - ``cumulative_mean_contrast``

        Despite the column name, ``cumulative_mean_contrast`` stores the
        cumulative retrodictive validity after progressively adding datasets.

    Raises
    ------
    KeyError
        Raised if required columns such as ``method_id`` or ``data_id`` are
        missing from ``df``.

    Notes
    -----
    This function is the dataframe equivalent of
    ``getCumulativeContrastByMethod(int taskId)``.

    For each method, datasets are ordered by ``data_id``. At each step, the
    function concatenates all datasets seen so far, computes cumulative
    retrodictive validity, and records the current dataset contrast.

    The field name ``cumulative_mean_contrast`` is preserved to match the Java
    service output, although the stored value is cumulative retrodictive
    validity rather than a mean contrast.

    Examples
    --------
    Compute cumulative curves by method.

    >>> out = get_cumulative_contrast_by_method(df, task_id=1)
    >>> isinstance(out, dict)
    True
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
    Compute mean dataset contrast by lab for a given method and task.

    Parameters
    ----------
    df : pandas.DataFrame
        Main score dataframe containing at least ``method_id``, ``task_id``,
        ``lab_head``, ``data_id``, ``parsed_value``, and ``output_order``.
    method_id : int
        Method identifier to filter on.
    task_id : int
        Task identifier to filter on.

    Returns
    -------
    result : pandas.DataFrame
        Dataframe with one row per lab and columns:

        - ``lab_head``
        - ``score``
        - ``ci_low``
        - ``ci_high``
        - ``n_datasets``

        Here ``score`` is the mean within-lab contrast across datasets.
        Results are sorted by descending ``score`` and ascending ``lab_head``.
        If no usable rows remain, an empty dataframe with those columns is
        returned.

    Raises
    ------
    KeyError
        Raised if required columns are missing from ``df``.

    Notes
    -----
    This function is the main-dataframe equivalent of
    ``getScoresByLabData(int methodId, int taskId, int challengeId)``.

    Processing proceeds as follows:

    1. filter rows by ``method_id`` and ``task_id``
    2. group by ``lab_head``, then by ``data_id``
    3. sort each dataset by ``output_order``
    4. compute dataset contrast as first ``parsed_value`` minus second
       ``parsed_value``
    5. average contrasts within each lab

    Confidence intervals for the lab mean contrast are computed with
    ``compute_mean_ci`` only when a lab has at least 3 dataset contrasts.

    Examples
    --------
    Compute mean contrast by lab.

    >>> out = get_scores_by_lab_data(df, method_id=10, task_id=1)
    >>> "lab_head" in out.columns
    True
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
    Return the top-ranked method for a task together with its latest update time.

    Parameters
    ----------
    df : pandas.DataFrame
        Main score dataframe containing the fields required by
        ``get_best_scores_by_task_id``. If available, ``created_at`` is used
        to derive the latest update timestamp.
    task_id : int
        Task identifier to evaluate.

    Returns
    -------
    result : pandas.DataFrame
        Single-row dataframe containing:

        - ``method_id``
        - ``task_id``
        - ``score``
        - ``ci_low``
        - ``ci_high``
        - ``n_datasets``
        - ``last_updated``

        If no ranked method is available for the task, an empty dataframe with
        those columns is returned.

    Raises
    ------
    KeyError
        Raised if required ranking columns are missing from ``df``.

    Notes
    -----
    This function is the dataframe equivalent of
    ``getLastUpdatedScoreByTaskId(int taskId)``.

    It first calls ``get_best_scores_by_task_id`` to identify the top-ranked
    method for the task, then filters ``df`` to that method-task combination
    and computes the maximum parsed ``created_at`` timestamp.

    If ``created_at`` is absent or all values are invalid, ``last_updated`` is
    returned as ``None`` or ``NaT`` depending on pandas conversion behavior.

    Examples
    --------
    Get the best score and latest update for a task.

    >>> out = get_last_updated_score_by_task_id(df, task_id=1)
    >>> list(out.columns)
    ['method_id', 'task_id', 'score', 'ci_low', 'ci_high', 'n_datasets', 'last_updated']
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
    Thin dataframe-backed wrapper around the standalone score service functions.

    Parameters
    ----------
    df : pandas.DataFrame
        Main score dataframe copied into the service instance. The dataframe
        should contain the columns required by the methods you intend to call.

    Attributes
    ----------
    df : pandas.DataFrame
        Internal copy of the input dataframe.

    Notes
    -----
    This class provides a convenience object-oriented interface over the
    module-level functions.

    All methods delegate directly to the standalone implementations using the
    stored dataframe, except ``compute_retrodictive_validity``, which may also
    accept an explicit dataframe override.

    Methods returning grouped outputs use ``dict[str, pandas.DataFrame]``;
    all others return either a single dataframe or a scalar retrodictive
    validity value.

    Examples
    --------
    Create a service and compute scores.

    >>> service = ScoreServiceDataFrame(df)
    >>> scores = service.get_scores_by_method(method_id=1)
    >>> isinstance(scores, pd.DataFrame)
    True

    Compute retrodictive validity on the stored dataframe.

    >>> rv = service.compute_retrodictive_validity()
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def compute_retrodictive_validity(
            self,
            rows: Optional[pd.DataFrame]=None
        ) -> Optional[float]:
        """
        Compute retrodictive validity using the stored dataframe or an override.

        Parameters
        ----------
        rows : pandas.DataFrame or None, optional
            Optional dataframe to use instead of ``self.df``.

        Returns
        -------
        r : float or None
            Retrodictive validity coefficient returned by the module-level
            ``compute_retrodictive_validity`` function.

        Notes
        -----
        If ``rows`` is ``None``, this method uses the dataframe stored on the
        service instance.
        """
        return compute_retrodictive_validity(self.df if rows is None else rows)

    def get_scores_by_method(
            self,
            method_id: int
        ) -> pd.DataFrame:
        """
        Return task-level score summaries for a single method.

        Parameters
        ----------
        method_id : int
            Method identifier to evaluate.

        Returns
        -------
        result : pandas.DataFrame
            Output of ``get_scores_by_method(self.df, method_id)``.
        """
        return get_scores_by_method(self.df, method_id)

    def get_best_scores_by_task_id(
            self,
            task_id: int
        ) -> pd.DataFrame:
        """
        Return method rankings for a task.

        Parameters
        ----------
        task_id : int
            Task identifier to evaluate.

        Returns
        -------
        result : pandas.DataFrame
            Output of ``get_best_scores_by_task_id(self.df, task_id)``.
        """
        return get_best_scores_by_task_id(self.df, task_id)

    def get_best_scores_by_task_id_grouped_by_observable(
            self,
            task_id: int
        ) -> Dict[str, pd.DataFrame]:
        """
        Return observable-specific method rankings for a task.

        Parameters
        ----------
        task_id : int
            Task identifier to evaluate.

        Returns
        -------
        result : dict of {str: pandas.DataFrame}
            Output of
            ``get_best_scores_by_task_id_grouped_by_observable(self.df, task_id)``.
        """
        return get_best_scores_by_task_id_grouped_by_observable(self.df, task_id)

    def get_rv_by_lab(
            self,
            task_id: int
        ) -> pd.DataFrame:
        """
        Return retrodictive validity by lab for the best method on a task.

        Parameters
        ----------
        task_id : int
            Task identifier to evaluate.

        Returns
        -------
        result : pandas.DataFrame
            Output of ``get_rv_by_lab(self.df, task_id)``.
        """
        return get_rv_by_lab(self.df, task_id)

    def get_cumulative_contrast_by_method(
            self,
            task_id: int
        ) -> Dict[str, pd.DataFrame]:
        """
        Return cumulative retrodictive validity trajectories by method.

        Parameters
        ----------
        task_id : int
            Task identifier to evaluate.

        Returns
        -------
        result : dict of {str: pandas.DataFrame}
            Output of ``get_cumulative_contrast_by_method(self.df, task_id)``.
        """
        return get_cumulative_contrast_by_method(self.df, task_id)

    def get_scores_by_lab_data(
            self,
            method_id: int,
            task_id: int
        ) -> pd.DataFrame:
        """
        Return mean dataset contrast by lab for a method-task combination.

        Parameters
        ----------
        method_id : int
            Method identifier to evaluate.
        task_id : int
            Task identifier to evaluate.

        Returns
        -------
        result : pandas.DataFrame
            Output of ``get_scores_by_lab_data(self.df, method_id, task_id)``.
        """
        return get_scores_by_lab_data(self.df, method_id, task_id)

    def get_last_updated_score_by_task_id(
            self,
            task_id: int
        ) -> pd.DataFrame:
        """
        Return the top-ranked score for a task with its latest update time.

        Parameters
        ----------
        task_id : int
            Task identifier to evaluate.

        Returns
        -------
        result : pandas.DataFrame
            Output of ``get_last_updated_score_by_task_id(self.df, task_id)``.
        """
        return get_last_updated_score_by_task_id(self.df, task_id)
    
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import math
import numpy as np
import pandas as pd
from pathlib import Path

from calinet.config import config
from calinet.utils import ensure_timestamp, update_kwargs
from calinet.core.io import read_physio_tsv_headerless

import logging
logger = logging.getLogger(__name__)

from typing import Union, Tuple, Dict, Any, List, Iterable


def diameter_to_mm(
        inputs: float,
        camera_eye_distance: float=None,
        measurement_type="AREA"
    ) -> tuple:
    """
    Convert pupil measurements to millimeters.

    This function converts EyeLink pupil measurements (either AREA or DIAMETER)
    into estimated pupil diameter in millimeters using a PsPM-style calibration
    approach. The conversion applies a scaling factor based on the camera-eye
    distance relative to a reference distance.

    Parameters
    ----------
    inputs : float or array-like
        Input pupil measurements. If `measurement_type="AREA"`, values are assumed
        to represent pupil area and will be square-root transformed. If
        `measurement_type="DIAMETER"`, values are treated as diameter directly.
    camera_eye_distance : float, optional
        Distance between the camera and the eye in millimeters. This value is used
        to scale the pupil measurement relative to a reference distance defined
        in the configuration.
    measurement_type : {"AREA", "DIAMETER"}, default="AREA"
        Type of input measurement:
        - "AREA": input values represent pupil area (will be square-rooted)
        - "DIAMETER": input values represent pupil diameter directly

    Returns
    -------
    pupil_mm : float or ndarray
        Estimated pupil diameter in millimeters.
    conversion_info : dict
        Dictionary containing metadata about the conversion, including:
        - "Name": Name of the conversion function
        - "Formula": String representation of the applied formula
        - "Parameters": Dictionary of parameters used (multiplier, reference distance, screen distance)
        - "Description": Brief description of the conversion method

    Raises
    ------
    ValueError
        If `measurement_type` is not one of {"AREA", "DIAMETER"}.

    Notes
    -----
    The conversion formula is:

        diameter_mm = multiplier * (camera_eye_distance / reference_distance) * f(inputs)

    where:
        - f(inputs) = sqrt(inputs) if measurement_type == "AREA"
        - f(inputs) = inputs if measurement_type == "DIAMETER"

    The `multiplier` and `reference_distance` are retrieved from the global
    configuration (`config["pupil_multiplication"]`).

    Examples
    --------
    >>> diameter_to_mm(4000, camera_eye_distance=600, measurement_type="AREA")
    (value_in_mm, {...})

    >>> diameter_to_mm(np.array([3.2, 3.5]), camera_eye_distance=600, measurement_type="DIAMETER")
    (array([...]), {...})
    """

    if measurement_type == "AREA":
        use_vals = np.sqrt(inputs)
        ffunc = "sqrt(input)"
    elif measurement_type == "DIAMETER":
        use_vals = inputs.copy()
        ffunc = "input"
    else:
        raise ValueError(f"measurement_type must be one of 'AREA' or 'DIAMETER', not '{measurement_type}'")

    pupil_config = config.get("pupil_multiplication")
    reference_distance = pupil_config.get("reference_distance")
    multiplier = pupil_config.get(measurement_type)

    logger.info(f"Converting {measurement_type.upper()} to mm; screen distance = {camera_eye_distance}mm | multiplier={multiplier}")

    pupil_mm = (
        multiplier * (camera_eye_distance / reference_distance) * use_vals
    )

    conversion_info = {
        "Name": "diameter_to_mm",
        "Formula": f"diameter_mm = multiplier * (camera_eye_distance / reference_distance) * {ffunc}",
        "Parameters": {
            "Multiplier": multiplier,
            "ReferenceDistance": reference_distance,
            "ScreenDistance": camera_eye_distance
        },
        "Description": "Converts EyeLink pupil AREA units to estimated pupil diameter in mm using PsPM-style calibration."
    }

    return pupil_mm, conversion_info


def pupil_unit_to_mm(
        df: pd.DataFrame,
        overwrite=True,
        column_name="pupil_size",
        **kwargs
    ) -> Tuple[pd.DataFrame, dict]:
    """
    Convert a DataFrame column of pupil measurements to millimeters.

    This function applies :func:`diameter_to_mm` to a specified column in a
    pandas DataFrame and appends the converted values as a new column. Optionally,
    the original column can be overwritten with the converted values.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing pupil measurements.
    overwrite : bool, default=True
        If True, replaces the original column (`column_name`) with the converted
        values in millimeters and removes the temporary column. If False, retains
        both the original and converted columns.
    column_name : str, default="pupil_size"
        Name of the column in `df` containing pupil measurements to convert.
    **kwargs
        Additional keyword arguments passed to :func:`diameter_to_mm`, such as
        `camera_eye_distance` and `measurement_type`.

    Returns
    -------
    out : pandas.DataFrame
        A copy of the input DataFrame with the converted pupil size values.
        If `overwrite=True`, the original column is replaced. Otherwise, a new
        column named ``f"{column_name}_mm"`` is added.
    settings : dict
        Dictionary containing metadata about the conversion, as returned by
        :func:`diameter_to_mm`.

    Notes
    -----
    This function does not modify the input DataFrame in place; a copy is always
    created.

    The conversion is performed element-wise on the specified column using
    :func:`diameter_to_mm`.

    Examples
    --------
    >>> df_mm, info = pupil_unit_to_mm(df, camera_eye_distance=600)
    
    >>> df_mm, info = pupil_unit_to_mm(
    ...     df,
    ...     column_name="pupil_area",
    ...     measurement_type="AREA",
    ...     overwrite=False
    ... )
    """

    out = df.copy()
    out[f"{column_name}_mm"], settings = diameter_to_mm(
        out[column_name],
        **kwargs
    )

    # overwrite column
    if overwrite:
        out[column_name] = out[f"{column_name}_mm"]
        out.drop(columns=[f"{column_name}_mm"], inplace=True)

    return out, settings


def correct_to_fixation_hist_peak(
        df: pd.DataFrame,
        screen_mm: Union[tuple, list],
        x_col: str="x_coordinate",
        y_col: str="y_coordinate",
        bin_size_mm: float=2.0,
        return_shift: bool=True,
    ) -> Tuple[pd.DataFrame, tuple]:
    """
    Recenters gaze coordinates to screen fixation using a 2D histogram-peak (mode) estimator.

    This method estimates the fixation location as the densest region of gaze samples
    in physical screen space (millimeters) and applies a rigid translation so that
    this fixation cluster aligns with the geometric center of the screen.

    Rationale
    ---------
    During fixation tasks, the majority of gaze samples cluster around a central
    fixation target. However, global statistics such as the mean or median can be
    biased by:
        - saccades to stimuli
        - asymmetric exploration
        - blinks or signal dropouts
        - drift or calibration offsets

    Instead of using a global median, this method estimates the fixation location
    as the statistical *mode* of the 2D gaze distribution using a coarse spatial
    histogram. The bin with the highest sample count is taken as the fixation
    cluster center.

    Method
    ------
    1. Extract valid (finite) gaze samples in millimeters.
    2. Construct a 2D histogram over the full physical screen extent with
       square bins of size `bin_size_mm`.
    3. Identify the histogram bin with the maximum count (density peak).
    4. Compute the center of that bin (fx, fy).
    5. Compute the geometric screen center (cx, cy) = (W/2, H/2).
    6. Compute translation shift:
           dx = fx - cx
           dy = fy - cy
    7. Apply rigid translation:
           x' = clip(x - dx, 0, W)
           y' = clip(y - dy, 0, H)

    This is a pure translational correction (no rotation or scaling).

    Parameters
    ----------
    df : pandas.DataFrame
        Gaze dataframe with coordinates in millimeters.
    screen_mm : tuple(float, float)
        Physical screen size in millimeters as (width_mm, height_mm).
    x_col : str
        Column name containing horizontal gaze coordinates (mm).
    y_col : str
        Column name containing vertical gaze coordinates (mm).
    bin_size_mm : float
        Size of square bins used in histogram estimation (mm).
        Typical values: 2–5 mm.
        Smaller bins increase spatial precision but may increase noise sensitivity.
    return_shift : bool
        If True, also return the applied translation (dx, dy).

    Returns
    -------
    df_corr : pandas.DataFrame
        Copy of input dataframe with corrected gaze coordinates.
    (dx, dy) : tuple of float
        Translation shift applied in millimeters (optional).

    Assumptions
    -----------
    - Gaze coordinates are already expressed in physical screen space (mm).
    - The dominant cluster corresponds to fixation.
    - The physical screen size is correctly specified.
    - No rotation or scaling errors are present (only translation drift).

    Limitations
    -----------
    - If fixation is not the dominant gaze state, the method may center
      on a task-relevant stimulus instead.
    - Very small bin sizes can overfit noise.
    - Does not correct non-linear distortions or gain errors.

    Notes
    -----
    This method is robust to asymmetric outliers and saccadic excursions
    because it relies on the distribution mode rather than mean/median.
    It is particularly suitable for large fixation-heavy datasets.
    """

    W, H = screen_mm
    cx, cy = W / 2.0, H / 2.0

    df_corr = df.copy()

    x = df_corr[x_col].to_numpy()
    y = df_corr[y_col].to_numpy()

    # Remove invalid samples
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    # Define histogram bins
    x_bins = np.arange(0, W + bin_size_mm, bin_size_mm)
    y_bins = np.arange(0, H + bin_size_mm, bin_size_mm)

    H2d, x_edges, y_edges = np.histogram2d(x, y, bins=[x_bins, y_bins])

    # Find peak bin
    peak_idx = np.unravel_index(np.argmax(H2d), H2d.shape)

    # Convert bin index to center coordinate
    fx = (x_edges[peak_idx[0]] + x_edges[peak_idx[0] + 1]) / 2
    fy = (y_edges[peak_idx[1]] + y_edges[peak_idx[1] + 1]) / 2

    # Compute shift
    dx = fx - cx
    dy = fy - cy

    # Apply correction
    df_corr[x_col] = np.clip(df_corr[x_col] - dx, 0, W)
    df_corr[y_col] = np.clip(df_corr[y_col] - dy, 0, H)

    if return_shift:
        return df_corr, (dx, dy)
    else:
        return df_corr


def gaze_pixel_to_mm(
        df: pd.DataFrame,
        screen_size_mm=(311.25, 249.09),
        screen_resolution_px=(1152, 864),
        center=False,
        overwrite=True,
        **kwargs
    ) -> pd.DataFrame:
    """
    Convert gaze coordinates from pixel units to millimeters using the physical screen size.

    This function performs a linear pixel-to-metric conversion for 2D gaze coordinates.
    It assumes that the gaze coordinates in `df[x_coordinate, y_coordinate]` are defined
    in a screen coordinate system where pixel (0,0) corresponds to the top-left of the
    active display area, and pixel values increase rightward (x) and downward (y).

    The conversion is performed by computing mm-per-pixel scaling factors separately
    for x and y based on the provided physical screen dimensions and the pixel
    resolution:

        mm_per_px_x = screen_width_mm  / screen_width_px
        mm_per_px_y = screen_height_mm / screen_height_px

    Then:

        x_mm = x_px * mm_per_px_x
        y_mm = y_px * mm_per_px_y

    Optionally, the gaze coordinates can be corrected so that the origin is
    at the screen center instead of the top-left corner via :function:`calinet.import.eyelink.correct_to_fixation_hist_peak()`:

        Instead of using a global median, this method estimates the fixation location
        as the statistical *mode* of the 2D gaze distribution using a coarse spatial
        histogram. The bin with the highest sample count is taken as the fixation
        cluster center.

    This produces coordinates in a “centered mm” frame where (0,0) corresponds to
    the geometric center of the screen.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data containing gaze coordinates in pixel units in columns:
        `x_coordinate` and `y_coordinate` (unless you adapt the column names).
    screen_size_mm : tuple(float, float)
        Physical screen size in millimeters as (width_mm, height_mm). This should
        reflect the actual active display area corresponding to the pixel coordinates.
    screen_resolution_px : tuple(int, int)
        Pixel resolution as (width_px, height_px) corresponding to the coordinate
        system of the gaze data (e.g., from EyeLink GAZE_COORDS).
    center : bool
        If True, recenter the converted mm coordinates so the screen center is (0,0).
        If False (default), the origin remains the top-left corner of the screen.
    overwrite : bool
        If True (default), overwrite `x_coordinate` and `y_coordinate` with the
        millimeter coordinates, and drop the temporary `x_mm`, `y_mm` columns.
        If False, keep `x_coordinate`/`y_coordinate` unchanged and add `x_mm`, `y_mm`.

    Returns
    -------
    pandas.DataFrame
        A copy of the input dataframe with gaze coordinates converted to millimeters.

    Notes
    -----
    - This conversion is a pure linear scaling (+ optional translation). It does not
      correct calibration drift, rotation, or non-linear distortions.
    - X/Y scaling are computed independently; if pixels are square and the physical
      dimensions are consistent with the resolution, mm_per_px_x ≈ mm_per_px_y.
    - The function does not clip values to screen bounds; it assumes the input data
      are valid within the pixel coordinate range.

    Comparison to PsPM (pspm_convert_gaze) in "millimeter mode"
    -----------------------------------------------------------
    When `pspm_convert_gaze` is used with:
        conversion.from   = 'pixel'
        conversion.target = 'mm'
        conversion.screen_width / screen_height set (mm)

    PsPM calls `pspm_convert_pixel2unit_core`, which:
      1) scales pixel coordinates to metric units using the provided screen lengths, and
      2) also converts/updates the *channel range* metadata accordingly.

    This Python function matches the essential *numerical scaling* step (pixel→mm),
    but differs in two ways:

      - Metadata/range handling:
        PsPM updates channel header.range consistently; this function does not update
        any external metadata unless you do so elsewhere.

      - Centering behavior:
        In PsPM, centering is primarily relevant when converting to degrees (visual
        angle), because degrees are defined relative to screen center. Your `center=True`
        option provides a centered-mm coordinate frame (useful, but not strictly the
        same as PsPM’s degree conversion pathway).

    If you want strict PsPM equivalence for pixel→mm, use:
      - center=False
      - consistent screen_size_mm and screen_resolution_px derived from the EyeLink
        coordinate system (GAZE_COORDS / DISPLAY_COORDS).
    """

    w_mm = screen_size_mm[0]
    h_mm = screen_size_mm[1]
    w_px, h_px = screen_resolution_px

    mm_per_px_x = w_mm / float(w_px)
    mm_per_px_y = h_mm / float(h_px)

    out = df.copy()
    
    out["x_mm"] = out["x_coordinate"].astype(float) * mm_per_px_x
    out["y_mm"] = out["y_coordinate"].astype(float) * mm_per_px_y

    if overwrite:
        out["x_coordinate"] = out["x_mm"]
        out["y_coordinate"] = out["y_mm"]
        out.drop(["x_mm", "y_mm"], axis=1, inplace=True)

    # center using histograms
    if center:
        defs = {
            "bin_size_mm": 2.0,
        }

        for key, val in defs.items():
            kwargs = update_kwargs(
                kwargs,
                key,
                val
            )

        out = correct_to_fixation_hist_peak(
            out,
            screen_size_mm,
            x_col="x_coordinate",
            y_col="y_coordinate",
            **kwargs
        )

    return out


def estimate_screen_dimensions(
        diagonal_inch: float,
        width_px: int,
        height_px: int
    ) -> tuple:
    """
    Estimate physical screen dimensions from diagonal size and resolution.

    This function computes the physical width and height of a display (in
    millimeters) based on its diagonal size (in inches) and pixel resolution.
    The calculation assumes square pixels and uses the aspect ratio derived
    from the resolution.

    Parameters
    ----------
    diagonal_inch : float
        Screen diagonal in inches.
    width_px : int
        Horizontal resolution of the screen in pixels.
    height_px : int
        Vertical resolution of the screen in pixels.

    Returns
    -------
    width_mm : float
        Estimated screen width in millimeters.
    height_mm : float
        Estimated screen height in millimeters.

    Notes
    -----
    The computation proceeds as follows:

    1. Convert diagonal from inches to millimeters:
       ``diagonal_mm = diagonal_inch * 25.4``

    2. Compute aspect ratio:
       ``r = width_px / height_px``

    3. Solve for height using the Pythagorean relation:
       ``height_mm = diagonal_mm / sqrt(r^2 + 1)``

    4. Compute width:
       ``width_mm = r * height_mm``

    Assumes:
        - Square pixels
        - Accurate diagonal measurement
        - No display scaling

    Examples
    --------
    >>> estimate_screen_dimensions(24, 1920, 1080)
    (531.3..., 298.8...)

    >>> estimate_screen_dimensions(13.3, 2560, 1600)
    (286.4..., 179.0...)
    """

    # Convert diagonal to mm
    diagonal_mm = diagonal_inch * 25.4

    # Aspect ratio
    r = width_px / height_px

    # Compute height in mm
    height_mm = diagonal_mm / math.sqrt(r**2 + 1)

    # Compute width in mm
    width_mm = r * height_mm

    return width_mm, height_mm


def correct_to_fixation(
        df,
        screen_mm: list,
        x_col: str="x_coordinate",
        y_col: str="y_coordinate",
        radius_mm: float=30,
        return_shift: bool=True,
    ) -> tuple:
    """
    Robustly recenters gaze coordinates to screen fixation.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing gaze coordinates in mm.
    screen_mm : tuple(float, float)
        Physical screen size in mm (W, H).
    x_col, y_col : str
        Column names for gaze coordinates.
    radius_mm : float
        Radius (in mm) used to select central fixation cluster.
    return_shift : bool
        Whether to return (dx, dy) along with dataframe.

    Returns
    -------
    df_corr : pandas.DataFrame
        Corrected dataframe.
    (dx, dy) : tuple (optional)
        Estimated correction shift in mm.
    """

    W, H = screen_mm
    cx, cy = W / 2.0, H / 2.0

    df_corr = df.copy()

    x = df_corr[x_col].to_numpy(dtype=float)
    y = df_corr[y_col].to_numpy(dtype=float)

    # Robust overall center estimate
    mx = np.nanmedian(x)
    my = np.nanmedian(y)

    # Select points within radius of cluster center
    mask = (
        np.isfinite(x)
        & np.isfinite(y)
        & ((x - mx) ** 2 + (y - my) ** 2 <= radius_mm**2)
    )

    if np.sum(mask) < 10:
        raise ValueError("Not enough fixation samples inside radius.")

    # Compute shift
    dx = np.nanmedian(x[mask]) - cx
    dy = np.nanmedian(y[mask]) - cy

    # Apply correction
    x_corr = x - dx
    y_corr = y - dy

    # Clip to screen bounds
    x_corr = np.clip(x_corr, 0, W)
    y_corr = np.clip(y_corr, 0, H)

    df_corr[x_col] = x_corr
    df_corr[y_col] = y_corr

    if return_shift:
        return df_corr, (dx, dy)
    else:
        return df_corr


def eyelink_blink_runs(
        pupil: np.ndarray,
        blink_threshold: float=0.1,
        max_gap_samples: int=2,
        mask_dropout: bool=False,
        dropout_threshold: Union[float, None]=None,
        pad_before_samples: int=0,
        pad_after_samples: int=0,
    ) -> List[Tuple[int, int]]:
    """
    Detect contiguous blink/dropout runs from a pupil signal.

    A sample is classified as part of a blink if the pupil value is not finite
    or is less than or equal to ``blink_threshold``. Optionally, a more
    permissive dropout mask can be added using ``dropout_threshold``.

    Parameters
    ----------
    pupil : numpy.ndarray
        One-dimensional array of pupil size values.
    blink_threshold : float
        Threshold below which a sample is considered a blink core.
    max_gap_samples : int
        Maximum length of non-blink gaps (in samples) that are bridged
        within blink segments.
    mask_dropout : bool
        If True, also classify samples ``<= dropout_threshold`` as blink/dropout.
    dropout_threshold : float | None
        Threshold for broader dropout masking. Only used if ``mask_dropout=True``.
        If None, defaults to ``blink_threshold``.
    pad_before_samples : int
        Number of samples to extend each detected run backward.
    pad_after_samples : int
        Number of samples to extend each detected run forward.

    Returns
    -------
    runs : list of tuple of int
        List of ``(start_idx, end_idx)`` pairs representing blink runs.
        The ``end_idx`` is exclusive.
    """

    pupil = np.asarray(pupil, dtype=float)

    # strict blink core
    is_blink = (~np.isfinite(pupil)) | (pupil <= blink_threshold)

    # optional broader dropout mask
    if mask_dropout:
        if dropout_threshold is None:
            dropout_threshold = blink_threshold
        is_blink |= (pupil <= dropout_threshold)

    if is_blink.size == 0:
        return []

    # bridge tiny false gaps inside blink/dropout runs
    if max_gap_samples > 0:
        arr = is_blink.copy()
        n = len(arr)
        i = 0
        while i < n:
            if arr[i]:
                i += 1
                continue

            j = i
            while j < n and not arr[j]:
                j += 1

            gap_len = j - i
            left_blink = i > 0 and arr[i - 1]
            right_blink = j < n and arr[j]

            if left_blink and right_blink and gap_len <= max_gap_samples:
                arr[i:j] = True

            i = j

        is_blink = arr

    changes = np.where(is_blink[1:] != is_blink[:-1])[0] + 1
    starts = np.r_[0, changes]
    ends = np.r_[changes, len(is_blink)]

    runs = [(s, e) for s, e in zip(starts, ends) if is_blink[s]]

    # optional padding
    if pad_before_samples > 0 or pad_after_samples > 0:
        padded = []
        n = len(pupil)
        for s, e in runs:
            s2 = max(0, s - pad_before_samples)
            e2 = min(n, e + pad_after_samples)
            padded.append((s2, e2))
        runs = padded

    return runs


def blink_runs_to_rows(
        t: np.ndarray,
        runs: Iterable[Tuple[int, int]],
        min_blink_s: float=0.01
    ) -> List[Dict[str, Union[float, int, str]]]:
    """
    Convert blink run indices into event rows.

    Transforms contiguous blink index ranges into dictionaries containing
    onset time, duration, and standardized event fields.

    Parameters
    ----------
    t : numpy.ndarray
        One-dimensional array of timestamps corresponding to samples.
    runs : iterable of tuple of int
        Iterable of ``(start_idx, end_idx)`` blink runs, where
        ``end_idx`` is exclusive.
    min_blink_s : float
        Minimum blink duration (in seconds). Runs shorter than this
        threshold are discarded.

    Returns
    -------
    rows : list of dict of str to float or int or str
        List of event dictionaries with keys:
        ``"onset"``, ``"duration"``, ``"trial_type"``,
        ``"blink"``, and ``"message"``.

    Notes
    -----
    - Onset is taken from ``t[start_idx]`` and offset from
      ``t[end_idx - 1]``.
    - Duration is computed as ``offset - onset`` and clipped to be
      non-negative.
    - The ``"message"`` field is always set to ``"n/a"``.
    """
    
    rows = []
    for s, e in runs:
        onset = t[s]
        offset = t[e - 1]
        dur = max(0.0, float(offset) - float(onset))
        if dur < min_blink_s:
            continue
        rows.append({
            "onset": float(onset),
            "duration": float(dur),
            "trial_type": "blink",
            "blink": 1,
            "message": "n/a",
        })
    return rows


def _ivt_fix_sacc_events(
        df: pd.DataFrame,
        time: pd.Series,
        vel_thresh: float=80.0,
        min_fix_s: float=0.06,
        min_sacc_s: float=0.01
    ) -> Tuple[List[Dict[str, Union[float, int, str]]], Dict[str, Union[str, float]]]:
    """
    Detect fixation and saccade events using a velocity-threshold method.

    Computes sample-to-sample velocity from ``"x_coordinate"`` and
    ``"y_coordinate"`` positions, labels segments as ``"fixation"`` or
    ``"saccade"`` based on ``vel_thresh``, and converts valid segments into
    event rows.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing ``"x_coordinate"`` and ``"y_coordinate"``
        columns.
    time : pandas.Series
        Time values in seconds relative to the start of the recording.
    vel_thresh : float
        Velocity threshold in mm/s used to classify samples as saccades.
    min_fix_s : float
        Minimum duration in seconds for a fixation to be retained.
    min_sacc_s : float
        Minimum duration in seconds for a saccade to be retained.

    Returns
    -------
    rows : list of dict of str to float or int or str
        List of event dictionaries with keys ``"onset"``, ``"duration"``,
        ``"trial_type"``, ``"blink"``, and ``"message"``. The ``"blink"``
        field is always ``0`` and ``"message"`` is always ``"n/a"``.
    settings : dict of str to str or float
        Dictionary describing the event detection settings, including
        ``"SaccadeDetectionAlgorithm"``,
        ``"SaccadeVelocityThreshold_mm_s"``,
        ``"MinimumFixationDuration_s"``, and
        ``"MinimumSaccadeDuration_s"``.

    Notes
    -----
    - Coordinates and time are coerced to numeric with
      ``errors="coerce"``.
    - Only rows with finite ``"x_coordinate"``, ``"y_coordinate"``, and
      time values are used.
    - Non-positive time differences are treated as invalid by replacing
      them with ``NaN`` before velocity calculation.
    - Returns an empty list if fewer than 3 valid samples are available.
    """

    logger.info(f"Calculating saccades from dataframe: vel_thresh_mm_s={vel_thresh} | min. fixation (s)={min_fix_s} | min. saccade (s)={min_sacc_s}")

    x = pd.to_numeric(df["x_coordinate"], errors="coerce").to_numpy()
    y = pd.to_numeric(df["y_coordinate"], errors="coerce").to_numpy()
    t = pd.to_numeric(time, errors="coerce").to_numpy()

    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(t)
    if valid.sum() < 3:
        logger.warning(
            "Skipping saccade detection: insufficient valid samples | "
            f"valid={valid.sum()} / total={len(valid)} | "
            f"invalid_x={(~np.isfinite(x)).sum()} | "
            f"invalid_y={(~np.isfinite(y)).sum()} | "
            f"invalid_t={(~np.isfinite(t)).sum()} | "
            f"vel_thresh_mm_s={vel_thresh} | "
            f"min_fix_s={min_fix_s} | "
            f"min_sacc_s={min_sacc_s}"
        )
        logger.debug(
            f"First 5 samples | x={x[:5]} | y={y[:5]} | t={t[:5]}"
        )
        return [], {}

    x2, y2, t2 = x[valid], y[valid], t[valid]

    dt = np.diff(t2)
    dx = np.diff(x2)
    dy = np.diff(y2)

    dt_safe = np.where(dt <= 0, np.nan, dt)
    vel = np.sqrt(dx * dx + dy * dy) / dt_safe  # units per second

    is_sacc = np.r_[vel[0] >= vel_thresh, vel >= vel_thresh]
    labels = np.where(is_sacc, "saccade", "fixation")

    changes = np.where(labels[1:] != labels[:-1])[0] + 1
    starts = np.r_[0, changes]
    ends = np.r_[changes, len(labels)]

    rows = []
    for s, e in zip(starts, ends):
        seg_type = labels[s]
        onset = t2[s]
        offset = t2[e - 1]
        dur = max(0.0, float(offset) - float(onset))

        if seg_type == "fixation" and dur < min_fix_s:
            continue
        if seg_type == "saccade" and dur < min_sacc_s:
            continue

        rows.append({
            "onset": float(onset),
            "duration": float(dur),
            "trial_type": str(seg_type),
            "blink": 0,
            "message": "n/a",
        })

    settings = {
        "SaccadeDetectionAlgorithm": "velocity threshold",
        "SaccadeVelocityThreshold_mm_s": vel_thresh,
        "MinimumFixationDuration_s": min_fix_s,
        "MinimumSaccadeDuration_s": min_sacc_s
    }

    return rows, settings


def ivt_events_plus_eyelink_blinks(
        df: pd.DataFrame,
        time: pd.Series,
        blink_threshold: float=0.15,
        min_blink_s: float=0.02,
        max_blink_gap_samples: int=2,
        mask_dropout: bool=False,
        dropout_threshold: Union[float, None]=None,
        pad_blink_before_s: float=0.0,
        pad_blink_after_s: float=0.0,
        **kwargs: Any
    ) -> Tuple[List[Dict[str, Union[float, int, str]]], Dict[str, Any]]:
    """
    Detect blink, fixation, and saccade events from eye-tracking data.

    Detects blinks from ``"pupil_size"`` using a pupil-threshold method,
    removes blink samples, and then detects fixations and saccades from the
    remaining ``"x_coordinate"`` and ``"y_coordinate"`` samples using
    ``_ivt_fix_sacc_events``.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing ``"x_coordinate"``, ``"y_coordinate"``, and
        ``"pupil_size"`` columns.
    time : pandas.Series
        Time values in seconds relative to the start of the recording.
    blink_threshold : float
        Pupil threshold at or below which a sample is classified as a blink.
        Non-finite pupil values are also treated as blinks.
    min_blink_s : float
        Minimum blink duration in seconds for a blink event to be retained.
    max_blink_gap_samples : int
        Maximum number of consecutive non-blink samples that may be bridged
        within a blink run.
    **kwargs : Any
        Additional keyword arguments passed directly to
        ``_ivt_fix_sacc_events``.

    Returns
    -------
    rows : list of dict of str to float or int or str
        Combined list of blink, fixation, and saccade event dictionaries.
        Each row contains ``"onset"``, ``"duration"``, ``"trial_type"``,
        ``"blink"``, and ``"message"``.
    settings : dict of str to Any
        Combined settings dictionary containing blink detection parameters
        and the settings returned by ``_ivt_fix_sacc_events``.

    Notes
    -----
    - Time values are coerced to numeric and only finite timestamps are kept.
    - Blink detection uses ``eyelink_blink_runs`` and
      ``blink_runs_to_rows``.
    - Samples classified as blinks are excluded before fixation and saccade
      detection.
    - The returned settings include ``"BlinkDetectionAlgorithm"``,
      ``"BlinkThreshold"``, ``"MaxBlinkGapSamples"``,
      ``"MinimumBlinkDuration_s"``, and ``"SourceDataframeColumns"``.

    Examples
    --------
    ``kwargs`` can be used to pass fixation and saccade detection settings,
    such as ``vel_thresh``, ``min_fix_s``, or ``min_sacc_s``.
    """

    x = pd.to_numeric(df["x_coordinate"], errors="coerce").to_numpy()
    y = pd.to_numeric(df["y_coordinate"], errors="coerce").to_numpy()
    p = pd.to_numeric(df["pupil_size"], errors="coerce").to_numpy()
    t = pd.to_numeric(time, errors="coerce").to_numpy()

    valid_t = np.isfinite(t)
    x, y, p, t = x[valid_t], y[valid_t], p[valid_t], t[valid_t]

    if len(t) < 3:
        return [], {}

    dt = np.nanmedian(np.diff(t))
    fs = 1.0 / dt if np.isfinite(dt) and dt > 0 else None

    if fs is not None:
        pad_before_samples = int(round(pad_blink_before_s * fs))
        pad_after_samples = int(round(pad_blink_after_s * fs))
    else:
        pad_before_samples = 0
        pad_after_samples = 0

    logger.info(
        f"Detecting blinks | max_blink_gap_samples={max_blink_gap_samples} | "
        f"blink_threshold={blink_threshold} | mask_dropout={mask_dropout} | "
        f"dropout_threshold={dropout_threshold}"
    )

    runs = eyelink_blink_runs(
        p,
        blink_threshold=blink_threshold,
        max_gap_samples=max_blink_gap_samples,
        mask_dropout=mask_dropout,
        dropout_threshold=dropout_threshold,
        pad_before_samples=pad_before_samples,
        pad_after_samples=pad_after_samples,
    )

    blink_rows = blink_runs_to_rows(t, runs, min_blink_s=min_blink_s)

    blink_mask = np.zeros(len(p), dtype=bool)
    for start, stop in runs:
        blink_mask[start:stop] = True

    non_blink = ~blink_mask

    df_nb = pd.DataFrame({
        "x_coordinate": x[non_blink],
        "y_coordinate": y[non_blink],
        "pupil_size": p[non_blink],
    })
    t_nb = pd.Series(t[non_blink])

    fs_rows, sac_settings = _ivt_fix_sacc_events(
        df=df_nb,
        time=t_nb,
        **kwargs
    )

    blink_settings = {
        "BlinkDetectionAlgorithm": "pupil threshold",
        "BlinkThreshold": blink_threshold,
        "MaskDropout": mask_dropout,
        "DropoutThreshold": dropout_threshold,
        "MaxBlinkGapSamples": max_blink_gap_samples,
        "MinimumBlinkDuration_s": min_blink_s,
        "PadBlinkBefore_s": pad_blink_before_s,
        "PadBlinkAfter_s": pad_blink_after_s,
        "SourceDataframeColumns": [
            "timestamp",
            "x_coordinate",
            "y_coordinate",
            "pupil_size",
        ],
    }

    return blink_rows + fs_rows, {**sac_settings, **blink_settings}


def fetch_physioevents_from_df(
        df: pd.DataFrame,
        sr: Union[float, int]=None,
        **kwargs: Any
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Build fixation, saccade, and blink physioevents from an eye DataFrame in mm.

    Uses ``"timestamp"``, ``"x_coordinate"``, ``"y_coordinate"``, and
    ``"pupil_size"`` to generate physioevent rows with onset, duration,
    event type, blink flag, and message fields. If ``"timestamp"`` is
    missing, timestamps are added using ``sr``.

    Parameters
    ----------
    df : pandas.DataFrame
        Eye-tracking DataFrame in millimeters. Required columns are
        ``"timestamp"``, ``"x_coordinate"``, ``"y_coordinate"``, and
        ``"pupil_size"``. If ``"timestamp"`` is missing, it is created from
        ``sr``.
    sr : float or int, optional
        Sampling frequency used to add ``"timestamp"`` when it is not
        already present. If ``None`` and ``"timestamp"`` is missing, a
        ``ValueError`` is raised.
    **kwargs : Any
        Additional keyword arguments passed to
        ``ivt_events_plus_eyelink_blinks``.

    Returns
    -------
    events : pandas.DataFrame
        DataFrame with columns ``"onset"``, ``"duration"``,
        ``"trial_type"``, ``"blink"``, and ``"message"``. Returns an empty
        DataFrame with these columns if no events are detected.
    settings : dict of str to Any
        Event detection settings returned by
        ``ivt_events_plus_eyelink_blinks``.

    Raises
    ------
    ValueError
        Raised if ``"timestamp"`` is missing and ``sr`` is ``None``.
    ValueError
        Raised if any required columns are missing after timestamp handling.

    Notes
    -----
    - If ``"timestamp"`` is missing, the function calls
      ``ensure_timestamp`` and reassigns ``df``.
    - Event rows are generated from a copied subset containing only
      ``"x_coordinate"``, ``"y_coordinate"``, and ``"pupil_size"``.
    - Returned events are sorted by ``"onset"`` and reindexed.
    """

    if not "timestamp" in df.columns:
        if sr is None:
            raise ValueError(f"Input does not have timestamp and SamplingFrequency (sr) was not specified. Cannot calculate events in time")

        logger.debug(f"Adding timestamp to dataframe with SamplingFrequency={sr}")
        df = ensure_timestamp(
            df,
            fs=sr,
            force=bool
        )

    required = {"timestamp", "x_coordinate", "y_coordinate", "pupil_size"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    time = pd.to_numeric(df["timestamp"], errors="coerce")

    model_rows, settings = ivt_events_plus_eyelink_blinks(
        df=df[["x_coordinate", "y_coordinate", "pupil_size"]].copy(),
        time=time,
        **kwargs
    )

    events = pd.DataFrame(
        model_rows,
        columns=["onset", "duration", "trial_type", "blink", "message"],
    )

    if events.empty:
        return pd.DataFrame(columns=["onset", "duration", "trial_type", "blink", "message"])

    events = events.sort_values("onset").reset_index(drop=True)
    return events, settings

def pupil_summary(
        df: Union[Path, str, pd.DataFrame]
    ) -> Tuple[float, float, Union[int, float]]:
    
    if isinstance(df, (str, Path)):
        df = read_physio_tsv_headerless(df)

    # check if range is reasonable
    n = len(df["pupil_size"])
    pupil_m = df["pupil_size"].mean()
    pupil_sd = df["pupil_size"].std()
    
    warning_at_mm = config.get("warning_at_mm", 9)
    if pupil_m>warning_at_mm:
        logger.warning(f"Average pupil size ({round(pupil_m, 3)}±{round(pupil_sd, 3)}) > plausible threshold ({warning_at_mm}) over n={n} samples. Output may be invalid..")
    else:
        logger.warning(f"Pupil range is reasonable: {round(pupil_m, 3)}±{round(pupil_sd, 3)}. This is within plausible limit ({warning_at_mm})")

    return pupil_m, pupil_sd, warning_at_mm
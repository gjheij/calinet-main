# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.signal import find_peaks

import logging
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MatReadResult:
    df: pd.DataFrame
    sampling_rate_hz: float
    peak_indices: np.ndarray
    peak_values: np.ndarray
    selected_threshold_low: float
    selected_threshold_high: float


def read_mat_file(
        mat_file: str,
        *,
        data_key: str = "data",
        scr_channel: int = 2,
        ttl_channel: int = 3,
        sampling_rate_hz: float = 1000.0,
        ttl_pulse_width_s: float = 0.01,
        min_peak_distance_s: float = 0.5,
        low_percentile: float = 60.0,
        high_percentile: float = 95.0,
    ) -> MatReadResult:
    """
    Read a simple MAT file where channel 1 is SCR and channel 2 is TTL.

    Parameters
    ----------
    mat_file
        Path to the .mat file.
    data_key
        Variable name inside the MAT file. Defaults to "data".
    scr_channel
        1-based column index for the SCR channel.
    ttl_channel
        1-based column index for the TTL channel.
    sampling_rate_hz
        Sampling rate in Hz.
    ttl_pulse_width_s
        Width of each reconstructed TTL pulse in seconds.
    min_peak_distance_s
        Minimum distance between TTL peaks in seconds.
    low_percentile, high_percentile
        Percentile band used to keep "medium-high" TTL peaks.
        Example: 60..95 keeps stronger peaks but discards tiny and extreme outliers.

    Returns
    -------
    MatReadResult
        Dataframe with time, SCR, raw TTL, and selected TTL.
    """

    mat = sio.loadmat(
        mat_file,
        squeeze_me=True,
        struct_as_record=False
    )

    logger.debug(f"Available keys in MAT file: {list(mat.keys())}")

    if data_key not in mat:
        raise KeyError(f"'{data_key}' not found in {mat_file}. Keys: {list(mat.keys())}")

    data = np.asarray(mat[data_key], dtype=np.float64)
    logger.info(f"Loaded data with shape: {data.shape}")

    if data.ndim != 2:
        raise ValueError(f"Expected a 2D array in '{data_key}', got shape {data.shape}")

    scr_idx = scr_channel - 1
    ttl_idx = ttl_channel - 1

    logger.info(f"Using SCR channel={scr_channel} (idx={scr_idx}), TTL channel={ttl_channel} (idx={ttl_idx})")

    if scr_idx >= data.shape[1] or ttl_idx >= data.shape[1]:
        raise IndexError(
            f"Requested channels SCR={scr_channel}, TTL={ttl_channel}, "
            f"but data only has {data.shape[1]} columns"
        )

    scr = np.asarray(data[:, scr_idx], dtype=np.float64).squeeze()
    ttl_raw = np.asarray(data[:, ttl_idx], dtype=np.float64).squeeze()

    logger.debug(f"SCR stats: min={np.min(scr):.3f}, max={np.max(scr):.3f}")
    logger.debug(f"TTL raw stats: min={np.min(ttl_raw):.3f}, max={np.max(ttl_raw):.3f}")

    peak_indices, peak_values, low_thr, high_thr = _select_medium_high_peaks(
        ttl_raw,
        sampling_rate_hz=sampling_rate_hz,
        min_peak_distance_s=min_peak_distance_s,
        low_percentile=low_percentile,
        high_percentile=high_percentile,
    )

    logger.info(f"Detected {len(peak_values)} candidate TTL peaks")
    logger.info(f"Selected {len(peak_indices)} medium-high peaks")
    logger.debug(f"Peak thresholds: low={low_thr:.3f}, high={high_thr:.3f}")

    ttl_selected = _peaks_to_ttl(
        peak_indices,
        n_samples=len(ttl_raw),
        sampling_rate_hz=sampling_rate_hz,
        pulse_width_s=ttl_pulse_width_s,
        pulse_amplitude=5,
    )

    logger.debug(f"Constructed TTL signal with {np.sum(ttl_selected > 0)} active samples")
    time_s = np.arange(len(scr), dtype=np.float64) / float(sampling_rate_hz)

    df = pd.DataFrame(
        {
            "time_s": time_s,
            "SCR": scr,
            "TTL_raw": ttl_raw,
            "TTL": ttl_selected,
        }
    )

    logger.info(f"Constructed dataframe with {len(df)} samples")

    return MatReadResult(
        df=df,
        sampling_rate_hz=float(sampling_rate_hz),
        peak_indices=peak_indices,
        peak_values=peak_values,
        selected_threshold_low=float(low_thr),
        selected_threshold_high=float(high_thr),
    )


def _select_medium_high_peaks(
        ttl_raw: np.ndarray,
        *,
        sampling_rate_hz: float,
        min_peak_distance_s: float,
        low_percentile: float,
        high_percentile: float,
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Detect local maxima in a TTL signal and retain only medium-high peaks.

    Parameters
    ----------
    ttl_raw
        1D array containing the raw TTL signal.
    sampling_rate_hz
        Sampling rate of the signal in Hz.
    min_peak_distance_s
        Minimum allowed distance between peaks in seconds.
        This is converted to samples and passed to the peak detector.
    low_percentile
        Lower percentile threshold for peak height filtering.
        Peaks below this value are discarded.
    high_percentile
        Upper percentile threshold for peak height filtering.
        Peaks above this value are discarded.

    Returns
    -------
    peak_indices
        Indices of the selected peaks in the input signal.
    peak_values
        Amplitudes of the selected peaks.
    low_thr
        Computed lower threshold (percentile of all detected peak heights).
    high_thr
        Computed upper threshold (percentile of all detected peak heights).

    Notes
    -----
    - Peak detection is performed using `scipy.signal.find_peaks`.
    - All positive peaks are detected first, then filtered based on percentile thresholds.
    - This approach removes both very small peaks (noise) and very large peaks (outliers),
      keeping only the "medium-high" range of TTL events.
    """

    ttl_raw = np.asarray(ttl_raw, dtype=np.float64).squeeze()
    if ttl_raw.ndim != 1:
        raise ValueError("ttl_raw must be 1D")

    logger.debug("Starting peak detection")

    min_distance_samples = max(1, int(round(min_peak_distance_s * sampling_rate_hz)))
    logger.debug(f"Minimum peak distance (samples): {min_distance_samples}")

    peak_indices, properties = find_peaks(ttl_raw, height=0, distance=min_distance_samples)

    logger.info(f"Total detected peaks: {len(peak_indices)}")

    if peak_indices.size == 0:
        logger.warning("No peaks detected in TTL signal")
        return ...

    peak_values = np.asarray(properties["peak_heights"], dtype=np.float64)

    low_thr = np.percentile(peak_values, low_percentile)
    high_thr = np.percentile(peak_values, high_percentile)

    logger.debug(f"Percentile thresholds: {low_percentile}%={low_thr:.3f}, {high_percentile}%={high_thr:.3f}")

    keep = (peak_values >= low_thr) & (peak_values <= high_thr)

    logger.info(f"Peaks after filtering: {np.sum(keep)} / {len(peak_values)}")

    return peak_indices[keep], peak_values[keep], low_thr, high_thr


def _peaks_to_ttl(
        peak_indices: np.ndarray,
        *,
        n_samples: int,
        sampling_rate_hz: float,
        pulse_width_s: float,
        pulse_amplitude: float = 1.0,
    ) -> np.ndarray:
    """
    Convert peak indices into a reconstructed TTL pulse signal.

    Parameters
    ----------
    peak_indices
        1D array of sample indices where peaks occur.
    n_samples
        Total number of samples in the output signal.
    sampling_rate_hz
        Sampling rate of the signal in Hz.
    pulse_width_s
        Width of each TTL pulse in seconds.
        Each peak is expanded into a pulse of this duration.
    pulse_amplitude
        Amplitude of the TTL pulses. Defaults to 1.0.

    Returns
    -------
    ttl
        1D array of length `n_samples` containing the reconstructed TTL signal,
        where detected peaks are represented as pulses.

    Notes
    -----
    - Each peak index is expanded into a rectangular pulse of fixed width.
    - Pulse width is converted from seconds to samples using the sampling rate.
    - Overlapping pulses will overwrite values but retain the same amplitude.
    - Indices outside the valid range `[0, n_samples)` are ignored.
    """

    ttl = np.zeros(int(n_samples), dtype=np.float64)

    logger.debug(f"Converting {len(peak_indices)} peaks into TTL pulses")

    pulse_width_samples = max(1, int(round(pulse_width_s * sampling_rate_hz)))
    logger.debug(f"Pulse width (samples): {pulse_width_samples}")

    for idx in np.asarray(peak_indices, dtype=int):
        if 0 <= idx < n_samples:
            ttl[idx:min(idx + pulse_width_samples, n_samples)] = pulse_amplitude

    logger.debug("Finished TTL reconstruction")

    return ttl

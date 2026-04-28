# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

import numpy as np
import pandas as pd
from pathlib import Path

from dataclasses import dataclass
from typing import Union, List, Iterable, Optional, Tuple

import scipy.io as sio
from scipy.signal import resample_poly, butter, filtfilt, lfilter, lfilter_zi

import logging
logger = logging.getLogger(__name__)

unit_defaults = {
    "scr": "uS",
    "ppg": "V",
    "resp": "mV",
    "markers": "events",
    "ttl": "V"
}


@dataclass(frozen=True)
class PsPMReadResult:
    """
    Container for PsPM physiological data and metadata.

    Attributes
    ----------
    df : pd.DataFrame
        Concatenated dataframe containing physiological signals.
    sampling_rate_hz : float
        Sampling rate in Hertz.
    channel_info : pd.DataFrame
        Dataframe describing channel metadata (e.g., names, units, length).
    """
    df: pd.DataFrame
    sampling_rate_hz: float
    channel_info: pd.DataFrame


def read_pspm_files(
        mat_files: Union[str, Path, List[Union[str, Path]]]
    ) -> PsPMReadResult:
    """
    Read and combine PsPM MAT files into a unified dataset.

    This function loads one or more PsPM MAT files, extracts and resamples
    physiological channels, concatenates them into a single dataframe, and
    constructs channel metadata. All input files must share the same sampling
    rate.

    Parameters
    ----------
    mat_files : str | pathlib.Path or list of str | pathlib.Path
        Path(s) to PsPM MAT file(s). A single string is treated as a list
        with one element.

    Returns
    -------
    result : PsPMReadResult
        Object containing:
        - ``df``: concatenated physiological data
        - ``sampling_rate_hz``: sampling rate in Hertz
        - ``channel_info``: metadata describing the channels

    Raises
    ------
    ValueError
        If sampling rates differ across input files.

    Notes
    -----
    - Channel data are extracted and resampled via `_extract_and_resample_channels`.
    - Channel metadata is constructed using `_build_channel_table`.
    - All input files must have identical sampling rates.

    Examples
    --------
    >>> result = read_pspm_files(["file1.mat", "file2.mat"])
    >>> result.df.head()
    """

    if isinstance(mat_files, (str, Path)):
        mat_files = [mat_files]

    # ensure everything is Path (or str if you prefer)
    mat_files = [Path(f) for f in mat_files]

    physio_df = []
    srs = []
    logger.debug("Resampling channels in PsPM-files")
    for f in mat_files:
        df, sr = _extract_and_resample_channels(f)
        physio_df.append(df)
        srs.append(sr)

    physio_df = pd.concat(physio_df)
    all_same = len(set(srs)) == 1
    if all_same:
        sr = srs[0]
    else:
        raise ValueError(f"Mismatch in extracted SamplingFrequency: {srs}")

    # create biopac-like channel info
    logger.debug("Create channel information table")
    channel_info = _build_channel_table(physio_df, sample_rate=sr)

    return PsPMReadResult(
        df=physio_df,
        sampling_rate_hz=float(sr) if sr is not None else np.nan,
        channel_info=channel_info
    )


def _onsets_to_ttl(
        onsets_s: Union[float, Iterable[float]],
        duration_s: float,
        sr: float,
        pulse_width_s: float=10,
        pulse_mag: int=5
    ) -> np.ndarray:
    """
    Convert onset times into a TTL signal array.

    This function generates a discrete TTL (trigger) signal from a set of onset
    times. Each onset produces a pulse of specified width and magnitude within
    a signal of given duration and sampling rate.

    Parameters
    ----------
    onsets_s : float or iterable of float
        Onset times in seconds. Can be a single value or a sequence of values.
    duration_s : float
        Total duration of the signal in seconds.
    sr : float
        Sampling rate in Hertz.
    pulse_width_s : float, default=10
        Width of each TTL pulse in seconds.
    pulse_mag : int, default=5
        Amplitude of the TTL pulses.

    Returns
    -------
    ttl : np.ndarray
        One-dimensional array representing the TTL signal.

    Notes
    -----
    - Onsets are converted to sample indices using the sampling rate.
    - Invalid (NaN or infinite) onset values are ignored.
    - Pulses are clipped to remain within the signal length.
    - Pulse width is enforced to be at least one sample.

    Examples
    --------
    >>> ttl = _onsets_to_ttl(
    ...     onsets_s=[1.0, 2.5],
    ...     duration_s=10,
    ...     sr=1000
    ... )
    >>> ttl.shape
    """

    n_samples = int(round(duration_s * sr))
    ttl = np.zeros(n_samples, dtype=int)

    onsets_s = np.asarray(onsets_s, dtype=float).squeeze()
    onsets_s = onsets_s[np.isfinite(onsets_s)]

    onset_idx = np.round(onsets_s * sr).astype(int)
    onset_idx = onset_idx[(onset_idx >= 0) & (onset_idx < n_samples)]

    pulse_width = max(1, int(round(pulse_width_s * sr)))

    for idx in onset_idx:
        ttl[idx:min(idx + pulse_width, n_samples)] = pulse_mag

    return ttl


def _build_channel_table(
        df: pd.DataFrame,
        sample_rate: Optional[float]=None
    ) -> pd.DataFrame:
    """
    Construct a channel metadata table from a dataframe.

    This function generates a channel information dataframe describing each
    column in the input dataframe, including index, name, units, sampling rate,
    and number of valid samples.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing signal channels as columns.
    sample_rate : float, optional
        Sampling rate in Hertz assigned to all channels.

    Returns
    -------
    channel_info : pd.DataFrame
        Dataframe with one row per channel containing:
        - ``index``: column index
        - ``output_name``: uppercase column name
        - ``units``: unit inferred from `unit_defaults`
        - ``samples_per_second``: sampling rate
        - ``length``: number of non-missing samples

    Notes
    -----
    - Units are assigned based on `unit_defaults` using lowercase column names.
    - The ``length`` field counts non-null values per column.
    - Output column names are converted to uppercase.

    Examples
    --------
    >>> channel_info = _build_channel_table(df_signals, sample_rate=1000)
    >>> channel_info.head()
    """

    rows = []

    for i, col in enumerate(df.columns):
        rows.append(
            {
                "index": i,
                "output_name": col.upper(),
                "units": unit_defaults.get(col.lower()),
                "samples_per_second": float(sample_rate),
                "length": df[col].notna().sum(),
            }
        )
        
    return pd.DataFrame(rows)


def _extract_and_resample_channels(
        mat_file: Union[str, Path],
        target_sr: Optional[int] = None
    ) -> Tuple[pd.DataFrame, int]:
    """
    Extract and resample physiological channels from a PsPM MAT file.

    This function loads a PsPM MAT file, extracts physiological channels,
    resamples them to a common sampling rate, and constructs a dataframe.
    If a marker channel is present, it is converted into a synthetic TTL signal.

    Parameters
    ----------
    mat_file : str | pathlib.Path
        Path to the PsPM MAT file.
    target_sr : int, optional
        Target sampling rate in Hertz. If ``None``, the maximum sampling
        rate among available non-marker channels is used.

    Returns
    -------
    df : pd.DataFrame
        Dataframe containing all available resampled channels. A ``TTL``
        column is added only if a marker channel is present.
    target_sr : int
        Sampling rate used for all output channels.

    Raises
    ------
    ValueError
        If no non-marker channels are found.

    Notes
    -----
    - Channels are extracted from the ``data`` field of the MAT file.
    - Resampling is performed using `resample_poly` when necessary.
    - Output signals are trimmed to a common duration across channels.
    - Marker channels (if present) are converted to TTL using `_onsets_to_ttl`.
    """

    mat_file = Path(mat_file)
    mat = sio.loadmat(
        mat_file,
        squeeze_me=True,
        struct_as_record=False
    )

    channels = np.atleast_1d(mat["data"])

    extracted = {}
    srs = {}

    logger.debug(f"Found {len(channels)} channels in {mat_file}")
    for ch in channels:
        chantype = getattr(ch.header, "chantype", None)
        sr = getattr(ch.header, "sr", None)

        if chantype is None or sr is None:
            continue

        data = np.asarray(ch.data, dtype=np.float64).squeeze()
        extracted[chantype] = data
        srs[chantype] = int(sr)

    # Use all available non-marker channels as signal channels
    signal_channels = [c for c in extracted if c != "marker"]

    if not signal_channels:
        raise ValueError("No physiological channels found in MAT file.")

    if target_sr is None:
        target_sr = max(srs[c] for c in signal_channels)

    logger.debug(f"Target SamplingFrequency={target_sr}")

    duration_s = min(len(extracted[c]) / srs[c] for c in signal_channels)
    target_len = int(round(duration_s * target_sr))

    def _resample_to_target(x, sr, channel_name):
        x = np.asarray(x, dtype=np.float64).squeeze()

        if sr == target_sr:
            logger.debug(
                f"[{channel_name}]: Source SamplingFrequency ({sr}) matches target ({target_sr}) -> do nothing"
            )
            y = x.copy()
        else:
            logger.debug(f"[{channel_name}]: Resampling channel from {sr} to {target_sr}")
            y = resample_poly(x, up=target_sr, down=sr)

        return np.asarray(y, dtype=np.float64).squeeze()[:target_len]

    data_dict = {}
    for chan in signal_channels:
        data_dict[chan.upper()] = _resample_to_target(
            extracted[chan],
            srs[chan],
            channel_name=chan
    )

    marker = extracted.get("marker", None)
    if marker is not None:
        logger.debug("Convert marker channel to synthetic TTL")
        data_dict["TTL"] = _onsets_to_ttl(
            marker,
            duration_s=duration_s,
            sr=target_sr,
            pulse_width_s=10
        )[:target_len]

    logger.debug("Conversion complete")
    df = pd.DataFrame(data_dict)

    return df, target_sr


def spm_orth_like(X: np.ndarray) -> np.ndarray:
    """Approximate SPM-style column orthogonalization via QR."""
    Q, R = np.linalg.qr(X)
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1
    return Q * signs


def pspm_bf_scrf_f(td, p=None):
    """
    Canonical skin conductance response function.
    Python equivalent of pspm_bf_scrf_f.m.
    """
    if p is None:
        p = np.array([3.0745, 0.7013, 0.3176, 0.0708], dtype=float)
    else:
        p = np.asarray(p, dtype=float).copy()

    if td > 90:
        raise ValueError("Time resolution is larger than duration of the function.")
    if td == 0:
        raise ValueError("Time resolution must be larger than 0.")

    t0, sigma, lambda1, lambda2 = p

    t = np.arange(0, 90, td, dtype=float)

    gt = np.exp(-((t - t0) ** 2) / (2 * sigma ** 2))
    ht = np.exp(-t * lambda1) + np.exp(-t * lambda2)

    ft = np.convolve(gt, ht)
    ft = ft[: t.size]
    ft = ft / np.max(ft)

    return ft, p, t


def pspm_bf_scrf(*args):
    """
    Construct a PsPM-style informed basis set for skin conductance responses (SCR).

    This function is a Python reimplementation of the MATLAB function
    `pspm_bf_scrf` from the PsPM (Psychophysiological Modeling) toolbox.
    It generates a canonical SCR function based on a biexponentially
    modified Gaussian and optionally augments it with temporal and
    dispersion derivatives, following:

        Bach DR, Flandin G, Friston KJ, Dolan RJ (2010).
        Modelling event-related skin conductance responses.
        International Journal of Psychophysiology, 75, 349–356.

    The resulting basis set can be used in GLM-based analyses of SCR data,
    analogous to SPM-style hemodynamic response modeling.

    Parameters
    ----------
    *args : tuple
        Input arguments follow PsPM conventions:

        - (td,) :
            Time resolution (sampling interval) in seconds.
            Equivalent to MATLAB call: pspm_bf_scrf(TD)

        - (td, d) :
            Time resolution and number of derivatives.
            Equivalent to MATLAB call: pspm_bf_scrf(TD, D)

        - ([td, d],) :
            Alternative vector-style input.
            Equivalent to MATLAB call: pspm_bf_scrf([TD, D])

        where:
        td : float
            Sampling interval in seconds (must be > 0 and ≤ 90).
        d : int, optional
            Number of derivatives to include:
                0 → canonical SCR only
                1 → + temporal derivative
                2 → + temporal and dispersion derivatives
            Values outside [0, 2] default to 0.

    Returns
    -------
    bs : ndarray of shape (n_timepoints, n_basis)
        Basis set matrix. Columns correspond to:
            1st column: canonical SCR function
            2nd column (if d ≥ 1): temporal derivative
            3rd column (if d ≥ 2): dispersion derivative

        The basis functions are:
            - Orthogonalized using a QR-based approximation of SPM's `spm_orth`
            - Normalized column-wise to unit range (max - min = 1),
              matching PsPM behavior

    x : ndarray of shape (n_timepoints,)
        Time vector in seconds, ranging from 0 to <90 with step size `td`.

    Notes
    -----
    - The canonical SCR function is generated by `pspm_bf_scrf_f`, which
      implements a convolution of a Gaussian with a biexponential decay.
    - The temporal derivative is computed via first differences.
    - The dispersion derivative is approximated by modifying the Gaussian
      width parameter and subtracting the resulting function.
    - Orthogonalization ensures regressors are linearly independent,
      consistent with SPM/PsPM modeling practices.

    Differences from MATLAB PsPM
    ---------------------------
    - Uses NumPy for numerical operations.
    - Uses QR decomposition to approximate `spm_orth`.
    - Input handling follows Python conventions but preserves PsPM semantics.

    Examples
    --------
    >>> bs, x = pspm_bf_scrf(0.1)
    >>> bs, x = pspm_bf_scrf(0.1, 2)
    >>> bs, x = pspm_bf_scrf([0.1, 2])

    References
    ----------
    PsPM Toolbox:
    https://github.com/bachlab/PsPM

    Bach DR et al. (2010):
    https://doi.org/10.1016/j.ijpsycho.2010.01.005
    """
    if len(args) == 0:
        raise ValueError("No sampling interval stated.")

    first = np.asarray(args[0]).ravel()
    td = float(first[0])

    if first.size == 1 and len(args) == 1:
        d = 0
    elif first.size == 2:
        d = int(first[1])
    else:
        d = int(np.asarray(args[1]).ravel()[0])

    if td > 90:
        raise ValueError("Time resolution is larger than duration of the function.")
    if td == 0:
        raise ValueError("Time resolution must be larger than 0.")

    if d < 0 or d > 2:
        d = 0

    bf, p, x = pspm_bf_scrf_f(td)

    columns = [bf]

    if d > 0:
        time_deriv = np.concatenate([[0.0], np.diff(bf)])
        time_deriv = time_deriv / np.sum(np.abs(time_deriv))
        columns.append(time_deriv)

    if d > 1:
        p_disp = p.copy()
        p_disp[1] = 1.8 * p_disp[1]

        bf_disp, _, _ = pspm_bf_scrf_f(td, p_disp)
        disp_deriv = bf - bf_disp
        disp_deriv = disp_deriv / np.sum(np.abs(disp_deriv))
        columns.append(disp_deriv)

    bs = np.column_stack(columns)

    # orthogonalize
    bs = spm_orth_like(bs)

    # normalize each column by max-min, like MATLAB:
    # bs = bs ./ repmat((max(bs) - min(bs)), size(bs, 1), 1)
    denom = np.max(bs, axis=0) - np.min(bs, axis=0)
    bs = bs / denom

    return bs, x


def highpass_filter_array(x, fs, cutoff=0.05, order=1, axis=0):
    nyq = fs / 2
    b, a = butter(order, cutoff / nyq, btype="high")
    return filtfilt(b, a, x, axis=axis)


def butter_filter_array_uni(x, fs, cutoff_hz, btype="low", order=1, axis=0):
    x = np.asarray(x, dtype=float)

    nyq = fs / 2
    if cutoff_hz is None or cutoff_hz <= 0:
        return x
    if cutoff_hz >= nyq:
        return x

    b, a = butter(order, cutoff_hz / nyq, btype=btype)

    # handle 1D
    if x.ndim == 1:
        zi = lfilter_zi(b, a) * x[0]
        y, _ = lfilter(b, a, x, zi=zi)
        return y

    # handle 2D (time x features)
    if axis != 0:
        x = np.swapaxes(x, axis, 0)

    y = np.zeros_like(x)

    for i in range(x.shape[1]):
        xi = x[:, i]
        zi = lfilter_zi(b, a) * xi[0]
        y[:, i], _ = lfilter(b, a, xi, zi=zi)

    if axis != 0:
        y = np.swapaxes(y, 0, axis)

    return y


def pspm_downsample_array(
    x,
    down=10,
    fs=None,
    lpfreq=5.0,
    lporder=1,
    axis=0,
):
    """
    PsPM-like low-pass then downsample for 1D or 2D arrays.

    Parameters
    ----------
    x : ndarray
        1D signal or 2D array.
    down : int
        Downsampling factor. PsPM SCR default is 10.
    fs : float
        Sampling rate before downsampling.
    lpfreq : float
        Low-pass cutoff in Hz. PsPM SCR default is 5 Hz.
    lporder : int
        Butterworth low-pass order. PsPM SCR default is 1.
    axis : int
        Time axis.

    Returns
    -------
    y : ndarray
        Filtered and decimated array.
    """
    if fs is None:
        raise ValueError("fs must be the sampling rate before downsampling.")

    y = butter_filter_array(
        x,
        fs=fs,
        cutoff_hz=lpfreq,
        btype="low",
        order=lporder,
        axis=axis,
    )

    slicer = [slice(None)] * y.ndim
    slicer[axis] = slice(None, None, down)

    return y[tuple(slicer)]
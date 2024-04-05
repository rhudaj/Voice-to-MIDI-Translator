#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pitch-tracking and tuning estimation"""

import numpy as np
import numba
from librosa import util
import matplotlib.pyplot as pyplot

__all__ = ["estimate_tuning", "pitch_tuning", "piptrack", "yin", "pyin"]

@numba.stencil  # type: ignore
def _pi_stencil(x: np.ndarray) -> np.ndarray:
    """Stencil to compute local parabolic interpolation"""
    a = x[1] + x[-1] - 2 * x[0]
    b = (x[1] - x[-1]) / 2

    if np.abs(b) >= np.abs(a):
        # If this happens, we'll shift by more than 1 bin
        # Suppressing types because mypy has no idea about stencils
        return 0  # type: ignore

    return -b / a  # type: ignore


@numba.guvectorize(
    ["void(float32[:], float32[:])", "void(float64[:], float64[:])"],
    "(n)->(n)",
    cache=True,
    nopython=True,
)  # type: ignore
def _pi_wrapper(x: np.ndarray, y: np.ndarray) -> None:  # pragma: no cover
    """Vectorized wrapper for the parabolic interpolation stencil"""
    y[:] = _pi_stencil(x)


def _cumulative_mean_normalized_difference(
    y_frames: np.ndarray,
    frame_length: int,
    win_length: int,
    min_period: int,
    max_period: int,
) -> np.ndarray:
    """Cumulative mean normalized difference function (equation 8 in [#]_)

    Parameters
    ----------
    y_frames : np.ndarray [shape=(frame_length, n_frames)]
        framed audio time series.
    frame_length : int > 0 [scalar]
        length of the frames in samples.
    win_length : int > 0 [scalar]
        length of the window for calculating autocorrelation in samples.
    min_period : int > 0 [scalar]
        minimum period.
    max_period : int > 0 [scalar]
        maximum period.

    Returns
    -------
    yin_frames : np.ndarray [shape=(max_period-min_period+1,n_frames)]
        Cumulative mean normalized difference function for each frame.
    """
    '''
        Autocorrelation.

    '''
    a = np.fft.rfft(y_frames, frame_length, axis=-2)                                # same shape: (frame_length, # frames)
    b = np.fft.rfft(y_frames[..., win_length:0:-1, :], frame_length, axis=-2)
    acf_frames = np.fft.irfft(a * b, frame_length, axis=-2)[..., win_length:, :]
    acf_frames[np.abs(acf_frames) < 1e-6] = 0                                       # Shape => (window_Length, # frames)

    # Energy terms.
    energy_frames = np.cumsum(y_frames**2, axis=-2)
    energy_frames = (
        energy_frames[..., win_length:, :] - energy_frames[..., :-win_length, :]
    )
    energy_frames[np.abs(energy_frames) < 1e-6] = 0                                 # same shape as acf_frames

    # Difference function.
    yin_frames = energy_frames[..., :1, :] + energy_frames - 2 * acf_frames         # same shape as acf_frames/energy_frames


    # Cumulative mean normalized difference function:
    '''
        yin_numerator
        shape is (max_period - min_period + 1, # frames)
    '''
    yin_numerator = yin_frames[..., min_period : max_period + 1, :]

    # broadcast this shape to have leading ones
    tau_range = util.expand_to(                                                     # shape = (max_period, 1)
        np.arange(1, max_period + 1), ndim=yin_frames.ndim, axes=-2
    )

    cumulative_mean = (                                                             # shape = (max_period, # frames)
        np.cumsum(yin_frames[..., 1 : max_period + 1, :], axis=-2) / tau_range
    )

    yin_denominator = cumulative_mean[..., min_period - 1 : max_period, :]          # shape: same as yin_numerator


    yin_frames: np.ndarray = yin_numerator / (
        yin_denominator + util.tiny(yin_denominator)                                # shape: same as yin_numerator
    )

    return yin_frames


def _parabolic_interpolation(x: np.ndarray, *, axis: int = -2) -> np.ndarray:
    """Piecewise parabolic interpolation for yin and pyin.

    Parameters
    ----------
    x : np.ndarray
        array to interpolate
    axis : int
        axis along which to interpolate

    Returns
    -------
    parabolic_shifts : np.ndarray [shape=x.shape]
        position of the parabola optima (relative to bin indices)

        Note: the shift at bin `n` is determined as 0 if the estimated
        optimum is outside the range `[n-1, n+1]`.
    """
    # Rotate the target axis to the end
    xi = x.swapaxes(-1, axis)

    # Allocate the output array and rotate target axis
    shifts = np.empty_like(x)
    shiftsi = shifts.swapaxes(-1, axis)

    # Call the vectorized stencil
    _pi_wrapper(xi, shiftsi)

    # Handle the edge condition not covered by the stencil
    shiftsi[..., -1] = 0
    shiftsi[..., 0] = 0

    return shifts


def yin(
    y: np.ndarray,
    fmin: float,
    fmax: float,
    sr: float,
    frame_length: int,
    win_length: int,
    hop_length: int,
    trough_threshold: float = 0.1,
    pad_mode = "constant",
) -> np.ndarray:
    """Fundamental frequency (F0) estimation using the YIN algorithm.

    YIN is an autocorrelation based method for fundamental frequency estimation [#]_.
    First, a normalized difference function is computed over short (overlapping) frames of audio.
    Next, the first minimum in the difference function below ``trough_threshold`` is selected as
    an estimate of the signal's period.
    Finally, the estimated period is refined using parabolic interpolation before converting
    into the corresponding frequency.

    Parameters
    ----------
    y : np.ndarray [shape=(..., n)]
        audio time series. Multi-channel is supported..
    fmin : number > 0 [scalar]
        minimum frequency in Hertz.
        The recommended minimum is ``librosa.note_to_hz('C2')`` (~65 Hz)
        though lower values may be feasible.
    fmax : number > fmin, <= sr/2 [scalar]
        maximum frequency in Hertz.
        The recommended maximum is ``librosa.note_to_hz('C7')`` (~2093 Hz)
        though higher values may be feasible.
    sr : number > 0 [scalar]
        sampling rate of ``y`` in Hertz.
    frame_length : int > 0 [scalar]
        length of the frames in samples.
        By default, ``frame_length=2048`` corresponds to a time scale of about 93 ms at
        a sampling rate of 22050 Hz.
    win_length : None or int > 0 [scalar]
        length of the window for calculating autocorrelation in samples.
        If ``None``, defaults to ``frame_length // 2``
    hop_length : None or int > 0 [scalar]
        number of audio samples between adjacent YIN predictions.
        If ``None``, defaults to ``frame_length // 4``.
    trough_threshold : number > 0 [scalar]
        absolute threshold for peak estimation.
    center : boolean
        If ``True``, the signal `y` is padded so that frame
        ``D[:, t]`` is centered at `y[t * hop_length]`.
        If ``False``, then ``D[:, t]`` begins at ``y[t * hop_length]``.
        Defaults to ``True``,  which simplifies the alignment of ``D`` onto a
        time grid by means of ``librosa.core.frames_to_samples``.
    pad_mode : string or function
        If ``center=True``, this argument is passed to ``np.pad`` for padding
        the edges of the signal ``y``. By default (``pad_mode="constant"``),
        ``y`` is padded on both sides with zeros.
        If ``center=False``,  this argument is ignored.
        .. see also:: `np.pad`

    Returns
    -------
    f0: np.ndarray [shape=(..., n_frames)]
        time series of fundamental frequencies in Hertz.

        If multi-channel input is provided, f0 curves are estimated separately for each channel.

    Note:
        Check the feasibility of yin/pyin parameters against
        the following conditions:

        1. 0 < fmin < fmax <= sr/2
        2. frame_length - win_length - 1 > sr/fmax
    """

    '''
        Pad the time series so that frames are centered:
            adds frame_length // 2 0's to either side of y:
                (FL // 2 0's) - y - (FL // 2 0's)
    '''
    padding = [(0, 0)] * y.ndim
    padding[-1] = (frame_length // 2, frame_length // 2)
    y = np.pad(y, padding, mode=pad_mode)                  # shape => (y.size + frame_length, )

    '''
        Frame audio.
            splits y into frames of size <frame_length>. Next one starts at last one + hop-Length
                ie: frame( [0, 1, 2, 3, 4, 5], 3, 2 ) => [0,1,2], [2,3,4], [4, 5]
                # frames = (y.size - padding) // hop_length
    '''
    y_frames = util.frame(y, frame_length=frame_length, hop_length=hop_length)

    # Calculate minimum and maximum periods
    min_period = int(np.floor(sr / fmax))
    max_period = min(int(np.ceil(sr / fmin)), frame_length - win_length - 1)

    '''
        Calculate cumulative mean normalized difference function.
        yin_frames.shape = (max_period - min_period + 1, # frames)
    '''
    yin_frames = _cumulative_mean_normalized_difference(
        y_frames, frame_length, win_length, min_period, max_period
    )

    # pyplot.close()
    # pyplot.imshow(yin_frames, cmap='magma')
    # pyplot.title( "YIN Frames" )
    # pyplot.xlabel('Frame #')
    # pyplot.ylabel('Period')
    # pyplot.colorbar()
    # pyplot.show()

    '''
        Parabolic interpolation.
            Does not change shape
    '''
    parabolic_shifts = _parabolic_interpolation(yin_frames)

    # Find local minima.
    is_trough = util.localmin(yin_frames, axis=-2)
    is_trough[..., 0, :] = yin_frames[..., 0, :] < yin_frames[..., 1, :]

    # Find minima below peak threshold.
    is_threshold_trough = np.logical_and(is_trough, yin_frames < trough_threshold)

    '''
        Absolute threshold.
        "The solution we propose is to set an absolute threshold and choose the
        smallest value of tau that gives a minimum of d' deeper than
        this threshold. If none is found, the global minimum is chosen instead.
    '''
    target_shape = list(yin_frames.shape)
    target_shape[-2] = 1

    global_min = np.argmin(yin_frames, axis=-2)
    yin_period = np.argmax(is_threshold_trough, axis=-2)

    global_min = global_min.reshape(target_shape)
    yin_period = yin_period.reshape(target_shape)

    no_trough_below_threshold = np.all(~is_threshold_trough, axis=-2, keepdims=True)
    yin_period[no_trough_below_threshold] = global_min[no_trough_below_threshold]

    # Refine peak by parabolic interpolation.

    yin_period = (
        min_period
        + yin_period
        + np.take_along_axis(parabolic_shifts, yin_period, axis=-2)
    )[..., 0, :]

    # Convert period to fundamental frequency.
    f0: np.ndarray = sr / yin_period

    return f0

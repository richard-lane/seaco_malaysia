"""
For smoothing and removing noise from time series

"""
import numpy as np
import pandas as pd

from scipy import signal

from . import util


def convolve_rectangle(pts: np.ndarray, width: int) -> np.ndarray:
    """
    Smooth an array of points using the given window

    :param pts: array of time series points to smooth
    :param width: width of window/kernel to use
    :returns: smoothed array same shape as pts

    """
    return np.convolve(pts, np.ones(width), "same") / width


def moving_avg(pts: pd.Series, width: int) -> pd.Series:
    """
    Rolling moving average

    :param pts: pandas Series of points to smooth
    :param width: width of the moving window, as an integer (number of points)

    :returns: smoothed points with same length as pts.
    Points near the edges are not smoothed

    """
    # Use the median because large outliers are likely real signal
    return pts.rolling(window=width, min_periods=1).median()


def remove_low_freqs(pts: np.ndarray, *, method="convolve", width=None) -> np.ndarray:
    """
    Remove the slowly-varying part part of a time series

    Either smooths the time series ("convolve" or "moving avg") then removes this from the
    series, or uses a high-pass filter to remove

    :param pts: array of points
    :param width: width of window/kernel to use in the case of "convolve" or "moving avg".
                  Must be specified in these cases.
    :param method: "convolve", "filter" or "moving avg"
    TODO add new parameters for the FFT version

    :returns: an array with the same shape as pts, with the slow frequency part removed

    """
    if method not in {"convolve", "moving avg", "filter"}:
        raise ValueError(f"Invalid method specified: {method}")

    if method == "filter":
        # TODO un hard code
        filter = signal.butter(2, 1, "hp", fs=util.SAMPLE_RATE_HZ, output="sos")
        filtered = signal.sosfilt(filter, pts)

        return filtered

    # Default moving average window width of 10 points (0.1s)
    if width is None:
        width = 10

    # Take the moving average, using the appropriate algorithm
    avg_fcn = moving_avg if method == "moving avg" else convolve_rectangle
    smoothed = avg_fcn(pts, width)

    return pts - smoothed

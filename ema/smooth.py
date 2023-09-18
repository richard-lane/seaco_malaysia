"""
For smoothing and removing noise from time series

"""
import numpy as np
import pandas as pd

from scipy import signal

from . import util


def convolve_rectangle(pts: pd.Series, width: int) -> pd.Series:
    """
    Smooth a series of points using the given window

    :param pts: array of time series points to smooth
    :param width: width of window/kernel to use
    :returns: smoothed array same shape as pts

    """
    avg = np.convolve(pts, np.ones(width), "same") / width
    return pd.Series(data=pts - avg, index=pts.index)


def moving_avg(pts: pd.Series, width: int) -> pd.Series:
    """
    Rolling moving average

    :param pts: pandas Series of points to smooth
    :param width: width of the moving window, as an integer (number of points)

    :returns: smoothed points with same length as pts.
    Points near the edges are not smoothed

    """
    # Use the median because large outliers are likely real signal
    avg = pts.rolling(window=width, min_periods=1).median()

    return pd.Series(data=pts - avg, index=pts.index)


def highpass_filter(
    pts: pd.Series,
    *,
    order: int,
    critical_freq: float,
) -> pd.Series:
    """
    Remove low frequencies using a high-pass filter

    :param pts: time series values

    :returns: an array with the same shape as pts, with the low frequency part removed

    """
    filter = signal.butter(
        N=order, Wn=critical_freq, btype="hp", fs=util.SAMPLE_RATE_HZ, output="sos"
    )

    filtered = signal.sosfilt(filter, pts)

    return pd.Series(data=filtered, index=pts.index)

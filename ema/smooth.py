"""
For smoothing and removing noise from time series

"""
import numpy as np


def convolve_rectangle(pts: np.ndarray, width: int) -> np.ndarray:
    """
    Smooth an array of points using the given window

    :param pts: array of time series points to smooth
    :param width: width of window/kernel to use
    :returns: smoothed array same shape as pts

    """
    return np.convolve(pts, np.ones(width), "same") / width


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

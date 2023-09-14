"""
Helpers to wrap up concepts for analysis stuff

"""
import numpy as np
import pandas as pd

from scipy.signal import convolve
from scipy import integrate as sciint


def moving_avg(array: np.ndarray, width: int):
    """
    Use FFT to convolve an array with a rectangular window

    :param array: the array of values
    :param width: the window width; width=1 corresponds to no smoothing
    :returns: an array of smoothed values

    """
    return convolve(array, np.ones(width) / width, method="fft", mode="valid")


def magnitude(accel_df: pd.DataFrame) -> np.ndarray:
    """
    Magnitude of space acceleration

    :param accel_df: accelerometer dataframe,
                     where down is z and gravity is not taken off,
                     in units or g
    :returns: the overall magnitude of acceleration

    """
    return np.sqrt(
        accel_df["accel_x"] ** 2
        + accel_df["accel_y"] ** 2
        + (accel_df["accel_z"] - 1) ** 2
    )


def smooth(pts: np.ndarray, width: int):
    """
    Smooth an array of points using the given window

    :param pts: array of time series points to smooth
    :param width: width of window/kernel to use
    :returns: smoothed array same shape as pts

    """
    return np.convolve(pts, np.ones(width), "same") / width


def integrate(y: np.ndarray, dx) -> np.ndarray:
    """
    Approximate numerical integral

    """
    return sciint.cumtrapz(y, initial=0, dx=dx)

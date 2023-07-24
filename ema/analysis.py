"""
Helpers to wrap up concepts for analysis stuff

"""
import numpy as np
from scipy.signal import convolve


def moving_avg(array: np.ndarray, width: int):
    """
    Use FFT to convolve an array with a rectangular window

    :param array: the array of values
    :param width: the window width; width=1 corresponds to no smoothing
    :returns: an array of smoothed values

    """
    return convolve(array, np.ones(width) / width, method="fft", mode="valid")

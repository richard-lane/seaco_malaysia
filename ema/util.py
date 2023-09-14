"""
General utility functions

"""
import numpy as np

# How often the accelerometer took a measurement
SAMPLE_RATE_HZ = 100

def count_dict(array: np.ndarray) -> dict:
    """
    Return a dict of unique values + counts in an array

    :param array: the array of values
    :param returns: a dict {k: v, ...} where k are the unique values in the array
                    and v are the corresponding counts

    """
    return dict(zip(*np.unique(array, return_counts=True)))

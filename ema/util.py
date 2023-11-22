"""
General utility functions

"""
import warnings
import numpy as np
import pandas as pd

# How often the accelerometer took a measurement
SAMPLE_RATE_HZ = 100

# Gravity
GRAVITY_MS2 = 9.81


def count_dict(array: np.ndarray) -> dict:
    """
    Return a dict of unique values + counts in an array

    :param array: the array of values
    :param returns: a dict {k: v, ...} where k are the unique values in the array
                    and v are the corresponding counts

    """
    return dict(zip(*np.unique(array, return_counts=True)))


def ramadan_2022() -> pd.Series:
    """
    Ramadan dates

    """
    return pd.to_datetime(["2022-04-02", "2022-05-02"])


def in_ramadan_2022(dates: pd.Series) -> pd.Series:
    """
    Boolean mask indicating whether a series of datetimes are n ramadan

    :param dates: series of datetimes
    :raises: warning if the dates are not all in 2022

    """
    # Check if the dates are all in 2022
    try:
        if set(dates.year.unique()) != {2022}:
            warnings.warn(f"Not all dates are in 2022: {set(dates.year.unique())=} ")
    except AttributeError:
        warnings.warn(f"Dates are not a datetime series: {type(dates)=}")

    # Check whether they're in 2022 ramadan
    start, end = ramadan_2022()
    return (start <= dates) & (dates <= end)

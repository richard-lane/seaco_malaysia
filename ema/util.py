"""
General utility functions

"""
import warnings
import numpy as np
import pandas as pd
import traceback

# How often the accelerometer took a measurement
SAMPLE_RATE_HZ = 100

# Gravity
GRAVITY_MS2 = 9.81

class bcolour:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


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


def in_ramadan_2022(dates: pd.Series, verbose: bool = True) -> pd.Series:
    """
    Boolean mask indicating whether a series of datetimes are n ramadan

    :param dates: series of datetimes
    :param verbose: warnings
    :raises: warning if the dates are not all in 2022

    """
    # Check if the dates are all in 2022
    try:
        if set(dates.year.unique()) != {2022}:
            if verbose:
                warnings.warn(
                    f"Not all dates are in 2022: {set(dates.year.unique())=} "
                )
                traceback.print_stack(limit=5)

    except AttributeError:
        if verbose:
            warnings.warn(f"Dates are not a datetime series: {type(dates)=}")
            traceback.print_stack(limit=5)

    # Check whether they're in 2022 ramadan
    start, end = ramadan_2022()
    return (start <= dates) & (dates <= end)

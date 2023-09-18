"""
Helpers to wrap up concepts for analysis stuff

"""
import numpy as np
import pandas as pd

from scipy.signal import convolve
from scipy import integrate as sciint


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


def integrate(y: pd.Series, dx: float) -> pd.Series:
    """
    Approximate numerical integral

    """
    return pd.Series(data=sciint.cumtrapz(y, initial=0, dx=dx), index=y.index)

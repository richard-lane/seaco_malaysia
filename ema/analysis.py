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


def find_participant_entries(
    meal_info: pd.DataFrame,
) -> dict[int, tuple[pd.Series, list]]:
    """
    Find the number of entries per day per participant

    :param meal_info: smartwatch entries dataframe

    :returns: dictionary linking participant number and (dates, number of entries) tuple

    """
    # Let's also build up a dictionary of the number of entries per day per participant
    participant_entries = {}

    # Iterate over participants
    for participant in meal_info["p_id"].unique():
        # For each participant, find how many entries there were on each day of the study
        df_slice = meal_info[meal_info["p_id"] == participant]

        # Get the start date
        assert df_slice.index.is_monotonic_increasing
        dates = np.unique(df_slice.index.date)

        # Iterate over the participation days, finding how many entries were on each day
        entries_per_day = [np.sum(df_slice.index.date == date) for date in dates]

        participant_entries[participant] = (dates, entries_per_day)

    return participant_entries

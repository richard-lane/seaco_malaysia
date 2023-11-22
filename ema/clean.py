"""
Data cleaning stuff

"""
import numpy as np
import pandas as pd

from . import read


def duplicates(meal_info: pd.DataFrame, delta_minutes: int = 5) -> np.ndarray:
    """
    Boolean mask indicating which rows in the dataframe are duplicates

    Index in the dataframe must be sorted

    :param meal_info: smartwatch entry dataframe
    :param delta_minutes: maximum time difference between entries before they are considered duplicates

    :returns: boolean mask

    """
    assert meal_info.index.is_monotonic_increasing, "Index not sorted"

    # Bool mask
    mask = np.full(len(meal_info), True)

    # Check if the previous entry matches in all these columns
    for column in ["p_id", "meal_type", "portion_size", "utensil", "location"]:
        series = meal_info[column]
        mask &= series.eq(series.shift(1))

    # Check whether the time is withint 5 minutes of the previous entry
    minutes_diff = meal_info.index.to_series().diff().dt.total_seconds().div(60)
    time_mask = minutes_diff < delta_minutes

    mask &= time_mask

    return mask


def catchups_mask(meal_info: pd.DataFrame) -> pd.Series:
    """
    Find a boolean mask indicating which rows indicate the start and end of the catchup period

    :param meal_info: dataframe holding smartwatch entries

    :returns: a boolean mask indicating which rows are catchups

    """
    return meal_info["meal_type"].isin(
        ["No catch-up", "Catch-up start", "Catch-up end"]
    )


def cleaned_smartwatch(*, remove_catchups: bool = False) -> pd.DataFrame:
    """
    Return a dataframe of meal time info that has:
        - had duplicates removed (as defined above)
        - had events before the participant watch distribution date removed
        - had events on the watch distribution date removed
        - additional columns indicating whether each meal or period fell within Ramadan

    :param remove_catchups: whether to remove the markers indicating the start and end of the catchup period

    :returns: a cleaned copy of the dataframe

    """
    meal_info = read.all_meal_info()
    meal_info = read.add_timedelta(meal_info)

    # Find early entries
    meal_info = meal_info[meal_info["delta"].dt.days >= 1]

    # Find duplicates
    meal_info = meal_info[~duplicates(meal_info)]

    if remove_catchups:
        meal_info = meal_info[~catchups_mask(meal_info)]

    # Add Ramadan info

    return meal_info

"""
Data cleaning stuff

"""
import numpy as np
import pandas as pd

from . import read, util


def duplicates(meal_info: pd.DataFrame, delta_minutes: int = 5) -> pd.Series:
    """
    Boolean mask indicating which rows in the dataframe are duplicates

    Index in the dataframe must be sorted

    :param meal_info: smartwatch entry dataframe
    :param delta_minutes: maximum time difference between entries before they are considered duplicates

    :returns: boolean mask

    """
    assert meal_info.index.is_monotonic_increasing, "Index not sorted"

    # Bool mask for the whole dataframe
    mask = pd.Series(False, index=meal_info.index)

    # Iterate over each participant
    for p_id, group in meal_info.groupby("p_id"):
        group_mask = np.full(len(group), True)

        # Check if the previous entry matches in all these columns
        for column in ["meal_type", "portion_size", "utensil", "location"]:
            series = group[column]
            group_mask &= series.eq(series.shift(1))

        # Check whether the time is withint 5 minutes of the previous entry
        minutes_diff = group.index.to_series().diff().dt.total_seconds().div(60)
        time_mask = minutes_diff < delta_minutes

        group_mask &= time_mask.values

        pid_mask = meal_info["p_id"] == p_id
        mask.loc[pid_mask] = group_mask

    mask.index = meal_info.index

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


def clean_meal_info(meal_df: pd.DataFrame, *, verbose: bool = False) -> pd.DataFrame:
    """
    Clean the provided meal info dataframe.

    Returns a dataframe of meal time info that has:
        - had duplicates removed (as defined above)
        - had events before the participant watch distribution date removed
        - had events on the watch distribution date removed

    :param meal_df: dataframe of meal info
    :param verbose: extra print information

    :returns: a cleaned copy of the dataframe

    """
    retval = meal_df.copy()

    # Find early entries
    retval = retval[retval["delta"].dt.days >= 1]

    # Find duplicates
    meal_info = meal_info[~duplicates(meal_info)]

    return meal_info

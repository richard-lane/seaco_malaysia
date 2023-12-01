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


def catchup_category(meal_info: pd.DataFrame) -> pd.DataFrame:
    """
    Find the catchup category for each "Catch-up start" entry in the dataframe.

    Adds a new column "catchup_category" and returns a new dataframe.
    Dataframe must have datetime as index

    Checks in this order:
        - Open-ended: no Catch-up end
        - Long: end longer than 60s after start
        - Early: started before 0800
        - Normal: end within 60s of start, started between 0800 and 0805
        - Late: started after 0805
    ValueError otherwise.

    :param meal_info: dataframe holding smartwatch entries

    :returns: a new dataframe with the catchup category column added
    :raises ValueError: if a catchup is encountered outside of these categories

    """
    copy = meal_info.copy()

    # Add a new column
    col_name = "catchup_category"
    copy["catchup_category"] = np.nan

    # Iterate rows
    # Slow, but it's fine
    in_catchup = False
    start_time = None
    for time, row in copy.iterrows():
        # We've encountered a new catchup
        if row["meal_type"] == "Catch-up start":
            start_time = time

            # If there's no end time, then it's open-ended
            if in_catchup:
                copy.loc[time, col_name] = "Open-ended"
                in_catchup = False
                continue
            else:
                in_catchup = True
                continue

        elif row["meal_type"] == "Catch-up end":
            if not in_catchup:
                raise ValueError("Catch-up end without start")

            in_catchup = False
            catchup_length = (time - start_time).total_seconds()

            # If the end is more than 60s after the start, then it's long
            if catchup_length > 60:
                copy.loc[start_time, col_name] = "Long"
                continue

            # If the start time is before 0800, then it's early
            if start_time.hour < 8:
                copy.loc[start_time, col_name] = "Early"
                continue

            # If the start time is after 0805, then it's late
            if start_time.hour > 8 or (start_time.hour == 8 and start_time.minute > 5):
                copy.loc[start_time, col_name] = "Late"
                continue

            # Otherwise, it's normal
            copy.loc[start_time, col_name] = "Normal"
    
    return copy


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
    retval = retval[~duplicates(retval)]

    return retval

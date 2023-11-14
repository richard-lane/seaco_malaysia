"""
Data cleaning stuff

"""
import numpy as np
import pandas as pd


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

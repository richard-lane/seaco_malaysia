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


def cleaned_smartwatch() -> pd.DataFrame:
    """
    Return a dataframe of meal time info that has:
        - had duplicates removed (as defined above)
        - had events before the participant watch distribution date removed
        - had events on the watch distribution date removed

    :param meal_info: dataframe holding smartwatch entries

    :returns: a cleaned copy of the dataframe

    """
    meal_info = read.all_meal_info()
    feasibility_info = read.smartwatch_feasibility()

    # We only care about ones who consented to the smartwatch study
    feasibility_info = feasibility_info[feasibility_info["smartwatchwilling"] == 1]
    feasibility_info = feasibility_info[["residents_id", "actualdateofdistribution1st"]]

    # Join dataframes
    meal_info = (
        meal_info.reset_index()
        .merge(feasibility_info, left_on="p_id", right_on="residents_id", how="left")
        .set_index(meal_info.index)
    )

    # Find early entries
    meal_info["delta"] = (
        meal_info.index.to_series() - meal_info["actualdateofdistribution1st"]
    )
    meal_info = meal_info[meal_info["delta"].dt.days >= 1]

    # Find duplicates
    return meal_info[~duplicates(meal_info)]

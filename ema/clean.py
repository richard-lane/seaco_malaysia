"""
Data cleaning stuff

"""
import numpy as np
import pandas as pd

from . import read, util


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


def _first_last_date(meal_info: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Return a series of the first date for each participant

    :param meal_info: dataframe of smartwatch entries

    :returns: series of the first and last date for each participant. Indexed by p_id

    """
    dates = meal_info.reset_index().groupby("p_id")["Datetime"]
    return dates.first(), dates.last()


def _ramadan_info(meal_info: pd.DataFrame) -> pd.DataFrame:
    """
    Information about whether the participant's period intersected with Ramadan

    """
    # Whether the first and last entry for this participant was within Ramadan
    first, last = _first_last_date(meal_info)
    ramadan_df = (
        util.in_ramadan_2022(first)
        .to_frame()
        .rename(columns={"Datetime": "first_in_ramadan"})
    )  # Find whether the first entry was in ramadan
    ramadan_df = ramadan_df.merge(
        util.in_ramadan_2022(last)
        .to_frame()
        .rename(columns={"Datetime": "last_in_ramadan"}),
        on="p_id",
    )  # Find whether the last entry was in ramadan, and add it to the df

    ramadan_df["all_in_ramadan"] = (
        ramadan_df["first_in_ramadan"] & ramadan_df["last_in_ramadan"]
    )

    ramadan_df["any_in_ramadan"] = (
        ramadan_df["first_in_ramadan"] | ramadan_df["last_in_ramadan"]
    )

    return ramadan_df


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

    # Ramadan info
    # Whether each entry was within Ramadan
    meal_info["in_ramadan"] = util.in_ramadan_2022(meal_info.index)

    # Whether the participants period was within Ramadan
    ramadan_info = _ramadan_info(meal_info)

    # Need to do this to preserve the index
    meal_info["Datetime"] = meal_info.index
    return meal_info.merge(ramadan_info, on="p_id").set_index("Datetime")

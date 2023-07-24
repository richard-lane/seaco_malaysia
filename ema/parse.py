"""
Helper functions for parsing, cleaning data etc.

"""
import pandas as pd
import numpy as np

from . import util


def extract_meals(
    meal_df: pd.DataFrame, allowed_meal_types: set, verbose: bool = False
) -> pd.DataFrame:
    """
    Slice the dataframe to only include the specified meal types,
    then return the slice.

    :param meal_df: dataframe holding the smartwatch meal info, e.g. as returned from read.meal_info
    :param allowed_meal_types: the meal types to keep; e.g. {"Snack", "Meal"}
    :param verbose: whether to print stuff to console

    :returns: a slice of the original dataframe containing only the allowed meal types
    :raises: AssertionError if meal_type is not a column header in the meal_df

    """
    assert "meal_type" in meal_df

    keep = meal_df["meal_type"].isin(allowed_meal_types)

    if verbose:
        print(f"Discarding: {util.count_dict(meal_df[~keep].meal_type)}")

    retval = meal_df[keep]
    print(f"Kept: {util.count_dict(retval.meal_type)}")

    return retval

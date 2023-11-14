"""
Tests for the smartwatch modules

Some are UT, some are bigger

"""
import pandas as pd

from ema import clean


def test_duplicates():
    """
    Check that the duplicate removing fcn does the right thing

    """
    # Read the test dataframe
    test_df = pd.read_parquet("test/data/duplicates.parquet")

    # Find duplicates
    duplicates = clean.duplicates(test_df)

    # Check they are as expected
    expected_duplicates = [
        *[0] * 5,
        *[1] * 1,
        *[0] * 1,
        *[1] * 1,
        *[0] * 6,
        *[1] * 1,
        *[0] * 125,
    ]

    assert all(duplicates == expected_duplicates)

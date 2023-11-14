"""
Sort out the files needed for the tests

"""
from ema import read


def duplicates_df():
    """
    Create a slice of the real dataframe used for checking for duplicates

    """
    # Read in the smartwatch dataframe
    meal_df = read.all_meal_info()

    # Slice the relevant bits off it
    meal_df = meal_df[meal_df["p_id"] == 25279]
    meal_df = meal_df[["meal_type", "portion_size", "utensil", "location", "p_id"]]

    # Store it
    meal_df.to_parquet("test/data/duplicates.parquet")


def main():
    duplicates_df()


if __name__ == "__main__":
    main()

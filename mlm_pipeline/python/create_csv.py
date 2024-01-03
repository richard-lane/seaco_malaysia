"""
Create a CSV file holding the relevant, cleaned data from file;
ready to run a multi-level model

"""
import sys
import pathlib
import pandas as pd

sys.path.append(str(pathlib.Path(__file__).parent.parents[1].absolute()))

from ema import clean


def main():
    """
    Read, clean data + send to csv

    """
    meal_info = clean.cleaned_smartwatch(keep_catchups=False)

    model_df = pd.DataFrame()

    # Participant ID and entry day
    model_df["p_id"] = meal_info["p_id"]
    model_df["day"] = meal_info["delta"].dt.days

    # Whether each entry was a reponse or not
    model_df["entry"] = (
        meal_info["meal_type"]
        .isin({"Meal", "Drink", "Snack", "No food/drink"})
        .astype(int)
    )

    model_df.to_csv(
        pathlib.Path(__file__).parents[1] / "data" / "model_df.csv", index=False
    )


if __name__ == "__main__":
    main()

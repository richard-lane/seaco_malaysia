"""
Create a CSV file holding the relevant, cleaned data from file;
ready to run a multi-level model

"""
import sys
import pathlib
import pandas as pd

sys.path.append(str(pathlib.Path(__file__).parent.parents[1].absolute()))

from ema import read, clean


def main():
    """
    Read, clean data + send to csv

    """
    meal_info = clean.cleaned_smartwatch(keep_catchups=False)

    model_df = pd.DataFrame()

    # Participant ID and entry day
    model_df["p_id"] = meal_info["p_id"]
    model_df["day"] = meal_info["delta"].dt.days

    # Demographic information
    demographic_df = read._qnaire_df()
    demographic_df = demographic_df[demographic_df["respondent_status"] == 1]
    demographic_df = demographic_df[
        ["residents_id", "respondent_sex", "respondent_ethnicity", "age_dob"]
    ]
    model_df = (
        model_df.reset_index()
        .merge(demographic_df, left_on="p_id", right_on="residents_id", how="left")
        .set_index(model_df.index)
    )

    # Add a column indicating whether the participants are over the age of 12
    model_df["age_group"] = (model_df["age_dob"] > 12).astype(int)

    model_df.rename(
        columns={
            "respondent_sex": "sex",
            "respondent_ethnicity": "ethnicity",
        },
        inplace=True,
    )

    # Whether each day was a weekend or weekday
    model_df["weekend"] = meal_info.index.dayofweek.isin({5, 6}).astype(int)

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

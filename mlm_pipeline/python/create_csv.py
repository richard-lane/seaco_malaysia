"""
Create a CSV file holding the relevant, cleaned data from file;
ready to run a multi-level model

"""

import sys
import pathlib
import numpy as np
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

    # Weekday information
    model_df["weekday"] = meal_info["week_day"]
    model_df["is_weekend"] = (
        meal_info["week_day"].isin({"Saturday", "Sunday"}).astype(int)
    )

    # Ramadan
    model_df["all_in_ramadan"] = meal_info["all_in_ramadan"].astype(int)

    # Demographic information
    demographic_df = read._qnaire_df()
    demographic_df = demographic_df[demographic_df["respondent_status"] == 1]
    demographic_df = demographic_df[
        [
            "residents_id",
            "respondent_sex",
            "respondent_ethnicity",
            "age_dob",
            "phyactq1",
            "smart1_10to17",
            "smart1_7to9",
        ]
    ]
    model_df = (
        model_df.reset_index()
        .merge(demographic_df, left_on="p_id", right_on="residents_id", how="left")
        .set_index(model_df.index)
    )

    # Keep only participants who wore the smartwatch
    keep = (model_df["smart1_10to17"] == 1) | (model_df["smart1_7to9"] == 1)
    keep_participants = model_df.loc[keep, "p_id"].unique()
    print(len(keep_participants))
    model_df = model_df[keep]

    model_df["age_dob"] = model_df["age_dob"].astype(int)

    # Convert sex to 1 or 0 (instead of 1 or 2)
    model_df["respondent_sex"] -= 1

    # Add a column indicating whether the participants are over the age of 12
    model_df["age_group"] = (model_df["age_dob"] > 12).astype(int)

    # Whether each day was a weekend or weekday
    model_df["weekend"] = model_df["Datetime"].dt.dayofweek.isin({5, 6}).astype(int)

    # What day of the week each participant started on
    model_df["first_weekday"] = pd.NA
    for p_id in model_df["p_id"].unique():
        model_df.loc[model_df["p_id"] == p_id, "first_weekday"] = model_df.loc[
            model_df["p_id"] == p_id, "weekday"
        ].iloc[0]

    # How many days each participant spent in school
    model_df["over_2_days_in_school"] = (model_df["phyactq1"] > 2).astype(int)

    # For now, if the participant didn't answer the days in school question just set their answer to NaN
    # TODO a better solution
    model_df.loc[
        (model_df["phyactq1"] == -99) | (model_df["phyactq1"].isnull()),
        "over_2_days_in_school",
    ] = np.nan

    # Whether each entry was a reponse or not
    model_df["entry"] = (
        model_df["meal_type"]
        .isin({"Meal", "Drink", "Snack", "No food/drink"})
        .astype(int)
    )

    model_df.rename(
        columns={
            "respondent_sex": "sex",
            "respondent_ethnicity": "ethnicity",
        },
        inplace=True,
    )

    model_df.to_csv(
        pathlib.Path(__file__).parents[1] / "data" / "model_df.csv", index=False
    )


if __name__ == "__main__":
    main()

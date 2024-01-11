"""
Create demographic plots for all the variables we're looking at

"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def age_hist(demographic_df):
    fig, ax = plt.subplots()

    bins = np.arange(
        np.min(demographic_df["age_dob"]), np.max(demographic_df["age_dob"])
    )
    bins = np.concatenate([bins - 0.2, bins + 0.2])
    bins = np.sort(bins)

    ax.hist(demographic_df["age_dob"], bins=bins)
    ax.set_ylabel("Count")
    ax.set_xlabel("Age")

    fig.tight_layout()
    fig.savefig("mlm_pipeline/outputs/demographics/age_hist.png")


def age_stacked_bar(demographic_df):
    # Count the number of participants in each age
    age_counts = demographic_df["age_dob"].value_counts().sort_index()

    # Split the counts into two groups
    age_group1_counts = age_counts.loc[7:12]
    age_group2_counts = age_counts.loc[13:17]

    # Create a DataFrame for the bar plot
    bar_data = pd.DataFrame({"7-12": age_group1_counts, "13-17": age_group2_counts})

    # Stacked bar plot
    fig, ax = plt.subplots()
    bar_data.plot(kind="bar", stacked=True, ax=ax)
    ax.set_ylabel("Count")
    ax.set_xlabel("Age Group")

    fig.tight_layout()
    fig.savefig("mlm_pipeline/outputs/demographics/age_stacked.png")


def age_hists(demographic_df):
    """
    Histogram of ages, and stacked histogram showing age groups

    """
    age_hist(demographic_df)
    # age_stacked_bar(demographic_df)


def ethnicity_plot():
    """ """


def school_plot():
    """ """


def sex_plot():
    """ """


def weekdays_plot():
    """ """


def main():
    if not os.path.isdir("mlm_pipeline/outputs/demographics/"):
        os.mkdir("mlm_pipeline/outputs/demographics/")

    entries_df = pd.read_csv("mlm_pipeline/data/model_df.csv")

    # Only keep the first row for each participant
    entries_df = entries_df.drop_duplicates(subset="p_id")

    age_hists(entries_df)


if __name__ == "__main__":
    main()

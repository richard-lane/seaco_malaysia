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
        np.min(demographic_df["age_dob"]), np.max(demographic_df["age_dob"] + 1)
    )
    bins = np.concatenate([bins - 0.2, bins + 0.2])
    bins = np.sort(bins)

    ax.hist(demographic_df["age_dob"], bins=bins)
    ax.set_ylabel("Count")
    ax.set_xlabel("Age")

    fig.tight_layout()
    fig.savefig("mlm_pipeline/outputs/demographics/age_hist.png")


def age_stacked_bar(demographic_df):
    # Stacked bar plot
    fig, ax = plt.subplots()

    bottom = 0
    for age in range(7, 13):
        count = (demographic_df["age_dob"] == age).sum()
        ax.bar(0, count, bottom=bottom)
        if count > 2:
            ax.text(0, bottom + count / 2, age, ha="center", va="center")
        bottom += count

    bottom = 0
    for age in range(13, 19):
        count = (demographic_df["age_dob"] == age).sum()
        ax.bar(1, count, bottom=bottom)
        ax.text(1, bottom + count / 2, age, ha="center", va="center")
        bottom += count

    ax.set_xticks((0, 1), ("7-12", "13-17"))

    ax.set_ylabel("Count")
    ax.set_xlabel("Age group")

    fig.tight_layout()
    fig.savefig("mlm_pipeline/outputs/demographics/age_stacked.png")


def age_hists(demographic_df):
    """
    Histogram of ages, and stacked histogram showing age groups

    """
    age_hist(demographic_df)
    age_stacked_bar(demographic_df)


def ethnicity_plot(demographic_df):
    ethnicities = demographic_df["ethnicity"].value_counts()

    total = len(demographic_df)

    def autopct(val):
        return f"{int(val * total / 100)}\n({val:.1f}%)"

    fig, axis = plt.subplots()
    axis.pie(
        ethnicities,
        labels=[f"Ethnicity {int(i)}" for i in ethnicities.index],
        autopct=autopct,
    )
    fig.tight_layout()
    fig.savefig("mlm_pipeline/outputs/demographics/ethnicity.png")


def sex_plot(demographic_df):
    """ """
    sexes = demographic_df["sex"].value_counts()

    total = len(demographic_df)

    def autopct(val):
        return f"{int(val * total / 100)}\n({val:.1f}%)"

    fig, axis = plt.subplots()
    axis.pie(
        sexes,
        labels=["Female" if i else "Male" for i in sexes.index],
        autopct=autopct,
    )
    fig.tight_layout()
    fig.savefig("mlm_pipeline/outputs/demographics/sexes.png")


def school_hist(demographic_df):
    fig, ax = plt.subplots()

    bins = np.arange(0, 8)
    bins = np.concatenate([bins - 0.2, bins + 0.2])
    bins = np.sort(bins)

    ax.hist(demographic_df["phyactq1"], bins=bins)
    ax.set_ylabel("Count")
    ax.set_xlabel("Number of days in school")

    fig.tight_layout()
    fig.savefig("mlm_pipeline/outputs/demographics/school_hist.png")


def school_stacked_bar(demographic_df):
    # Stacked bar plot
    fig, ax = plt.subplots()

    bottom = 0
    for days in range(0, 3):
        count = (demographic_df["phyactq1"] == days).sum()
        ax.bar(0, count, bottom=bottom)
        if count > 2:
            ax.text(0, bottom + count / 2, days, ha="center", va="center")
        bottom += count

    bottom = 0
    for days in range(3, 8):
        count = (demographic_df["phyactq1"] == days).sum()
        ax.bar(1, count, bottom=bottom)
        if count > 2:
            ax.text(1, bottom + count / 2, days, ha="center", va="center")
        bottom += count

    ax.set_xticks((0, 1), ("0-2", "2-7"))

    ax.set_xlabel("Days in school")
    ax.set_ylabel("Count")

    fig.tight_layout()
    fig.savefig("mlm_pipeline/outputs/demographics/school_stacked.png")


def school_plot(demographic_df):
    school_hist(demographic_df)
    school_stacked_bar(demographic_df)


def weekdays_plot():
    """ """


def main():
    if not os.path.isdir("mlm_pipeline/outputs/demographics/"):
        os.mkdir("mlm_pipeline/outputs/demographics/")

    entries_df = pd.read_csv("mlm_pipeline/data/model_df.csv")

    # Only keep the first row for each participant
    entries_df = entries_df.drop_duplicates(subset="p_id")

    age_hists(entries_df)

    ethnicity_plot(entries_df)

    sex_plot(entries_df)

    school_plot(entries_df)


if __name__ == "__main__":
    main()

"""
Simple statistical stuff that's easier in python

"""
import numpy as np
import pandas as pd


def main():
    entries_df = pd.read_csv("mlm_pipeline/data/model_df.csv")

    # How many positive entries
    print(
        f"Fraction of positive entries: {100 * entries_df['entry'].sum() / len(entries_df):.2f}%"
    )

    # How many positive entries, split by sex
    for sex, group in entries_df.groupby("sex"):
        print(f"{sex} {100 * group['entry'].sum() / len(group):.2f}%")


if __name__ == "__main__":
    main()

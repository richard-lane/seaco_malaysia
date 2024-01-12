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


if __name__ == "__main__":
    main()

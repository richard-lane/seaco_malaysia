"""
Plotting tools

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_integrals(
    times: np.ndarray,
    points: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    fig=None,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a figure showing acceleration and its first two integrals (velocity and displacement)

    :param times: array of times
    :param points: tuple of arrays of acceleration, vely, position values
    :param fig: optional figure, if one exists already
    :param kwargs: further keyword arguments to pass to plt.plot

    :returns: the figure and axis

    """
    if fig is None:
        fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
    else:
        axes = fig.axes

    if "color" not in kwargs:
        kwargs["color"] = "k"

    if "linewidth" not in kwargs:
        kwargs["linewidth"] = 0.5

    for axis, data, label in zip(
        axes, points, [r"Accel / m/s$^{-2}$", r"Vely / ms$^{-1}$", r"Posn / m"]
    ):
        axis.axhline(0, linewidth=0.5, color="k", linestyle="--")

        axis.set_ylabel(label)

        axis.plot(times, data, **kwargs)


def entry_time_hist(
    meal_timing_df: pd.DataFrame,
    *,
    cumulative: bool = False,
    granularity: str = "1D",
    fig_ax: tuple = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a histogram of the times of each type of entry in `meal_info`

    :param meal_timing_df: dataframe of smartwatch entries
    :param cumulative: whether to plot a cumulative histogram
    :param granularity: bin granularity; "D", "H", etc. Or bins
    :param fig_ax: optional figure and axis to plot on; creates a new figure if not specified

    :returns

    """
    min_time, max_time = meal_timing_df.index.min(), meal_timing_df.index.max()

    fig, axis = plt.subplots() if fig_ax is None else fig_ax
    bins = (
        pd.date_range(min_time, max_time, freq=granularity)
        if isinstance(granularity, str)
        else granularity
    )

    # Sort the meal type labels
    labels = meal_timing_df["meal_type"].unique()
    labels = [
        *[l for l in labels if l.startswith("No ")],
        *[l for l in labels if l.startswith("Catch-up")],
        *labels,
    ]  # Put them in the desired order
    labels = list(dict.fromkeys(labels))  # Remove duplicates, preserve order

    # Get a list of Series giving the times of each meal type
    data = [
        meal_timing_df[meal_timing_df["meal_type"] == meal_type].index
        for meal_type in labels
    ]

    axis.hist(data, bins=bins, stacked=True, label=labels, cumulative=cumulative)
    axis.legend()

    # If fig and ax specified, we shouldn't take control of the formatting
    if fig_ax is None:
        fig.tight_layout()
        fig.autofmt_xdate()

    return fig, axis

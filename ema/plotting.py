"""
Plotting tools

"""
import numpy as np
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

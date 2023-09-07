"""
Unit test

"""
import sys
import pathlib

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from ema import util


def test_integrate():
    """
    Test integration

    """
    x = np.linspace(0, 2 * np.pi)
    y = np.sin(x)

    integral = util.integrate(y, dx=x[1] - x[0])

    assert np.allclose(integral, 1 - np.cos(x), atol=1e-2)

"""
This module stores all the utility functions
that are required for the implementation of
of any Machine Learning logic but not associated
to any particular technique.
"""

import numpy as np


def euclidean_distance(data_points):
    """
    It calculates euclidean distance between two
    set of points.

    :param data_points: np.narray for two set of points
    :return: Euclidean Distance
    """
    assert len(data_points) == 2
    # Sum of Squared distance as in Euclidean distance
    return np.sqrt(np.sum((data_points[0] - data_points[1]) ** 2))

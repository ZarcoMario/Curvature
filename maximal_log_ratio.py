'''
The ratio of the target distance to the alternative distance can be calculated for each (x, y) position.
Ratio values closer to 1 suggest a position near the middle, higher values indicate a deviation toward the alternative response.

Maximal log ratio: the point that maximizes the log distance ratio
https://link.springer.com/article/10.3758/s13428-018-01194-x#Fig7

NOTE: t2 is always the correct target, t1 is the alternative target
'''

import numpy as np


def maximal_log_ratio_2d(x_tr, y_tr, t1, t2):
    '''
    Max log ratio in 2D (See the document Geometric Descriptors of Curvature for more details)
    :param x_tr: trajectory data along a first dimension (e.g. x)
    :param y_tr: trajectory data along a second dimension (e.g. z)
    :param t1: alternative target
    :param t2: correct target
    :return:
        - Max log ratio (MLR)
        - Coordinate of the first dimension corresponding to MLR
        - Coordinate of the second dimension corresponding to MLR
    '''
    d_1, d_2 = [], []
    for x, y in zip(x_tr, y_tr):
        # Distance to alternative target
        d_1 = np.append(d_1, np.sqrt((x - t1[0]) ** 2 + (y - t1[1]) ** 2))
        # Distance to correct target
        d_2 = np.append(d_2, np.sqrt((x - t2[0]) ** 2 + (y - t2[1]) ** 2))

    # Log Ratio
    log_ratio = np.log(d_2 / d_1)
    # Max Log Ratio
    max_log_ratio = np.max(log_ratio)
    # Index corresponding to max log ratio
    idx = np.argwhere(log_ratio == max_log_ratio)[0][0]

    return max_log_ratio, x_tr[idx], y_tr[idx]


def maximal_log_ratio_3d(x_tr, y_tr, z_tr, t1, t2):
    '''
    Max log ratio in 3D (See the document Geometric Descriptors of Curvature for more details)
    :param x_tr: trajectory data along a first dimension (e.g. x)
    :param y_tr: trajectory data along a second dimension (e.g. z)
    :param z_tr: trajectory data along a second dimension (e.g. y)
    :param t1: alternative target
    :param t2: correct target
    :return:
        - Max log ratio (MLR)
        - Coordinate of the first dimension corresponding to MLR
        - Coordinate of the second dimension corresponding to MLR
    '''
    d_1, d_2 = [], []
    for x, y, z in zip(x_tr, y_tr, z_tr):
        # Distance to alternative target
        d_1 = np.append(d_1, np.sqrt((x - t1[0]) ** 2 + (y - t1[1]) ** 2 + (z - t1[2]) ** 2))
        # Distance to correct target
        d_2 = np.append(d_2, np.sqrt((x - t2[0]) ** 2 + (y - t2[1]) ** 2 + (z - t2[2]) ** 2))

    # Log Ratio
    log_ratio = np.log(d_2 / d_1)
    # Max Log Ratio
    max_log_ratio = np.max(log_ratio)
    # Index corresponding to max log ratio
    idx = np.argwhere(log_ratio == max_log_ratio)[0][0]

    return np.max(max_log_ratio), x_tr[idx], y_tr[idx], z_tr[idx]
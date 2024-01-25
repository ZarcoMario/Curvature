import numpy as np


def _get_angle_2d(pi, pn, pf):
    '''
    Calculate the signed angle between two-dimensional vectors
        according to the maximum perpendicular deviation convention
    :param pi: first sample
    :param pn: n-th sample
    :param pf: last sample
    :return: angle
    '''
    if pf[0] < 0:
        v2 = pn - pi
        v1 = pf - pi
    else:
        v1 = pn - pi
        v2 = pf - pi

    theta = np.arctan2(np.linalg.det(np.row_stack((v2, v1))), np.dot(v1, v2))

    return theta


def total_curvature_2d(x_tr, y_tr):
    '''
    Calculate Total curvature in 2D (See the document Geometric Descriptors of Curvature for more details)
    :param x_tr: trajectory data along a first dimension (e.g. x)
    :param y_tr: trajectory data along a second dimension (e.g. z)
    :return: total curvature
    '''
    xy = np.column_stack((x_tr, y_tr))
    # Initial sample
    pi = xy[0, :]
    # Final sample
    pf = xy[-1, :]
    # Rest of the samples
    p_tr = xy[1:-1, :]

    dis = []
    ang = []
    for pn in p_tr:
        # Find Angle
        theta = _get_angle_2d(pi, pn, pf)
        # Perpendicular distance to each sample in p_tr
        d = np.linalg.norm(pn - pi) * np.sin(np.abs(theta))
        dis = np.append(dis, d)
        ang = np.append(ang, theta)

    # Total curvature
    tot_cur = np.mean(np.multiply(np.sign(ang), dis))

    return tot_cur


def _get_angle_3d(pi, pn, pf):
    '''
    Calculate the non-signed angle between two-dimensional vectors
        according to the maximum perpendicular deviation convention
    :param pi: first sample
    :param pn: n-th sample
    :param pf: final sample
    :return: angle
    '''
    v1 = pn - pi
    v2 = pf - pi

    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)

    arg = np.dot(v1, v2) / (v1_norm * v2_norm)

    theta = np.arccos(arg)

    return theta


def _get_sign_3d(p1, p2, p3, pn):
    '''
    Calculate the sign of maximum perpendicular deviation
    :param p1: vector to first sample
    :param p2: normal vector
    :param p3: vector to final sample
    :param pn: vector to n-th sample
    :return: sign
    '''
    if p3[0] < 0:
        vb = p3 - p1
        va = p2 - p1
    else:
        va = p3 - p1
        vb = p2 - p1

    va_n = va / np.linalg.norm(va)
    vb_n = vb / np.linalg.norm(vb)

    axb = np.cross(va_n, vb_n)
    vp_n = axb / np.linalg.norm(axb)

    vt = pn - p1
    vt_n = vt / np.linalg.norm(vt)

    theta = np.arcsin(np.dot(vp_n, vt_n))

    return np.sign(theta)


def total_curvature_3d(x_tr, y_tr, z_tr):
    '''
    Calculate total curvature in 3D (See the document Geometric Descriptors of Curvature for more details)
        NOTE: This function assumes a different coordinate system from UNITY 3D.
    :param x_tr: trajectory data along a first dimension (e.g. x)
    :param y_tr: trajectory data along a second dimension (e.g. z)
    :param z_tr: trajectory data along a third dimension (e.g. y)
    :return: total curvature
    '''
    xy = np.column_stack((x_tr, y_tr, z_tr))
    # Initial sample
    pi = xy[0, :]
    # Final sample
    pf = xy[-1, :]
    # Rest of the samples
    p_tr = xy[1:-1, :]
    # Normal Vector
    p2 = np.array([pf[0], pf[1], pi[2]])

    dis = []
    sign = []
    for pn in p_tr:
        # Find Angle
        theta = _get_angle_3d(pi, pn, pf)
        # Find Sign
        s = _get_sign_3d(pi, p2, pf, pn)
        # Perpendicular distance to each sample in p_tr
        d = np.linalg.norm(pn - pi) * np.sin(theta)
        dis = np.append(dis, d)
        sign = np.append(sign, s)

    # Total Curvature
    tot_cur = np.mean(np.multiply(sign, dis))

    return tot_cur

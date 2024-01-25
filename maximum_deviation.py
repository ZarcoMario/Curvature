import numpy as np


def _get_angle_2d(pi, pn, pf):
    '''
    Calculate
    :param pi:
    :param pn:
    :param pf:
    :return:
    '''
    if pf[0] < 0:
        v2 = pn - pi
        v1 = pf - pi
    else:
        v1 = pn - pi
        v2 = pf - pi

    theta = np.arctan2(np.linalg.det(np.row_stack((v2, v1))), np.dot(v1, v2))

    return theta


def maximum_deviation_2d(x_tr, y_tr):
    '''

    :param x_tr:
    :param y_tr:
    :return:
    '''
    xy = np.column_stack((x_tr, y_tr))
    # Initial Sample
    pi = xy[0, :]
    # Final Sample
    pf = xy[-1, :]
    # Rest of the Samples
    p_tr = xy[1:-1, :]

    dis = []
    ang = []
    for pn in p_tr:
        # Find angle
        theta = _get_angle_2d(pi, pn, pf)
        # Perpendicular distance to each sample in p_tr
        d = np.linalg.norm(pn - pi) * np.sin(np.abs(theta))
        dis = np.append(dis, d)
        ang = np.append(ang, theta)

    # Maximum Distance
    max_dis = np.max(dis)
    # Maximum Deviation (Maximum distance normalized by distance between pi and pf)
    max_dev = max_dis / np.linalg.norm(pf - pi)
    # Index
    idx = np.argwhere(max_dis == dis)[0][0]

    # Sample corresponding to Maximum distance
    # NOTE: Change this if you want the sample corresponding to maximum deviation
    x_max = p_tr[idx, 0]
    y_max = p_tr[idx, 1]

    return np.sign(ang[idx]) * max_dev, np.sign(ang[idx]) * max_dis, x_max, y_max, idx


def _get_sign_3d(p1, p2, p3, pn):
    '''
    
    :param p1:
    :param p2:
    :param p3:
    :param pn:
    :return:
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


def _get_angle_3d(pi, pn, pf):
    '''

    :param pi:
    :param pn:
    :param pf:
    :return:
    '''
    v1 = pn - pi
    v2 = pf - pi

    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)

    arg = np.dot(v1, v2) / (v1_norm * v2_norm)

    theta = np.arccos(arg)

    return theta


def maximum_deviation_3d(x_tr, y_tr, z_tr):
    '''

    :param x_tr:
    :param y_tr:
    :param z_tr:
    :return:
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

    # Maximum Distance
    max_dis = np.max(dis)
    # Maximum Deviation (Maximum distance normalized by distance between pi and pf)
    max_dev = max_dis / np.linalg.norm(pf - pi)
    # Index
    idx = np.argwhere(max_dis == dis)[0][0]

    # Sample corresponding to Maximum distance
    # NOTE: Change this if you want the sample corresponding to maximum deviation
    x_max = p_tr[idx, 0]
    y_max = p_tr[idx, 1]
    z_max = p_tr[idx, 2]

    return sign[idx] * max_dev, sign[idx] * max_dis, x_max, y_max, z_max, idx
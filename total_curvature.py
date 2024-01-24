import numpy as np


def _get_angle_2d(pi, pn, pf):
    if pf[0] < 0:
        v2 = pn - pi
        v1 = pf - pi
    else:
        v1 = pn - pi
        v2 = pf - pi

    theta = np.arctan2(np.linalg.det(np.row_stack((v2, v1))), np.dot(v1, v2))

    return theta


def total_curvature_2d(x_tr, y_tr):
    xy = np.column_stack((x_tr, y_tr))
    pi = xy[0, :]
    pf = xy[-1, :]
    p_tr = xy[1:-1, :]

    dis = []
    ang = []
    for pn in p_tr:
        theta = _get_angle_2d(pi, pn, pf)
        d = np.linalg.norm(pn - pi) * np.sin(np.abs(theta))
        dis = np.append(dis, d)
        ang = np.append(ang, theta)

    tot_cur = np.mean(np.multiply(np.sign(ang), dis))

    return tot_cur


def _get_angle_3d(pi, pn, pf):
    v1 = pn - pi
    v2 = pf - pi

    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)

    arg = np.dot(v1, v2) / (v1_norm * v2_norm)

    theta = np.arccos(arg)

    return theta


def _get_sign_3d(p1, p2, p3, pn):
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
    xy = np.column_stack((x_tr, y_tr, z_tr))
    pi = xy[0, :]
    pf = xy[-1, :]
    p_tr = xy[1:-1, :]
    p2 = np.array([pf[0], pf[1], pi[2]])

    dis = []
    sign = []
    for pn in p_tr:
        theta = _get_angle_3d(pi, pn, pf)
        s = _get_sign_3d(pi, p2, pf, pn)
        d = np.linalg.norm(pn - pi) * np.sin(theta)
        dis = np.append(dis, d)
        sign = np.append(sign, s)

    tot_cur = np.mean(np.multiply(sign, dis))

    return tot_cur

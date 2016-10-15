from math import *
from numba import jit
import sys


@jit(nopython=True)
def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


@jit(nopython=True)
def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@jit(nopython=True)
def vector(a, b):
    return (b[0] - a[0],
            b[1] - a[1],
            b[2] - a[2])


@jit(nopython=True)
def sign(a):
    if a > 0:
        return 1
    elif a < 0:
        return -1
    else:
        return 0


@jit(nopython=True)
def norm(a):
    return (a[0] ** 2 + a[1] ** 2 + a[2] ** 2) ** 0.5


@jit(nopython=True)
def t4(c, d, l1, l2, l4, n, t1, t2, x, y, z):
    bx = l1 * sin(t2) * cos(t1)
    by = l1 * sin(t1) * sin(t2)
    bz = l1 * cos(t2)
    b = (bx, by, bz)

    bc = vector(b, c)
    de = vector(d, (x, y, z))
    s = sign(dot(n, cross(bc, de)))

    return s * acos(dot(bc, de) / (l2 * l4))


@jit(nopython=True)
def solve_ik(lengths, constraints, epsilon=sys.float_info.epsilon):
    # Unpack
    l1, l2, l3, l4 = lengths
    x, y, z, phi = constraints

    # Compute theta0
    t0 = atan2(l3, (-l3 ** 2 + x ** 2 + y ** 2) ** 0.5)

    # Compute n
    _a = atan2(y, x)

    nx_1 = sin(t0 - _a)
    ny_1 = cos(t0 - _a)
    n_1 = (nx_1, ny_1, 0)

    nx_2 = -sin(t0 + _a)
    ny_2 = cos(t0 + _a)
    n_2 = (nx_2, ny_2, 0)

    # Compute d
    _b = l4 * sin(phi)
    _c = l4 * cos(phi)

    dx_1 = -_b * ny_1 + x
    dy_1 = _b * nx_1 + y
    dz_1 = -_c + z
    d_1 = (dx_1, dy_1, dz_1)

    dx_2 = -_b * ny_2 + x
    dy_2 = _b * nx_2 + y
    dz_2 = -_c + z
    d_2 = (dx_2, dy_2, dz_2)

    # Compute c
    cx_1 = dx_1 - l3 * nx_1
    cy_1 = dy_1 - l3 * ny_1
    c_1 = (cx_1, cy_1, dz_1)

    cx_2 = dx_2 + l3 * nx_2
    cy_2 = dy_2 + l3 * ny_2
    c_2 = (cx_2, cy_2, dz_2)

    # Compute t3
    t3_1 = -acos(-(cx_1 ** 2 + cy_1 ** 2 + dz_1 ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)) + pi
    t3_2 = -t3_1
    t3_3 = -acos(-(cx_2 ** 2 + cy_2 ** 2 + dz_2 ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)) + pi
    t3_4 = -t3_3

    # Compute hand-wavy t2
    k_1 = sign(cx_1 ** 2 + cy_1 ** 2 - (l2 * sin(t3_1)) ** 2)
    k_2 = sign(cx_1 ** 2 + cy_1 ** 2 - (l2 * sin(t3_2)) ** 2)
    k_3 = sign(cx_2 ** 2 + cy_2 ** 2 - (l2 * sin(t3_3)) ** 2)
    k_4 = sign(cx_2 ** 2 + cy_2 ** 2 - (l2 * sin(t3_4)) ** 2)

    t2_1 = -2 * atan(2 * (l2 * sin(t3_1) - k_1 * (-dz_1 ** 2 + l1 ** 2 + 2 * l1 * l2 * cos(t3_1) + l2 ** 2) ** 0.5)
                     * cos(t3_1 / 2) ** 2 / ((cos(t3_1) + 1) * (dz_1 + l1 - 2 * l2 * sin(t3_1 / 2) ** 2 + l2)))
    t2_2 = -2 * atan(2 * (l2 * sin(t3_2) - k_2 * (-dz_1 ** 2 + l1 ** 2 + 2 * l1 * l2 * cos(t3_2) + l2 ** 2) ** 0.5)
                     * cos(t3_2 / 2) ** 2 / ((cos(t3_2) + 1) * (dz_1 + l1 - 2 * l2 * sin(t3_2 / 2) ** 2 + l2)))
    t2_3 = -2 * atan(2 * (l2 * sin(t3_3) + k_3 * (-dz_2 ** 2 + l1 ** 2 + 2 * l1 * l2 * cos(t3_3) + l2 ** 2) ** 0.5)
                     * cos(t3_3 / 2) ** 2 / ((cos(t3_3) + 1) * (dz_2 + l1 - 2 * l2 * sin(t3_3 / 2) ** 2 + l2)))
    t2_4 = -2 * atan(2 * (l2 * sin(t3_4) + k_4 * (-dz_2 ** 2 + l1 ** 2 + 2 * l1 * l2 * cos(t3_4) + l2 ** 2) ** 0.5)
                     * cos(t3_4 / 2) ** 2 / ((cos(t3_4) + 1) * (dz_2 + l1 - 2 * l2 * sin(t3_4 / 2) ** 2 + l2)))

    # Compute t1
    t1_1 = atan2(dy_1, dx_1) - atan2(l3, l1 * sin(t2_1) + l2 * sin(t2_1 + t3_1))
    t1_2 = atan2(dy_1, dx_1) - atan2(l3, l1 * sin(t2_2) + l2 * sin(t2_2 + t3_2))
    t1_3 = atan2(dy_2, dx_2) - atan2(l3, l1 * sin(t2_3) + l2 * sin(t2_3 + t3_3))
    t1_4 = atan2(dy_2, dx_2) - atan2(l3, l1 * sin(t2_4) + l2 * sin(t2_4 + t3_4))

    # Compute t4
    t4_1 = t4(c_1, d_1, l1, l2, l4, n_1, t1_1, t2_1, x, y, z)
    t4_2 = t4(c_1, d_1, l1, l2, l4, n_1, t1_2, t2_2, x, y, z)
    t4_3 = -t4(c_2, d_2, l1, l2, l4, n_2, t1_3, t2_3, x, y, z)
    t4_4 = -t4(c_2, d_2, l1, l2, l4, n_2, t1_4, t2_4, x, y, z)

    # Return
    return (t1_1, t2_1, t3_1, t4_1), \
           (t1_2, t2_2, t3_2, t4_2), \
           (t1_3, t2_3, t3_3, t4_3), \
           (t1_4, t2_4, t3_4, t4_4)
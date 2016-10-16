from math import *
from numba import jit


@jit(nopython=True)
def solve_fk(lengths, angles):
    l1, l2, l3, l4 = lengths
    t1, t2, t3, t4 = angles

    # p0 is trivial
    x0, y0, z0 = 0, 0, 0
    p0 = (x0, y0, z0)

    # Find p1
    x1 = l1 * sin(t2) * cos(t1)
    y1 = l1 * sin(t1) * sin(t2)
    z1 = l1 * cos(t2)
    p1 = (x1, y1, z1)

    # Find p2
    x2 = x1 + l2 * sin(t2 + t3) * cos(t1)
    y2 = y1 + l2 * sin(t1) * sin(t2 + t3)
    z2 = z1 + l2 * cos(t2 + t3)
    p2 = (x2, y2, z2)

    # Find p3
    x3 = x2 + -l3 * sin(t1)
    y3 = y2 + l3 * cos(t1)
    z3 = z2
    p3 = (x3, y3, z3)

    # Find p4
    x4 = x3 + l4 * sin(t2 + t3 + t4) * cos(t1)
    y4 = y3 + l4 * sin(t1) * sin(t2 + t3 + t4)
    z4 = z3 + l4 * cos(t2 + t3 + t4)
    p4 = (x4, y4, z4)

    # Return
    return p0, p1, p2, p3, p4

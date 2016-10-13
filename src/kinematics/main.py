from math import *
from functools import lru_cache


class FK:
    @staticmethod
    def solve(lengths, angles):
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

        # Construct list
        points = (p0, p1, p2, p3, p4)

        return points


class IK:
    @staticmethod
    @lru_cache()
    def _n_1(t0, x, y):
        nx = sin(t0 - atan2(y, x))
        ny = cos(t0 - atan2(y, x))
        nz = 0

        return nx, ny, nz

    @staticmethod
    @lru_cache()
    def _n_2(t0, x, y):
        nx = -sin(t0 + atan2(y, x))
        ny = cos(t0 + atan2(y, x))
        nz = 0

        return nx, ny, nz

    @staticmethod
    @lru_cache()
    def _d_1(l4, phi, t0, x, y, z):
        dx = -l4 * sin(phi) * cos(t0 - atan2(y, x)) + x
        dy = l4 * sin(phi) * sin(t0 - atan2(y, x)) + y
        dz = -l4 * cos(phi) + z

        return dx, dy, dz

    @staticmethod
    @lru_cache()
    def _d_2(l4, phi, t0, x, y, z):
        dx = -l4 * sin(phi) * cos(t0 + atan2(y, x)) + x
        dy = -l4 * sin(phi) * sin(t0 + atan2(y, x)) + y
        dz = -l4 * cos(phi) + z

        return dx, dy, dz

    @staticmethod
    @lru_cache()
    def _c_1(d, l3, t0, x, y):
        cx = d[0] - l3 * sin(t0 - atan2(y, x))
        cy = d[1] - l3 * cos(t0 - atan2(y, x))
        cz = d[2]

        return cx, cy, cz

    @staticmethod
    @lru_cache()
    def _c_2(d, l3, t0, x, y):
        cx = d[0] - l3 * sin(t0 + atan2(y, x))
        cy = d[1] + l3 * cos(t0 + atan2(y, x))
        cz = d[2]

        return cx, cy, cz

    @staticmethod
    @lru_cache()
    def _t3_1(c, l1, l2):
        return -acos(-(c[0] ** 2 + c[1] ** 2 + c[2] ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)) + pi

    @classmethod
    @lru_cache()
    def _t3_2(cls, c, l1, l2):
        return -cls._t3_1(c, l1, l2)

    @staticmethod
    @lru_cache()
    def _t2_1(c, l1, l2, t3):
        return -2 * atan(2 * (l2 * sin(t3) - sqrt(
            -c[2] ** 2 + l1 ** 2 + 2 * l1 * l2 * cos(t3) + l2 ** 2)) * cos(
            t3 / 2) ** 2 / ((cos(t3) + 1) * (c[2] + l1 - 2 * l2 * sin(t3 / 2) ** 2 + l2)))

    @staticmethod
    @lru_cache()
    def _t2_2(c, l1, l2, t3):
        return -2 * atan(2 * (l2 * sin(t3) + sqrt(
            -c[2] ** 2 + l1 ** 2 + 2 * l1 * l2 * cos(t3) + l2 ** 2)) * cos(
            t3 / 2) ** 2 / ((cos(t3) + 1) * (c[2] + l1 - 2 * l2 * sin(t3 / 2) ** 2 + l2)))

    @staticmethod
    @lru_cache()
    def _t1_1(c):
        return atan2(c[1], c[0])

    @classmethod
    @lru_cache()
    def _t1_2(cls, c):
        return cls._t1_1(c) + pi

    @staticmethod
    def cross(a, b):
        return (a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0])

    @staticmethod
    def dot(a, b):
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    @staticmethod
    def vector(a, b):
        return (b[0] - a[0],
                b[1] - a[1],
                b[2] - a[2])

    @staticmethod
    def sign(a):
        if a > 0:
            return 1
        elif a < 0:
            return -1
        else:
            return 0

    @staticmethod
    def norm(a):
        return (a[0] ** 2 + a[1] ** 2 + a[2] ** 2) ** 0.5

    @classmethod
    @lru_cache()
    def _t4_1(cls, c, d, l1, n, t1, t2, x, y, z):
        x1 = l1 * sin(t2) * cos(t1)
        y1 = l1 * sin(t1) * sin(t2)
        z1 = l1 * cos(t2)
        p1 = (x1, y1, z1)

        bc = cls.vector(p1, c)
        de = cls.vector(d, (x, y, z))
        s = cls.sign(cls.dot(n, cls.cross(bc, de)))

        return s * acos(cls.dot(bc, de) / (cls.norm(bc) * cls.norm(de)))

    @classmethod
    @lru_cache()
    def _t4_2(cls, c, d, l1, n, t1, t2, x, y, z):
        return 2 * pi - cls._t4_1(c, d, l1, n, t1, t2, x, y, z)

    @classmethod
    def solve(cls, lengths, constraints):
        # Create data dictionary
        data = {}

        # Fill dictionary
        l1, l2, l3, l4 = lengths
        x, y, z, phi = constraints

        # Define solutions
        solutions = []

        # Define paths
        paths = (
            (cls._n_1, cls._d_1, cls._c_1, cls._t3_1, cls._t2_1, cls._t1_1, cls._t4_1),
            (cls._n_1, cls._d_1, cls._c_1, cls._t3_2, cls._t2_1, cls._t1_1, cls._t4_1),
            (cls._n_2, cls._d_2, cls._c_2, cls._t3_1, cls._t2_2, cls._t1_2, cls._t4_2),
            (cls._n_2, cls._d_2, cls._c_2, cls._t3_2, cls._t2_2, cls._t1_2, cls._t4_2)
        )

        # Compute theta0
        t0 = atan2(l3, (-l3 ** 2 + x ** 2 + y ** 2) ** 0.5)

        # For each path, compute solutions
        for path in paths:
            try:
                n = path[0](t0, x, y)
                d = path[1](l4, phi, t0, x, y, z)
                c = path[2](d, l3, t0, x, y)
                t3 = path[3](c, l1, l2)
                t2 = path[4](c, l1, l2, t3)
                t1 = path[5](c)
                t4 = path[6](c, d, l1, n, t1, t2, x, y, z)

                solution = (t1, t2, t3, t4)
                solutions.append(solution)
            except (ValueError, ZeroDivisionError):
                pass

        return solutions

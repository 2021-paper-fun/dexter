from math import sqrt, cos, sin, acos, degrees, radians, log, pi, atan2
from cmath import exp
from scipy.integrate import quad
import numpy as np


LENGTH_MIN_DEPTH = 5
LENGTH_ERROR = 1e-12
USE_SCIPY_QUAD = True


def segment_length(curve, start, end, start_point, end_point,
                   error=LENGTH_ERROR, min_depth=LENGTH_MIN_DEPTH, depth=0):
    mid = (start + end) / 2
    mid_point = curve.point(mid)
    length = abs(end_point - start_point)
    first_half = abs(mid_point - start_point)
    second_half = abs(end_point - mid_point)

    length2 = first_half + second_half
    if (length2 - length > error) or (depth < min_depth):
        # Calculate the length of each segment:
        depth += 1
        return (segment_length(curve, start, mid, start_point, mid_point,
                               error, min_depth, depth) +
                segment_length(curve, mid, end, mid_point, end_point,
                               error, min_depth, depth))

    return length2


class Line:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __repr__(self):
        return 'Line(start={}, end={})'.format(self.start, self.end)

    def __getitem__(self, item):
        return self.bpoints()[item]

    def bpoints(self):
        return self.start, self.end

    def point(self, t):
        """returns the coordinates of the Bezier curve evaluated at t."""
        distance = self.end - self.start
        return self.start + distance * t

    def length(self, error=None, min_depth=None):
        """returns the length of the line segment between t0 and t1."""
        return abs(self.end - self.start)

    def transform(self, matrix):
        self.start = matrix @ self.start
        self.end = matrix @ self.end


class QuadraticBezier:
    def __init__(self, start, control, end):
        self.start = start
        self.end = end
        self.control = control

        self.length_info = {'length': None, 'bpoints': None}

    def __repr__(self):
        return 'QuadraticBezier(start={}, control={}, end={})'.format(self.start, self.control, self.end)

    def __getitem__(self, item):
        return self.bpoints()[item]

    def point(self, t):
        return (1 - t) ** 2 * self.start + 2 * (1 - t) * t * self.control + t ** 2 * self.end

    def length(self, error=None, min_depth=None):
        if self.length_info['bpoints'] == self.bpoints():
            return self.length_info['length']

        a = self.start - 2*self.control + self.end
        b = 2*(self.control - self.start)
        a_dot_b = a.real*b.real + a.imag*b.imag

        if abs(a) < 1e-12:
            s = abs(b)
        elif abs(a_dot_b + abs(a)*abs(b)) < 1e-12:
            k = abs(b)/abs(a)
            if k >= 2:
                s = abs(b) - abs(a)
            else:
                s = abs(a)*(k**2/2 - k + 1)
        else:
            # For an explanation of this case, see
            # http://www.malczak.info/blog/quadratic-bezier-curve-length/
            a = 4 * (a.real ** 2 + a.imag ** 2)
            b = 4 * (a.real * b.real + a.imag * b.imag)
            c = b.real ** 2 + b.imag ** 2

            s_abc = 2 * sqrt(a + b + c)
            a2 = sqrt(a)
            a32 = 2 * a * a2
            c2 = 2 * sqrt(c)
            ba = b / a2

            s = (a32 * s_abc + a2 * b * (s_abc - c2) + (4 * c * a - b ** 2) *
                    log((2 * a2 + ba + s_abc) / (ba + c2))) / (4 * a32)

        self.length_info['length'] = s
        self.length_info['bpoints'] = self.bpoints()
        return self.length_info['length']

    def bpoints(self):
        return self.start, self.control, self.end

    def transform(self, matrix):
        self.start = matrix @ self.start
        self.control = matrix @ self.control
        self.end = matrix @ self.end


class CubicBezier:
    def __init__(self, start, control1, control2, end):
        self.start = start
        self.control1 = control1
        self.control2 = control2
        self.end = end

        # used to know if self.length needs to be updated
        self.length_info = {'length': None, 'bpoints': None, 'error': None,
                             'min_depth': None}

    def __repr__(self):
        return 'CubicBezier(start={}, control1={}, control2={}, end={})'.format(self.start, self.control1,
                                                                                self.control2, self.end)

    def __getitem__(self, item):
        return self.bpoints()[item]

    def point(self, t):
        """Evaluate the cubic Bezier curve at t using Horner's rule."""
        # algebraically equivalent to
        # P0*(1-t)**3 + 3*P1*t*(1-t)**2 + 3*P2*(1-t)*t**2 + P3*t**3
        # for (P0, P1, P2, P3) = self.bpoints()
        return self.start + t * (
            3 * (self.control1 - self.start) + t * (
                3 * (self.start + self.control2) - 6 * self.control1 + t * (
                    -self.start + 3 * (self.control1 - self.control2) + self.end
                )))

    def length(self, error=LENGTH_ERROR, min_depth=LENGTH_MIN_DEPTH):
        """Calculate the length of the path up to a certain position"""
        if self.length_info['bpoints'] == self.bpoints() \
                and self.length_info['error'] >= error \
                and self.length_info['min_depth'] >= min_depth:
            return self.length_info['length']

        if USE_SCIPY_QUAD:
            s = quad(lambda tau: abs(self._derivative(tau)), 0, 1,
                     epsabs=error, limit=1000)[0]
        else:
            s = segment_length(self, 0, 1, self.point(0), self.point(1),
                               error, min_depth, 0)

        self.length_info['length'] = s
        self.length_info['bpoints'] = self.bpoints()
        self.length_info['error'] = error
        self.length_info['min_depth'] = min_depth
        return self.length_info['length']

    def bpoints(self):
        """returns the Bezier control points of the segment."""
        return self.start, self.control1, self.control2, self.end

    def _derivative(self, t, n=1):
        """returns the nth derivative of the segment at t.
        Note: Bezier curves can have points where their derivative vanishes.
        If you are interested in the tangent direction, use the unit_tangent()
        method instead."""
        p = self.bpoints()

        if n == 1:
            return 3 * (p[1] - p[0]) * (1 - t) ** 2 + 6 * (p[2] - p[1]) * (1 - t) * t + 3 * (
                p[3] - p[2]) * t ** 2
        elif n == 2:
            return 6 * (
                (1 - t) * (p[2] - 2 * p[1] + p[0]) + t * (p[3] - 2 * p[2] + p[1]))
        elif n == 3:
            return 6 * (p[3] - 3 * (p[2] - p[1]) - p[0])
        elif n > 3:
            return 0
        else:
            raise ValueError("n should be a positive integer.")

    def transform(self, matrix):
        self.start = matrix @ self.start
        self.control1 = matrix @ self.control1
        self.control2 = matrix @ self.control2
        self.end = matrix @ self.end


class Arc:
    def __init__(self, start, radius, rotation, large_arc, sweep, end):

        self.start = start
        self.radius = abs(radius.real) + 1j * abs(radius.imag)
        self.rotation = rotation
        self.large_arc = large_arc
        self.sweep = sweep
        self.end = end

        # Convenience parameters.
        self.phi = None
        self.rot_matrix = None

        self.length_info = {'length': None, 'bpoints': None, 'error': None,
                             'min_depth': None}

        # Derive derived parameters
        self._parameterize()

    def __repr__(self):
        params = (self.start, self.radius, self.rotation,
                  self.large_arc, self.sweep, self.end)
        return ("Arc(start={}, radius={}, rotation={}, "
                "large_arc={}, sweep={}, end={})".format(*params))

    def _parameterize(self):
        # start cannot be the same as end as the ellipse would
        # not be well defined
        assert self.start != self.end

        # Compute convenience parameters.
        self.phi = radians(self.rotation)
        self.rot_matrix = exp(1j * self.phi)

        cosr = cos(radians(self.rotation))
        sinr = sin(radians(self.rotation))
        dx = (self.start.real - self.end.real) / 2
        dy = (self.start.imag - self.end.imag) / 2
        x1prim = cosr * dx + sinr * dy
        x1prim_sq = x1prim * x1prim
        y1prim = -sinr * dx + cosr * dy
        y1prim_sq = y1prim * y1prim

        rx = self.radius.real
        rx_sq = rx * rx
        ry = self.radius.imag
        ry_sq = ry * ry

        # Correct out of range radii
        radius_check = (x1prim_sq / rx_sq) + (y1prim_sq / ry_sq)
        if radius_check > 1:
            rx *= sqrt(radius_check)
            ry *= sqrt(radius_check)
            self.radius = rx + 1j * ry
            rx_sq = rx * rx
            ry_sq = ry * ry

        t1 = rx_sq * y1prim_sq
        t2 = ry_sq * x1prim_sq
        c = sqrt(abs((rx_sq * ry_sq - t1 - t2) / (t1 + t2)))

        if self.large_arc == self.sweep:
            c = -c
        cxprim = c * rx * y1prim / ry
        cyprim = -c * ry * x1prim / rx

        self.center = complex((cosr * cxprim - sinr * cyprim) +
                              ((self.start.real + self.end.real) / 2),
                              (sinr * cxprim + cosr * cyprim) +
                              ((self.start.imag + self.end.imag) / 2))

        ux = (x1prim - cxprim) / rx
        uy = (y1prim - cyprim) / ry
        vx = (-x1prim - cxprim) / rx
        vy = (-y1prim - cyprim) / ry
        n = sqrt(ux * ux + uy * uy)
        p = ux
        theta = degrees(acos(p / n))
        if uy < 0:
            theta = -theta
        self.theta = theta % 360

        n = sqrt((ux * ux + uy * uy) * (vx * vx + vy * vy))
        p = ux * vx + uy * vy
        d = p/n
        # In certain cases the above calculation can through inaccuracies
        # become just slightly out of range, f ex -1.0000000000000002.
        if d > 1.0:
            d = 1.0
        elif d < -1.0:
            d = -1.0
        delta = degrees(acos(d))
        if (ux * vy - uy * vx) < 0:
            delta = -delta
        self.delta = delta % 360
        if not self.sweep:
            self.delta -= 360

    def point(self, t):
        angle = np.radians(self.theta + t*self.delta)
        cosphi = self.rot_matrix.real
        sinphi = self.rot_matrix.imag
        rx = self.radius.real
        ry = self.radius.imag

        # z = self.rot_matrix*(rx*cos(angle) + 1j*ry*sin(angle)) + self.center
        x = rx*cosphi*np.cos(angle) - ry*sinphi*np.sin(angle) + self.center.real
        y = rx*sinphi*np.cos(angle) + ry*cosphi*np.sin(angle) + self.center.imag
        return x + 1j * y

    def bpoints(self):
        return self.start, self.radius, self.rotation, self.large_arc, self.sweep, self.end, self.phi, self.rot_matrix

    def length(self, error=LENGTH_ERROR, min_depth=LENGTH_MIN_DEPTH):
        if self.length_info['bpoints'] == self.bpoints() \
                and self.length_info['error'] >= error \
                and self.length_info['min_depth'] >= min_depth:
            return self.length_info['length']

        if USE_SCIPY_QUAD:
            s = quad(lambda tau: abs(self._derivative(tau)), 0, 1,
                     epsabs=error, limit=1000)[0]
        else:
            s = segment_length(self, 0, 1, self.point(0), self.point(1),
                                  error, min_depth, 0)

        self.length_info['length'] = s
        self.length_info['bpoints'] = self.bpoints()
        self.length_info['error'] = error
        self.length_info['min_depth'] = min_depth
        return self.length_info['length']

    def _derivative(self, t, n=1):
        """returns the nth derivative of the segment at t."""
        angle = radians(self.theta + t * self.delta)
        phi = radians(self.rotation)
        rx = self.radius.real
        ry = self.radius.imag
        k = (self.delta * 2 * pi / 360) ** n  # ((d/dt)angle)**n

        if n % 4 == 0 and n > 0:
            return rx * cos(phi) * cos(angle) - ry * sin(phi) * sin(angle) + 1j * (
                rx * sin(phi) * cos(angle) + ry * cos(phi) * sin(angle))
        elif n % 4 == 1:
            return k * (-rx * cos(phi) * sin(angle) - ry * sin(phi) * cos(angle) + 1j * (
                -rx * sin(phi) * sin(angle) + ry * cos(phi) * cos(angle)))
        elif n % 4 == 2:
            return k * (-rx * cos(phi) * cos(angle) + ry * sin(phi) * sin(angle) + 1j * (
                -rx * sin(phi) * cos(angle) - ry * cos(phi) * sin(angle)))
        elif n % 4 == 3:
            return k * (rx * cos(phi) * sin(angle) + ry * sin(phi) * cos(angle) + 1j * (
                rx * sin(phi) * sin(angle) - ry * cos(phi) * cos(angle)))
        else:
            raise ValueError('n should be a positive integer.')

    @staticmethod
    def _near_zero(x):
        if abs(x) < 1e-15:
            return True
        else:
            return False

    @staticmethod
    def _sign(x):
        if x > 0:
            return 1
        elif x < 0:
            return -1
        else:
            return 0

    def transform(self, matrix):
        rx, ry = self.radius.real, self.radius.imag
        rotation = radians(self.rotation)

        s = sin(rotation)
        c = cos(rotation)

        m = (matrix[0] * +rx * c + matrix[2] * rx * s,
             matrix[1] * +rx * c + matrix[3] * rx * s,
             matrix[0] * -ry * s + matrix[2] * ry * c,
             matrix[1] * -ry * s + matrix[3] * ry * c)

        a = m[0] ** 2 + m[2] ** 2
        c = m[1] ** 2 + m[3] ** 2
        b = (m[0] * m[1] + m[2] * m[3]) * 2

        ac = a - c

        if self._near_zero(b):
            rotation = 0
            a2 = a
            c2 = c
        else:
            if self._near_zero(ac):
                a2 = a + b * 0.5
                c2 = a - b * 0.5
                rotation = pi / 4 * self._sign(self.rotation)
            else:
                k = 1 + b ** 2 / ac ** 2

                if k < 0:
                    k = 0
                else:
                    k = sqrt(k)

                a2 = 0.5 * (a + c + k * ac)
                c2 = 0.5 * (a + c - k * ac)
                rotation = 0.5 * atan2(b, ac)

        if a2 < 0:
            a2 = 0
        else:
            a2 = sqrt(a2)

        if c2 < 0:
            c2 = 0
        else:
            c2 = sqrt(c2)

        if ac <= 0:
            ry = a2
            rx = c2
        else:
            rx = a2
            ry = c2

        if matrix[0] * matrix[3] - matrix[1] * matrix[2] < 0:
            self.sweep = not self.sweep

        self.radius = complex(rx, ry)
        self.rotation = degrees(rotation)
        self.start = matrix @ self.start
        self.end = matrix @ self.end

        self._parameterize()
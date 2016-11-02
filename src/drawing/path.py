from math import sqrt, cos, sin, acos, degrees, radians, log, pi, atan2
from cmath import exp
from scipy.integrate import quad


LENGTH_MIN_DEPTH = 5
LENGTH_ERROR = 1e-12
USE_SCIPY_QUAD = True


def segment_length(curve, start, end, start_point, end_point,
                   error=LENGTH_ERROR, min_depth=LENGTH_MIN_DEPTH, depth=0):
    """Recursively approximates the length by straight lines"""
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
    # This is accurate enough.
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
    def __init__(self, start, radius, rotation, large_arc, sweep, end,
                 autoscale_radius=True):
        """
        This should be thought of as a part of an ellipse connecting two
        points on that ellipse, start and end.
        Parameters
        ----------
        start : complex
            The start point of the large_arc.
        radius : complex
            rx + 1j*ry, where rx and ry are the radii of the ellipse (also
            known as its semi-major and semi-minor axes, or vice-versa or if
            rx < ry).
            Note: If rx = 0 or ry = 0 then this arc is treated as a
            straight line segment joining the endpoints.
            Note: If rx or ry has a negative sign, the sign is dropped; the
            absolute value is used instead.
            Note:  If no such ellipse exists, the radius will be scaled so
            that one does (unless autoscale_radius is set to False).
        rotation : float
            This is the CCW angle (in degrees) from the positive x-axis of the
            current coordinate system to the x-axis of the ellipse.
        large_arc : bool
            This is the large_arc flag.  Given two points on an ellipse,
            there are two elliptical arcs connecting those points, the first
            going the short way around the ellipse, and the second going the
            long way around the ellipse.  If large_arc is 0, the shorter
            elliptical large_arc will be used.  If large_arc is 1, then longer
            elliptical will be used.
            In other words, it should be 0 for arcs spanning less than or
            equal to 180 degrees and 1 for arcs spanning greater than 180
            degrees.
        sweep : bool
            This is the sweep flag.  For any acceptable parameters start, end,
            rotation, and radius, there are two ellipses with the given major
            and minor axes (radii) which connect start and end.  One which
            connects them in a CCW fashion and one which connected them in a
            CW fashion.  If sweep is 1, the CCW ellipse will be used.  If
            sweep is 0, the CW ellipse will be used.

        end : complex
            The end point of the large_arc (must be distinct from start).

        Note on CW and CCW: The notions of CW and CCW are reversed in some
        sense when viewing SVGs (as the y coordinate starts at the top of the
        image and increases towards the bottom).

        Derived Parameters
        ------------------
        self._parameterize() sets self.center, self.theta and self.delta
        for use in self.point() and other methods.  If
        autoscale_radius == True, then this will also scale self.radius in the
        case that no ellipse exists with the given parameters (see usage
        below).

        self.theta : float
            This is the phase (in degrees) of self.u1transform(self.start).
            It is $\theta_1$ in the official documentation and ranges from
            -180 to 180.

        self.delta : float
            This is the angular distance (in degrees) between the start and
            end of the arc after the arc has been sent to the unit circle
            through self.u1transform().
            It is $\Delta\theta$ in the official documentation and ranges from
            -360 to 360; being positive when the arc travels CCW and negative
            otherwise (i.e. is positive/negative when sweep == True/False).

        self.center : complex
            This is the center of the arc's ellipse.
        """

        self.start = start
        self.radius = abs(radius.real) + 1j * abs(radius.imag)
        self.rotation = rotation
        self.large_arc = large_arc
        self.sweep = sweep
        self.end = end
        self.autoscale_radius = autoscale_radius

        # Convenience parameters
        self.phi = radians(self.rotation)
        self.rot_matrix = exp(1j * self.phi)

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

        # See http://www.w3.org/TR/SVG/implnote.html#ArcImplementationNotes
        # my notation roughly follows theirs
        rx = self.radius.real
        ry = self.radius.imag
        rx_sqd = rx * rx
        ry_sqd = ry * ry

        # Transform z-> z' = x' + 1j*y'
        # = self.rot_matrix**(-1)*(z - (end+start)/2)
        # coordinates.  This translates the ellipse so that the midpoint
        # between self.end and self.start lies on the origin and rotates
        # the ellipse so that the its axes align with the xy-coordinate axes.
        # Note:  This sends self.end to -self.start
        zp1 = (1 / self.rot_matrix) * (self.start - self.end) / 2
        x1p, y1p = zp1.real, zp1.imag
        x1p_sqd = x1p * x1p
        y1p_sqd = y1p * y1p

        # Correct out of range radii
        # Note: an ellipse going through start and end with radius and phi
        # exists if and only if radius_check is true
        radius_check = (x1p_sqd / rx_sqd) + (y1p_sqd / ry_sqd)
        if radius_check > 1:
            if self.autoscale_radius:
                rx *= sqrt(radius_check)
                ry *= sqrt(radius_check)
                self.radius = rx + 1j * ry
                rx_sqd = rx * rx
                ry_sqd = ry * ry
            else:
                raise ValueError('No such elliptic arc exists.')

        # Compute c'=(c_x', c_y'), the center of the ellipse in (x', y') coords
        # Noting that, in our new coord system, (x_2', y_2') = (-x_1', -x_2')
        # and our ellipse is cut out by of the plane by the algebraic equation
        # (x'-c_x')**2 / r_x**2 + (y'-c_y')**2 / r_y**2 = 1,
        # we can find c' by solving the system of two quadratics given by
        # plugging our transformed endpoints (x_1', y_1') and (x_2', y_2')
        tmp = rx_sqd * y1p_sqd + ry_sqd * x1p_sqd
        radicand = (rx_sqd * ry_sqd - tmp) / tmp
        try:
            radical = sqrt(radicand)
        except ValueError:
            radical = 0
        if self.large_arc == self.sweep:
            cp = -radical * (rx * y1p / ry - 1j * ry * x1p / rx)
        else:
            cp = radical * (rx * y1p / ry - 1j * ry * x1p / rx)

        # The center in (x,y) coordinates is easy to find knowing c'
        self.center = exp(1j * self.phi) * cp + (self.start + self.end) / 2

        # Now we do a second transformation, from (x', y') to (u_x, u_y)
        # coordinates, which is a translation moving the center of the
        # ellipse to the origin and a dilation stretching the ellipse to be
        # the unit circle
        u1 = (x1p - cp.real) / rx + 1j * (y1p - cp.imag) / ry  # transformed start
        u2 = (-x1p - cp.real) / rx + 1j * (-y1p - cp.imag) / ry  # transformed end

        # Now compute theta and delta (we'll define them as we go)
        # delta is the angular distance of the arc (w.r.t the circle)
        # theta is the angle between the positive x'-axis and the start point
        # on the circle
        if u1.imag > 0:
            self.theta = degrees(acos(u1.real))
        elif u1.imag < 0:
            self.theta = -degrees(acos(u1.real))
        else:
            if u1.real > 0:  # start is on pos u_x axis
                self.theta = 0
            else:  # start is on neg u_x axis
                # Note: This behavior disagrees with behavior documented in
                # http://www.w3.org/TR/SVG/implnote.html#ArcImplementationNotes
                # where theta is set to 0 in this case.
                self.theta = 180

        det_uv = u1.real * u2.imag - u1.imag * u2.real

        acosand = u1.real * u2.real + u1.imag * u2.imag
        if acosand > 1 or acosand < -1:
            acosand = round(acosand)
        if det_uv > 0:
            self.delta = degrees(acos(acosand))
        elif det_uv < 0:
            self.delta = -degrees(acos(acosand))
        else:
            if u1.real * u2.real + u1.imag * u2.imag > 0:
                # u1 == u2
                self.delta = 0
            else:
                # u1 == -u2
                # Note: This behavior disagrees with behavior documented in
                # http://www.w3.org/TR/SVG/implnote.html#ArcImplementationNotes
                # where delta is set to 0 in this case.
                self.delta = 180

        if not self.sweep and self.delta >= 0:
            self.delta -= 360
        elif self.large_arc and self.delta <= 0:
            self.delta += 360

    def point(self, t):
        if t == 0:
            return self.start
        if t == 1:
            return self.end
        angle = radians(self.theta + t * self.delta)
        cosphi = self.rot_matrix.real
        sinphi = self.rot_matrix.imag
        rx = self.radius.real
        ry = self.radius.imag

        # z = self.rot_matrix*(rx*cos(angle) + 1j*ry*sin(angle)) + self.center
        x = rx * cosphi * cos(angle) - ry * sinphi * sin(angle) + self.center.real
        y = rx * sinphi * cos(angle) + ry * cosphi * sin(angle) + self.center.imag
        return complex(x, y)

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

    def transform(self, matrix):
            rh, rv = self.radius.real, self.radius.imag
            rotation = radians(self.rotation)

            s = sin(rotation)
            c = cos(rotation)

            m = (matrix[0] * rh * +c + matrix[2] * rv * s,
                 matrix[1] * rh * +c + matrix[3] * rv * s,
                 matrix[0] * rv * -c + matrix[2] * rh * c,
                 matrix[1] * rv * -c + matrix[3] * rh * c,)

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
                    rotation = pi / 4
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
                rv = a2
                rh = c2
            else:
                rh = a2
                rv = c2

            if matrix[0] * matrix[3] - matrix[1] * matrix[2] < 0:
                self.sweep = not self.sweep

            self.radius = (rh, rv)
            self.rotation = degrees(rotation)
            self.start = matrix @ self.start
            self.end = matrix @ self.end

            self._parameterize()



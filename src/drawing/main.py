import numpy as np
import re
import math
import copy
import inspect
import sys
import os
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button
from .path import *
import xml.etree.ElementTree as etree


logger = logging.getLogger('universe')

COMMAND_RE = re.compile('([MmZzLlHhVvCcSsQqTtAa])')
FLOAT_RE = re.compile('[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?')
UNIT_RE = re.compile('em|ex|px|in|cm|mm|pt|pc|%')
COORD_RE = re.compile('([+-]?\d+\.?\d+)\s*,\s*([+-]?\d+\.?\d+)')

svg_transforms = ['matrix', 'translate', 'scale', 'rotate', 'skewX', 'skewY']
TRANSFORMS_RE = re.compile('|'.join([x + '[^)]*\)' for x in svg_transforms]))

svg_ns = '{http://www.w3.org/2000/svg}'

unit_convert = {
    None: 1,           # Default unit (same as pixel)
    'px': 1,           # px: pixel. Default SVG unit
    'em': 10,          # 1 em = 10 px FIXME
    'ex': 5,           # 1 ex =  5 px FIXME
    'in': 96,          # 1 in = 96 px
    'cm': 96 / 2.54,   # 1 cm = 1/2.54 in
    'mm': 96 / 25.4,   # 1 mm = 1/25.4 in
    'pt': 96 / 72.0,   # 1 pt = 1/72 in
    'pc': 96 / 6.0,    # 1 pc = 1/6 in
    '%' :  1 / 100.0   # 1 percent
}


class SVGMatrix:
    def __init__(self, mat=(1, 0, 0, 1, 0, 0)):
        if len(mat) != 6:
            raise ValueError('Bad matrix size {}.'.format(len(mat)))

        self.mat = mat

    def __matmul__(self, other):
        if isinstance(other, SVGMatrix):
            a = self.mat[0] * other.mat[0] + self.mat[2] * other.mat[1]
            b = self.mat[1] * other.mat[0] + self.mat[3] * other.mat[1]
            c = self.mat[0] * other.mat[2] + self.mat[2] * other.mat[3]
            d = self.mat[1] * other.mat[2] + self.mat[3] * other.mat[3]
            e = self.mat[0] * other.mat[4] + self.mat[2] * other.mat[5] + self.mat[4]
            f = self.mat[1] * other.mat[4] + self.mat[3] * other.mat[5] + self.mat[5]
            return SVGMatrix((a, b, c, d, e, f))

        elif isinstance(other, complex):
            x = other.real * self.mat[0] + other.imag * self.mat[2] + self.mat[4]
            y = other.real * self.mat[1] + other.imag * self.mat[3] + self.mat[5]
            return complex(x, y)

        else:
            return NotImplemented

    def __str__(self):
        return str(self.mat)

    def __getitem__(self, item):
        return self.mat[item]

    def apply_matrix(self, mat):
        return self @ SVGMatrix(mat)

    def translate(self, tx, ty=None):
        if ty is None:
            ty = 0

        return self @ SVGMatrix((1, 0, 0, 1, tx, ty))

    def scale(self, sx, sy=None):
        if sy is None:
            sy = sx

        return self @ SVGMatrix((sx, 0, 0, sy, 0, 0))

    def rotate(self, a, tx=None, ty=None):
        cosa = math.cos(math.radians(a))
        sina = math.sin(math.radians(a))

        if ty is None:
            return self @ SVGMatrix((cosa, sina, -sina, cosa, 0, 0))
        else:
            return self @ \
                   SVGMatrix((1, 0, 0, 1, tx, ty)) @ \
                   SVGMatrix((cosa, sina, -sina, cosa, 0, 0)) @ \
                   SVGMatrix((1, 0, 0, 1, -tx, -ty))

    def skew_x(self, angle):
        tana = math.tan(math.radians(angle))
        return self @ SVGMatrix((1, 0, tana, 1, 0, 0))

    def skew_y(self, angle):
        tana = math.tan(math.radians(angle))
        return self @ SVGMatrix((1, tana, 0, 1, 0, 0))

    def flip_x(self):
        return self @ SVGMatrix((-1, 0, 0, 1, 0, 0))

    def flip_y(self):
        return self @ SVGMatrix((1, 0, 0, -1, 0, 0))


class Transformable:
    def __init__(self, elt=None):
        self.items = []
        self.matrix = SVGMatrix()
        self.viewport = (800, 600)

        # ID.
        self.id = hex(id(self))

        if elt is not None:
            self._get_transformations(elt)

    def _get_transformations(self, elt):
        t = elt.get('transform')

        if t is None:
            return

        transforms = TRANSFORMS_RE.findall(t)

        for t in transforms:
            op, args = t.split('(')
            op = op.strip()
            
            # Keep only numbers.
            args = [float(x) for x in FLOAT_RE.findall(args)]

            if op == 'matrix':
                self.matrix = self.matrix.apply_matrix(args)

            if op == 'translate':
                self.matrix = self.matrix.translate(*args)

            if op == 'scale':
                self.matrix = self.matrix.scale(*args)

            if op == 'rotate':
                self.matrix = self.matrix.rotate(*args)

            if op == 'skewX':
                self.matrix = self.matrix.skew_x(*args)

            if op == 'skewY':
                self.matrix = self.matrix.skew_y(*args)

    def length(self, v, mode=None):
        # Handle empty (non-existing) length element.
        if v is None:
            return 0

        # Get length value.
        m = FLOAT_RE.search(v)
        if m:
            value = float(m.group(0))
        else:
            raise TypeError(v + ' is not a valid length.')

        # Get length unit.
        m = UNIT_RE.search(v)
        if m:
            unit = m.group(0)
        else:
            unit = None

        if unit == '%':
            if mode == 'x':
                return value * unit_convert[unit] * self.viewport[0]
            if mode == 'y':
                return value * unit_convert[unit] * self.viewport[1]

        return value * unit_convert[unit]

    def lengths(self, x, y):
        return self.length(x, 'x'), self.length(y, 'y')

    def transform(self, matrix=None):
        for x in self.items:
            x.transform(matrix)

    def flatten(self):
        i = 0
        flat = copy.deepcopy(self.items)

        while i < len(flat):
            while isinstance(flat[i], Group):
                flat[i:i+1] = flat[i].items
            i += 1

        return flat

    def __len__(self):
        return len(self.items)

    def __getitem__(self, item):
        return self.items[item]


class SVG(Transformable):
    def __init__(self, filename=None, viewport=None):
        super().__init__()

        self.filename = None
        self.root = None

        if viewport is not None:
            self.viewport = viewport

        if filename:
            self.filename = filename
            tree = etree.parse(filename)
            self.parse(tree)

    def parse(self, tree):
        self.root = tree.getroot()

        if self.root.tag != svg_ns + 'svg':
            raise TypeError('Data does not seem to be a valid SVG.')

        # Create a base Group to group all other items (useful for viewBox elt).
        base_group = BaseGroup()
        base_group.viewport = self.viewport
        self.items.append(base_group)

        # SVG dimensions.
        width, height = self.lengths(self.root.get('width'), self.root.get('height'))

        # If width and height are not defined, scale to viewport.
        if width != 0 and height != 0:
            base_group.viewport = (width, height)
        else:
            width, height = base_group.viewport[0], base_group.viewport[1]

        # Handle scaling to viewport.
        preserve_ar = self.root.get('preserveAspectRatio', 'xMidYMid').split()

        if len(preserve_ar) != 1:
            mos = preserve_ar[1]
        else:
            mos = 'meet'

        align = preserve_ar[0].lower()
        x_align, y_align = align[:4], align[4:]

        # Scale if necessary.
        if self.root.get('viewBox') is not None:
            view_box = FLOAT_RE.findall(self.root.get('viewBox'))
            mx, my, w, h = [float(x) for x in view_box]

            sx = width / w
            sy = height / h

            if mos == 'slice':
                s = max(sx, sy)
            else:
                s = min(sx, sy)

            tx = -mx * s
            ty = -my * s

            if x_align == 'xmid':
                tx += (width - w * s) / 2
            elif x_align == 'xmax':
                tx += width - w * s

            if y_align == 'ymid':
                ty += (height - h * s) / 2
            elif y_align == 'ymax':
                ty += (height - h * s)

            base_group.matrix = SVGMatrix((s, 0, 0, s, tx, ty))

        # Parse XML elements hierarchically with groups.
        base_group.append(self.root)

        self.transform()

    def title(self):
        t = self.root.find(svg_ns + 'title')

        if t is not None:
            return t
        else:
            return os.path.splitext(os.path.basename(self.filename))[0]


class Group(Transformable):
    tag = 'g'

    def __init__(self, elt=None):
        super().__init__(elt)

    def append(self, element):
        for elt in element:
            elt_class = svg_classes.get(elt.tag, None)

            if elt_class is None:
                logger.debug('No immediate handler for element {}.'.format(elt.tag))

                if len(elt) > 0:
                    elt_class = HeuristicGroup
                else:
                    continue

            # Instantiate elt associated class.
            item = elt_class(elt)

            # Apply group matrix to the newly created object.
            item.matrix = self.matrix @ item.matrix
            item.viewport = self.viewport

            # Recursively append if elt is a group or an unknown element.
            if elt.tag == svg_ns + 'g' or elt_class is HeuristicGroup:
                item.append(elt)

            # Ensure that group has valid elements.
            if len(item) > 0:
                self.items.append(item)

    def __repr__(self):
        return '<Group ' + self.id + '>: ' + repr(self.items)


class HeuristicGroup(Group):
    def __init__(self, elt):
        super().__init__(elt)

    def __repr__(self):
        return '<HeuristicGroup ' + self.id + '>: ' + repr(self.items)


class BaseGroup(Group):
    def __init__(self):
        super().__init__()

    def __repr__(self):
        return '<BaseGroup ' + self.id + '>: ' + repr(self.items)


class SVGPath(Transformable):
    tag = 'path'

    COMMANDS = set('MmZzLlHhVvCcSsQqTtAa')
    UPPERCASE = set('MZLHVCSQTA')

    def __init__(self, elt=None):
        super().__init__(elt)

        if elt is not None:
            self.style = elt.get('style')
            self._parse(elt.get('d'))

    @staticmethod
    def _stringify(x):
        return ' '.join(str(a) for a in x)

    def _tokenize_path(self, pathdef):
        for x in COMMAND_RE.split(pathdef):
            if x in self.COMMANDS:
                yield x
            for token in FLOAT_RE.findall(x):
                yield token

    def _parse(self, pathdef):
        current_pos = 0j

        elements = list(self._tokenize_path(pathdef))
        elements.reverse()

        start_pos = None
        command = None

        while elements:
            if elements[-1] in self.COMMANDS:
                # New command.
                last_command = command  # Used by S and T
                command = elements.pop()
                absolute = command in self.UPPERCASE
                command = command.upper()
            else:
                # If this element starts with numbers, it is an implicit command
                # and we don't change the command. Check that it's allowed:
                if command is None:
                    raise ValueError('Disallowed implicit command in %s, position %s.' %
                                     (pathdef, len(pathdef.split()) - len(elements)))

            if command == 'M':
                # Moveto command.
                x = elements.pop()
                y = elements.pop()
                pos = float(x) + float(y) * 1j
                if absolute:
                    current_pos = pos
                else:
                    current_pos += pos

                # when M is called, reset start_pos
                # This behavior of Z is defined in svg spec:
                # http://www.w3.org/TR/SVG/paths.html#PathDataClosePathCommand
                start_pos = current_pos

                # Implicit moveto commands are treated as lineto commands.
                # So we set command to lineto here, in case there are
                # further implicit commands after this moveto.
                command = 'L'

            elif command == 'Z':
                # Close path
                if not (current_pos == start_pos):
                    self.items.append(Line(current_pos, start_pos))
                current_pos = start_pos
                start_pos = None
                command = None  # You can't have implicit commands after closing.

            elif command == 'L':
                x = elements.pop()
                y = elements.pop()
                pos = float(x) + float(y) * 1j
                if not absolute:
                    pos += current_pos
                self.items.append(Line(current_pos, pos))
                current_pos = pos

            elif command == 'H':
                x = elements.pop()
                pos = float(x) + current_pos.imag * 1j
                if not absolute:
                    pos += current_pos.real
                self.items.append(Line(current_pos, pos))
                current_pos = pos

            elif command == 'V':
                y = elements.pop()
                pos = current_pos.real + float(y) * 1j
                if not absolute:
                    pos += current_pos.imag * 1j
                self.items.append(Line(current_pos, pos))
                current_pos = pos

            elif command == 'C':
                control1 = float(elements.pop()) + float(elements.pop()) * 1j
                control2 = float(elements.pop()) + float(elements.pop()) * 1j
                end = float(elements.pop()) + float(elements.pop()) * 1j

                if not absolute:
                    control1 += current_pos
                    control2 += current_pos
                    end += current_pos

                self.items.append(CubicBezier(current_pos, control1, control2, end))
                current_pos = end

            elif command == 'S':
                # Smooth curve. First control point is the "reflection" of
                # the second control point in the previous path.

                if last_command not in 'CS':
                    # If there is no previous command or if the previous command
                    # was not an C, c, S or s, assume the first control point is
                    # coincident with the current point.
                    control1 = current_pos
                else:
                    # The first control point is assumed to be the reflection of
                    # the second control point on the previous command relative
                    # to the current point.
                    control1 = current_pos + current_pos - self.items[-1].control2

                control2 = float(elements.pop()) + float(elements.pop()) * 1j
                end = float(elements.pop()) + float(elements.pop()) * 1j

                if not absolute:
                    control2 += current_pos
                    end += current_pos

                self.items.append(CubicBezier(current_pos, control1, control2, end))
                current_pos = end

            elif command == 'Q':
                control = float(elements.pop()) + float(elements.pop()) * 1j
                end = float(elements.pop()) + float(elements.pop()) * 1j

                if not absolute:
                    control += current_pos
                    end += current_pos

                self.items.append(QuadraticBezier(current_pos, control, end))
                current_pos = end

            elif command == 'T':
                # Smooth curve. Control point is the "reflection" of
                # the second control point in the previous path.

                if last_command not in 'QT':
                    # If there is no previous command or if the previous command
                    # was not an Q, q, T or t, assume the first control point is
                    # coincident with the current point.
                    control = current_pos
                else:
                    # The control point is assumed to be the reflection of
                    # the control point on the previous command relative
                    # to the current point.
                    control = current_pos + current_pos - self.items[-1].control

                end = float(elements.pop()) + float(elements.pop()) * 1j

                if not absolute:
                    end += current_pos

                self.items.append(QuadraticBezier(current_pos, control, end))
                current_pos = end

            elif command == 'A':
                radius = float(elements.pop()) + float(elements.pop()) * 1j
                rotation = float(elements.pop())
                arc = bool(float(elements.pop()))
                sweep = bool(float(elements.pop()))
                end = float(elements.pop()) + float(elements.pop()) * 1j

                if not absolute:
                    end += current_pos

                self.items.append(Arc(current_pos, radius, rotation, arc, sweep, end))
                current_pos = end

    def length_info(self, error=LENGTH_ERROR, min_depth=LENGTH_MIN_DEPTH):
        lengths = [each.length(error=error, min_depth=min_depth) for each in
                   self.items]
        total_length = sum(lengths)

        return total_length, lengths

    def transform(self, matrix=None):
        if matrix is None:
            matrix = self.matrix

        for item in self.items:
            item.transform(matrix)

    def __str__(self):
        return '\n'.join(str(x) for x in self.items)

    def __repr__(self):
        return '<Path ' + self.id + '>'


class SVGPolyline(SVGPath):
    tag = 'polyline'

    def __init__(self, elt=None):
        if elt is not None:
            d = self._convert(elt)
            elt.set('d', d)

        super().__init__(elt)

    def _convert(self, elt):
        points = COORD_RE.findall(elt.get('points'))

        if points[0] == points[-1]:
            closed = True
        else:
            closed = False

        d = ['M', *points.pop(0)]

        for p in points:
            d.extend(('L', *p))

        if closed:
            d.append('z')

        return self._stringify(d)

    def __repr__(self):
        return '<Polyline ' + self.id + '>'


class SVGPolygon(SVGPath):
    tag = 'polygon'

    def __init__(self, elt=None):
        if elt is not None:
            d = self._convert(elt)
            elt.set('d', d)

        super().__init__(elt)

    def _convert(self, elt):
        points = COORD_RE.findall(elt.get('points'))

        d = ['M', *points.pop(0)]

        for p in points:
            d.extend(('L', *p))

        d.append('z')

        return self._stringify(d)

    def __repr__(self):
        return '<Polygon ' + self.id + '>'


class SVGLine(SVGPath):
    tag = 'line'

    def __init__(self, elt=None):
        if elt is not None:
            d = self._convert(elt)
            elt.set('d', d)

        super().__init__(elt)

    def _convert(self, elt):
        x1, x2 = elt.get('x1'), elt.get('x2')
        y1, y2 = elt.get('y1'), elt.get('y2')

        d = ['M', x1, y1, 'L', x2, y2]

        return self._stringify(d)

    def __repr__(self):
        return '<Line ' + self.id + '>'


class SVGEllipse(SVGPath):
    tag = 'ellipse'

    def __init__(self, elt=None):
        if elt is not None:
            d = self._convert(elt)
            elt.set('d', d)

        super().__init__(elt)

    def _convert(self, elt):
        rx, ry = elt.get('rx'), elt.get('ry')
        cx, cy = elt.get('cx'), elt.get('cy')

        if not ry:
            ry = rx

        rx, ry, cx, cy = float(rx), float(ry), float(cx), float(cy)

        d = ['M', cx - rx, cy,
             'a', rx, ry, 0, 1, 0, +(rx * 2), 0,
             'a', rx, ry, 0, 1, 0, -(rx * 2), 0]

        return self._stringify(d)

    def __repr__(self):
        return '<Ellipse ' + self.id + '>'


class SVGCircle(SVGPath):
    tag = 'circle'

    def __init__(self, elt=None):
        if elt is not None:
            d = self._convert(elt)
            elt.set('d', d)

        super().__init__(elt)

    def _convert(self, elt):
        cx, cy = elt.get('cx'), elt.get('cy')
        r = elt.get('r')

        cx, cy, r = float(cx), float(cy), float(r)

        d = ['M', cx - r, cy,
             'a', r, r, 0, 1, 0, +(r * 2), 0,
             'a', r, r, 0, 1, 0, -(r * 2), 0]

        return self._stringify(d)

    def __repr__(self):
        return '<Circle ' + self.id + '>'


class SVGRectangle(SVGPath):
    tag = 'rect'

    def __init__(self, elt=None):
        if elt is not None:
            d = self._convert(elt)
            elt.set('d', d)

        super().__init__(elt)

    @staticmethod
    def _valid(x):
        try:
            if abs(float(x)) >= 0:
                return True
            else:
                return False
        except (TypeError, ValueError):
            return False

    def _convert(self, elt):
        rx, ry = elt.get('rx'), elt.get('ry')
        x, y = float(elt.get('x')), float(elt.get('y'))
        w, h = float(elt.get('width')), float(elt.get('height'))

        a, b = self._valid(rx), self._valid(ry)
        if not a and not b:
            rx = ry = 0
        elif a and not b:
            rx = float(rx)
            ry = rx
        elif not a and b:
            ry = float(ry)
            rx = ry
        else:
            rx = float(rx)
            ry = float(ry)

            if rx > w / 2:
                rx = w / 2
            if ry > h / 2:
                ry = h / 2

        if rx == 0 and ry == 0:
            d = ['M', x, y,
                 'L', x + w, y,
                 'L', x + w, y + h,
                 'L', x, y + h,
                 'L', x, y,
                 'Z']
        else:
            d = ['M', x + rx, y,
                 'H', x + w - rx,
                 'A', rx, ry, 0, 0, 1, x + w, y + ry,
                 'V', y + h - ry,
                 'A', rx, ry, 0, 0, 1, x + w - rx, y + h,
                 'H', x + rx,
                 'A', rx, ry, 0, 0, 1, x, y + h - ry,
                 'V', y + ry,
                 'A', rx, ry, 0, 0, 1, x + rx, y]

        return self._stringify(d)

    def __repr__(self):
        return '<Rectangle ' + self.id + '>'


class Drawing:
    def __init__(self, file, viewport, **kwargs):
        self.file = file
        self.viewport = viewport

        self.dx = kwargs.get('dx', 5)
        self.min_points = kwargs.get('min_points', 5)

        resize = kwargs.get('resize', False)
        center = kwargs.get('center', True)
        transform = kwargs.get('transform', None)

        assert self.min_points >= 2

        logger.info('Loading SVG file "{}."'.format(self.file))
        tree = etree.parse(file)
        root = tree.getroot()

        if root.get('viewBox') is None:
            logger.info('No viewBox. Trying to interpolate.')

            width = root.get('width')
            height = root.get('height')

            if width and height:
                root.set('viewBox', '0 0 {} {}'.format(width, height))
            else:
                logger.warning('Unable to interpolate viewBox. Skipping scale and resize.')
                resize = False
                center = False

        if resize:
            root.set('width', str(self.viewport[0]))
            root.set('height', str(self.viewport[1]))

        if center:
            root.set('preserveAspectRatio', 'xMidYMid')

        logger.info('Parsing modified SVG file.'.format(self.file))
        self.svg = SVG(viewport=self.viewport)
        self.svg.parse(tree)

        logger.info('Applying transformations.')

        matrix = SVGMatrix()
        matrix = matrix.translate(0, self.viewport[1])
        matrix = matrix.flip_y()

        width, height = self.svg.lengths(root.get('width'), root.get('height'))

        if center and width != 0 and height != 0:
            tx = (self.viewport[0] - width) / 2
            ty = (self.viewport[1] - height) / 2
            matrix = matrix.translate(tx, ty)

        if transform:
            matrix = matrix @ transform

        self.svg.transform(matrix)

        logger.info('Flattening groups.')
        self.paths = self.svg.flatten()

        logger.info('Generating path points.')
        self.points = self._get_points()

        logger.info('Completed processing SVG.')

    def _get_points(self):
        points = []
        nan = np.nan + 1j * np.nan

        for path in self.paths:
            lengths = path.length_info()[1]
            last_point = path[0].start

            for segment, length in zip(path, lengths):
                if last_point != segment.start:
                    points.append(nan)

                n = length / self.dx

                if n < self.min_points:
                    n = self.min_points

                ts = np.linspace(0, 1, n, endpoint=False)
                points.extend(segment.point(t) for t in ts)

                last_point = segment.end

            points.append(last_point)
            points.append(nan)

        # Perform out of bound filter.
        points = np.array(points)

        with np.errstate(invalid='ignore'):
            points[(points.real > self.viewport[0]) | (points.real < 0) |
                   (points.imag > self.viewport[1]) | (points.imag < 0)] = nan

        return points

    def total_length(self):
        return sum(x.length_info()[0] for x in self.paths)

    def bounding_box(self):
        x = np.real(self.points)
        y = np.imag(self.points)

        x_min = x.min()
        y_min = y.min()
        x_max = x.max()
        y_max = y.max()

        return x_min, y_min, x_max, y_max

    def preview(self):
        fig, ax = plt.subplots()
        plt.axis('equal')

        ax.plot(np.real(self.points), np.imag(self.points), zorder=2, color='black')

        w, h = self.viewport
        ax.add_patch(patches.Rectangle((0, 0), w, h,
                                       facecolor='white', zorder=1, alpha=1, edgecolor='none'))

        border = 20
        ax.set_xlim(-border, w + border)
        ax.set_ylim(-border, h + border)

        plt.subplots_adjust(bottom=0.2)
        ax.set_axis_bgcolor('#FADBD8')

        reject = plt.axes([0.69, 0.05, 0.1, 0.075])
        accept = plt.axes([0.80, 0.05, 0.1, 0.075])

        b_reject = Button(reject, 'Reject')
        b_accept = Button(accept, 'Accept')

        class Callback:
            def __init__(self):
                self.status = False

            def reject(self, event):
                plt.close()

            def accept(self, event):
                plt.close()
                self.status = True

        callback = Callback()
        b_reject.on_clicked(callback.reject)
        b_accept.on_clicked(callback.accept)

        plt.show()

        return callback.status


svg_classes = {}

for name, cls in inspect.getmembers(sys.modules[__name__], inspect.isclass):
    tag = getattr(cls, 'tag', None)
    if tag:
        svg_classes[svg_ns + tag] = cls


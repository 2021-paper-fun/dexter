from .maestro import Maestro
from .usc import Usc
from .usc import enum
import numpy as np
import math
import time
from queue import Queue
from threading import Thread, Event
import logging


logger = logging.getLogger('universe')

uscSerialMode = enum.uscSerialMode
ChannelMode = enum.ChannelMode
HomeMode = enum.HomeMode


class ServoError(Exception):
    pass


class Servo:
    tau = 2 * math.pi

    def __init__(self, channel, min_ang, max_ang, min_pwm, max_pwm, max_vel, bias=0, direction=1):
        """
        Creates a servo object.
        :param channel: The channel number beginning with 0.
        :param min_ang: The minimum angle for the servo. [-360, 360] (degrees).
        :param max_ang: The maximum angle for the servo. [-360, 360] (degrees).
        :param min_pwm: The minimum PWM. [0, 4000] (us).
        :param max_pwm: The maximum PWM. [0, 4000] (us).
        :param max_vel: The maximum velocity of the servo. (0, 1000] (ms / 60 deg).
        :param bias: The bias for the servo. [-360, 360] (degrees).
        :param direction: The direction, either -1 or 1.

        Setting Bias
        ============
        Bias should be adjusted such that the servo is at kinematic "0" degree when it's target is 0 degrees.
        This is used to compensate for ridge spacing and inaccuracies during installation.
        Think of this like the "home" value of the servo.

        Setting Direction
        =================
        If the servo points in a positive axis and is grounded, direction is 1.
        If the servo points in a positive axis and is not grounded, direction is -1.
        If the servo points in a negative axis and is grounded, direction is -1.
        If the servo points in a negative axis and is not grounded, direction is 1.
        """

        self.channel = channel
        self.min_rad = math.radians(min_ang)
        self.max_rad = math.radians(max_ang)
        self.min_qus = min_pwm * 4
        self.max_qus = max_pwm * 4
        self.max_vel = max_vel

        self.bias = math.radians(bias)
        self.direction = direction

        # Dynamic current data.
        self.pwm = 0
        self.vel = 0
        self.accel = 0

        # User defined target. Also used to store last target.
        # In units of 0.25 us.
        self.target = 0

        # Compute constants.
        self.k_ang2qus = (self.max_qus - self.min_qus) / (self.max_rad - self.min_rad)
        self.k_qus2ang = (self.max_rad - self.min_rad) / (self.max_qus - self.min_qus)
        self.k_vel2mae = (60 * self.k_ang2qus) / self.max_vel * 10
        self.k_mae2vel = self.max_vel / ((60 * self.k_ang2qus) * 10)

    def zero(self):
        """
        Set the servo to zero, ignoring bias.
        """

        self.target = self.rad_to_qus(0)

    def set_target(self, rad):
        """
        Set the target for the servo.
        :param rad: The input radians.
        """

        rad = self.normalize(rad)
        self.target = self.rad_to_qus(rad)

    def normalize(self, rad):
        """
        Normalize a radian for the servo, taking into account direction and bias.
        :param rad: Input radians.
        :return: Output radians.
        """

        # Account for direction and bias.
        rad = rad * self.direction + self.bias

        # Normalize.
        if rad > self.max_rad:
            rad -= self.tau
        elif rad < self.min_rad:
            rad += self.tau

        if rad > self.max_rad or rad < self.min_rad:
            raise ServoError('Target out of range!')

        return rad

    def reachable(self, rad):
        """
        Determines if an angle is reachable.
        :param rad: The input angle.
        :return: True if reachable, False otherwise.
        """

        try:
            self.normalize(rad)
            return True
        except ServoError:
            return False

    def get_position(self):
        """
        Get the servo's current position in radians.
        :return: Output radians.
        """

        rad = self.qus_to_rad(self.pwm)
        rad = (rad - self.bias) * self.direction
        return rad

    def at_target(self):
        """
        Checks if the servo is at its target.
        :return: True if servo is at its target, else False.
        """

        return self.target == self.pwm

    def rad_to_qus(self, rad):
        """
        Converts radians to 0.25 us.
        :param rad: The input radians.
        :return: The PWM in units of 0.25 us.
        """

        return round(self.min_qus + self.k_ang2qus * (rad - self.min_rad))

    def qus_to_rad(self, pwm):
        """
        Converts 0.25 us to radians.
        :param pwm: The input PWM in units of 0.25 us.
        :return: Radians.
        """

        return self.min_rad + self.k_qus2ang * (pwm - self.min_qus)


class Arm:
    def __init__(self, servo1, servo2, servo3, servo4,
                 lengths, fk_solver, ik_solver):

        self.servos = [servo1, servo2, servo3, servo4]
        self.lengths = lengths
        self.zero = (0, lengths[2], lengths[0] + lengths[1] + lengths[3])
        self.length = np.linalg.norm(self.zero)

        self.fk_solver = fk_solver
        self.ik_solver = ik_solver

    def __getitem__(self, key):
        return self.servos[key]

    def __len__(self):
        return len(self.servos)

    def get_angles(self, point, solution=0):
        try:
            # Interpolate phi from [3*pi/4, 5*pi/4]
            r = np.linalg.norm(point) / self.length
            if r > 1:
                raise ValueError

            phi = 5 * math.pi / 4 - math.pi / 2 * r

            angles = self.ik_solver(self.lengths, (*point, phi))[solution]

            for servo, angle in zip(self.servos, angles):
                if not servo.reachable(angle):
                    logger.warning('Servo error at ({:.2f}, {:.2f}, {:.2f}) '
                                   'with angles ({:.2f}, {:.2f}, {:.2f}, {:.2f}).'.format(*point, *angles))
                    return None

            return angles
        except (ValueError, ZeroDivisionError):
            logger.warning('IK error at ({:.2f}, {:.2f}, {:.2f}).'.format(*point))
            return None

    def target_angles(self, angles):
        try:
            for servo, angle in zip(self.servos, angles):
                servo.set_target(angle)
        except ServoError:
            logger.error('Arm is unable to reach angles ({:.2f}, {:.2f}, {:.2f})'.format(*angles))

    def get_position(self):
        angles = tuple(servo.get_position() for servo in self.servos)

        try:
            return self.fk_solver(self.lengths, angles)[-1]
        except (ValueError, ZeroDivisionError):
            return None


class Dummy:
    """
    Implements a dummy class that is completely inert.
    """

    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, item):
        return self.dummy_function

    def __setattr__(self, key, value):
        pass

    def dummy_function(*args, **kwargs):
        pass


class Agility:
    def __init__(self, arm):
        # Set up arm.
        self.arm = arm

        # Set up Usc.
        try:
            self.usc = Usc()
            logger.info("Successfully attached to Maestro's low-level interface.")
        except ConnectionError:
            self.usc = Dummy()
            logger.warn("Failed to attached to Maestro's low-level interface. "
                        "If not debugging, consider this a fatal error.")

        # Set up virtual COM and TTL ports.
        try:
            self.maestro = Maestro()
            logger.info("Successfully attached to Maestro's command port.")
        except ConnectionError:
            self.maestro = Dummy()
            logger.warning("Failed to attached to Maestro's command port. "
                           "If not debugging, consider this a fatal error.")

        self.zero()

    @staticmethod
    def smooth(a, b, n):
        """
        Create a smooth transition from a to b in n steps.
        :param a: The first array.
        :param b: The second array.
        :param n: The number of steps.
        :return: An array from [a, b).
        """

        assert(a.shape == b.shape)
        assert(n > 1)

        # Compute delta.
        delta = (b - a) / n

        # Allocate n-1 with dimension d+1.
        shape = (n, *a.shape)
        inter = np.empty(shape)

        for i in range(n):
            inter[i] = a + i * delta

        return inter

    def draw(self, drawing, v, x, z, lift):
        """
        Draw a given drawing.
        :param drawing: A Drawing object.
        :param v: The maximum linear velocity in cm / s.
        :param x: The x-offset.
        :param z: The z-height at which the writing utensil contacts the paper.
        :param lift: The height to lift
        """

        # Pixel to cm conversion factor.
        px_cm = 2.54 / 96

        # Begin path generation.
        logger.info('Generating path.')

        # Image size info.
        width = drawing.viewport[0] / 2 * px_cm

        # Replace consecutive nans and leave only a single nan.
        a = drawing.points
        b = np.roll(a, 1)
        remove = np.where(~(np.isfinite(a) | np.isfinite(b)))
        a = np.delete(a, remove)

        # Convert points to 3D. Real component will by y.
        points = np.empty((len(a), 3))
        points[:, 0] = np.imag(a) * px_cm
        points[:, 1] = np.real(a) * px_cm
        points[:, 2] = z

        # Move up x and left by viewport[0] / 2
        points += (x, -width, 0)

        # Replace all nans with a lift.
        loc = np.where(~np.isfinite(points))[0]
        points[loc] = points[loc - 1] + (0, 0, lift)

        # Compute velocity in ms.
        diff = np.diff(points, axis=0)
        distances = np.linalg.norm(diff, axis=1)
        dts = distances / v * 1000
        dts = np.hstack((0, dts))

        # Convert to angles.
        logger.info('Converting to angles.')
        angles = [self.arm.get_angles(p) for p in points]

        # Check for out of bound.
        if None in angles:
            logger.error('Path has one or more unreachable poses.')
        else:
            logger.info('Completed path generation.')

        return angles, dts

    def execute(self, angles, dts):
        """
        A worker to consume angles in queue.
        :param angles: A list of angles.
        :param dts: A list of dt.
        """

        # Assertion check.
        assert len(dts) == len(angles)

        # Get initial servo positions.
        self.maestro.get_multiple_positions(self.arm)

        # Execute.
        for dt, angles in zip(dts, angles):
            self.arm.target_angles(angles)
            self.maestro.end_together(self.arm, dt)
            self.wait()

    def move_to(self, target, v):
        """
        Move the arm to a target with a given linear velocity.
        :param target: The target tuple (x, y, z, phi).
        :param v: The linear velocity in cm/s.
        """

        current = self.arm.get_position()

        x1, y1, z1 = current
        x2, y2, z2 = target[:3]
        dst = ((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2) ** (1 / 2)

        dt = dst / v * 1000
        angles = self.arm.get_angles(target)

        if angles:
            self.arm.target_angles(angles)
            self.maestro.end_together(self.arm, dt)
            self.wait()

    def configure(self):
        """
        Configure the Maestro by writing home positions and other configuration data to the device.
        """

        settings = self.usc.getUscSettings()
        settings.serialMode = uscSerialMode.SERIAL_MODE_USB_DUAL_PORT

        for servo in self.arm:
            servo.zero()
            channel = settings.channelSettings[servo.channel]
            channel.mode = ChannelMode.Servo
            channel.homeMode = HomeMode.Goto
            channel.home = servo.target
            channel.minimum = (servo.min_qus // 64) * 64
            channel.maximum = -(-servo.max_qus // 64) * 64

        self.usc.setUscSettings(settings, False)
        self.usc.reinitialize(500)

    def go_home(self):
        """
        Let the Maestro return all servos to home.
        """

        self.maestro.go_home()

    def zero(self):
        for servo in self.arm:
            servo.set_target(0)

        self.maestro.end_together(self.arm)
        self.wait()

    def wait(self, servos=None):
        """
        Block until all servos have reached their targets.
        :param servos: A list of servos. If None, checks if all servos have reached their targets.
        """

        while not self.is_at_target(servos):
            time.sleep(0.001)

    def is_at_target(self, servos=None):
        """
        Check if servos are at their target.
        :param servos: One or more servo objects. If None, checks if all servos have reached their targets.
        :return: True if all servos are at their targets, False otherwise.
        """

        if servos is None:
            return not self.maestro.get_moving_state()
        elif isinstance(servos, Servo):
            self.maestro.get_position(servos)

            if servos.at_target():
                return True

            return False
        else:
            self.maestro.get_multiple_positions(servos)

            if all(servo.at_target() for servo in servos):
                return True

            return False

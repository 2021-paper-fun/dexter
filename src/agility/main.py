from .maestro import Maestro
from .usc import Usc
from .usc import enum
import numpy as np
import math
import time
import logging


logger = logging.getLogger('universe')

uscSerialMode = enum.uscSerialMode
ChannelMode = enum.ChannelMode
HomeMode = enum.HomeMode


class ServoError(Exception):
    pass


class Servo:
    tau = 2 * math.pi

    def __init__(self, channel, min_ang, max_ang, min_pwm, max_pwm, max_vel,
                 bias=0, direction=1, period=20, multiplier=1):
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
        :param period: The period of each pulse (ms).
        :param multiplier: The multiplier for period.

        Angle Range
        ===========
        Different servo brands have different directions of spin.
        Higher PWM always corresponds to the positive direction.

        Setting Bias
        ============
        Bias should be adjusted such that the servo is at kinematic "0" degree when it's target is 0 degrees.
        This is used to compensate for ridge spacing and inaccuracies during installation.
        Think of this like the "home" value of the servo.

        Setting Direction
        =================
        If higher PWM corresponds to counterclockwise:
            If the servo points in a positive axis and is grounded, direction is 1.
            If the servo points in a positive axis and is not grounded, direction is -1.
            If the servo points in a negative axis and is grounded, direction is -1.
            If the servo points in a negative axis and is not grounded, direction is 1.

        If higher PWM corresponds to clockwise:
            If the servo points in a positive axis and is grounded, direction is -1.
            If the servo points in a positive axis and is not grounded, direction is 1.
            If the servo points in a negative axis and is grounded, direction is 1.
            If the servo points in a negative axis and is not grounded, direction is -1.
        """

        self.channel = channel
        self.min_rad = math.radians(min_ang)
        self.max_rad = math.radians(max_ang)
        self.min_qus = min_pwm * 4
        self.max_qus = max_pwm * 4
        self.max_vel = max_vel

        self.bias = math.radians(bias)
        self.direction = direction
        self.period = period
        self.multiplier = multiplier

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

        if self.period == 20:
            self.k_vel = 10
        elif self.period < 20:
            self.k_vel = self.period
        else:
            self.k_vel = self.period / 2

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

    def target(self, constraints, solution=0):
        try:
            angles = self.ik_solver(self.lengths, constraints)[solution]
            for servo, angle in zip(self.servos, angles):
                servo.set_target(angle)

            return True
        except (ValueError, ZeroDivisionError, ServoError):
            logger.error('Arm is unable to reach constraint ({:.2f}, {:.2f}, {:.2f}, {:.2f}).'.format(*constraints))

        return False

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

        logger.info('Agility initialized.')

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

    def get_position(self):
        """
        Get the position of the arm.
        :return: A tuple of (x, y, z).
        """

        self.maestro.get_multiple_positions(self.arm)
        return self.arm.get_position()

    def draw(self, drawing, v, x, z, lift):
        """
        Draw a given drawing.
        :param drawing: A Drawing object.
        :param v: The maximum linear velocity in cm / s.
        :param x: The x-offset.
        :param z: The z-height at which the writing utensil contacts the paper.
        :param lift: The height to lift
        """

        assert x >= 0
        assert v > 0
        assert lift > 0

        # Define constants.
        dz = -0.5
        min_phi = 3 * math.pi / 4
        max_phi = 5 * math.pi / 4
        slow_dt = 500

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
        points = np.empty((a.shape[0], 3))
        points[:, 0] = np.imag(a) * px_cm
        points[:, 1] = np.real(a) * px_cm * -1

        # Push down more as distance from origin increases.
        r = np.linalg.norm(points[:, :2], axis=1) / self.arm.length
        points[:, 2] = z + dz * r

        # Move up x and left by viewport[0] / 2.
        points += (x, width, 0)

        # Replace all nans with a lift and insert a move right after.
        loc = np.where(~np.isfinite(points[:, 0]))[0]
        print(loc)
        points[loc] = points[loc - 1] + (0, 0, lift)
        points = np.insert(points, loc + 1, points[(loc + 1) % len(points)] + (0, 0, lift), axis=0)

        # Insert last point at start.
        points = np.insert(points, 0, points[-1], axis=0)

        # Compute velocity in ms.
        diff = np.diff(points, axis=0)
        distances = np.linalg.norm(diff, axis=1)
        dts = distances / v * 1000

        # Slow move from current position to starting position.
        dts = np.hstack((2000, dts))

        # Slow move up and down.
        loc2 = loc[:-1] + 1 + np.arange(0, len(loc) - 1)
        loc2 = np.concatenate((loc2, loc2 + 2))
        dts[loc2] = slow_dt

        # Slow move to end.
        dts[-1] = slow_dt

        # Compute phi.
        r = np.linalg.norm(points[:, :2], axis=1) / self.arm.length
        phi = max_phi - (max_phi - min_phi) * r
        constraints = np.empty((points.shape[0], 4))
        constraints[:, :3] = points
        constraints[:, 3] = phi

        logger.info('Completed path generation.')

        return constraints, dts

    def execute(self, constraints, dts, event=None):
        """
        Execute given angles and times.
        :param constraints: A list of constraints.
        :param dts: A list of dt.
        :param event: Threading Event for early exit.
        :return: True if completed. False if exited early.
        """

        # Assertion check.
        assert len(dts) == len(constraints)

        # Get initial servo positions.
        self.maestro.get_multiple_positions(self.arm)

        # Execute.
        for constraint, dt in zip(constraints, dts):
            self.arm.target(constraint)
            self.sync(self.arm, dt)

            if event is not None and event.is_set():
                return False

        return True

    def move_relative(self, delta, phi, v):
        """
        Move the arm relative to its current position.
        :param delta: The tuple (dx, dy, dz).
        :param phi: The angle phi.
        :param v: The linear velocity in cm/s
        """

        assert v > 0

        self.maestro.get_multiple_positions(self.arm)
        current = self.arm.get_position()

        x1, y1, z1 = current
        dx, dy, dz = delta

        dst = (dx ** 2 + dy ** 2 + dz ** 2) ** (1 / 2)
        dt = dst / v * 1000

        if self.arm.target((x1 + dx, y1 + dy, z1 + dz, phi)):
            self.sync(self.arm, dt)

    def move_absolute(self, target, phi, v):
        """
        Move the arm to a target with a given linear velocity.
        :param target: The target tuple (x, y, z).
        :param phi: The angle phi.
        :param v: The linear velocity in cm/s.
        """

        assert v > 0

        self.maestro.get_multiple_positions(self.arm)
        current = self.arm.get_position()

        x1, y1, z1 = current
        x2, y2, z2 = target
        dst = ((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2) ** (1 / 2)

        dt = dst / v * 1000

        if self.arm.target((x2, y2, z2, phi)):
            self.sync(self.arm, dt)

    def configure(self):
        """
        Configure the Maestro by writing home positions and other configuration data to the device.
        """

        settings = self.usc.getUscSettings()
        settings.serialMode = uscSerialMode.SERIAL_MODE_USB_CHAINED
        settings.enableCrc = False

        multipliers = set(servo.multiplier for servo in self.arm)
        multipliers.remove(1)

        if len(multipliers) > 1:
            raise ValueError('More than one multiplier found.')
        elif len(multipliers) == 1:
            settings.servoMultiplier = multipliers.pop()
        else:
            settings.servoMultiplier = 1

        periods = set(servo.period for servo in self.arm)

        if len(periods) != 1:
            raise ValueError('More than one period found.')
        else:
            settings.miniMaestroServoPeriod = periods.pop() * 4000

        empty_channels = list(range(len(settings)))

        for servo in self.arm:
            servo.zero()
            empty_channels.remove(servo.channel)

            channel = settings.channelSettings[servo.channel]

            if servo.multiplier != 1:
                channel.mode = ChannelMode.ServoMultiplied
            else:
                channel.mode = ChannelMode.Servo

            channel.homeMode = HomeMode.Goto
            channel.home = servo.target
            channel.minimum = (servo.min_qus // 64) * 64
            channel.maximum = -(-servo.max_qus // 64) * 64

        for c in empty_channels:
            channel = settings.channelSettings[c]
            channel.mode = ChannelMode.Output
            channel.homeMode = HomeMode.Off

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

        self.sync(self.arm, 2000)

    def sync(self, servos, t):
        """
        Ensure that the given servos reach their target in a given time.
        Blocks until completion.
        :param servos: A list of servos.
        :param t: The time in ms for the operation. Set to 0 for max speed.
        """

        if t == 0:
            t = max(abs(servo.target - servo.pwm) / servo.max_vel * servo.period for servo in servos)

        end = time.time() + t / 1000
        self.maestro.end_together(servos, t, update=True)

        while not self.is_at_target(servos):
            dt = (end - time.time()) * 1000

            if dt < 0:
                dt = 0

            self.maestro.end_together(servos, dt, update=True)

    def wait(self, servos):
        """
        Block until all servos have reached their targets.
        :param servos: A list of servos.
        """

        while not self.is_at_target(servos):
            time.sleep(0.005)

    def is_moving(self):
        return self.maestro.get_moving_state()

    def is_at_target(self, servos):
        """
        Check if servos are at their target. Efficient when used on the whole arm.
        :param servos: One or more servo objects.
        :return: True if all servos are at their targets, False otherwise.
        """

        if isinstance(servos, Servo):
            self.maestro.get_position(servos)

            if servos.at_target():
                return True

            return False
        else:
            self.maestro.get_multiple_positions(servos)

            if all(servo.at_target() for servo in servos):
                return True

            return False

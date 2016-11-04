from agility.maestro import Maestro
import numpy as np
import math
import time
from util import logger


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
        self.length = sum(lengths)

        self.fk_solver = fk_solver
        self.ik_solver = ik_solver

        self.last_position = None

    def __getitem__(self, key):
        return self.servos[key]

    def __len__(self):
        return len(self.servos)

    def target_point(self, point, solution=0):
        try:
            angles = self.ik_solver(self.lengths, point)[solution]

            for servo, angle in zip(self.servos, angles):
                servo.set_target(angle)

            self.last_position = point
        except (ServoError, ValueError, ZeroDivisionError):
            logger.error('Arm is unable to reach point ({:.2f}, {:.2f}, {:.2f})'.format(*point))
            return False

        return True

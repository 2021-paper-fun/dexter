from agility import Servo, Maestro, usc
from math import *
import time
from util import logger


def is_at_target(maestro, servos=None):
    """
    Check if servos are at their target.
    :param servos: One or more servo objects. If None, checks if all servos have reached their targets.
    :return: True if all servos are at their targets, False otherwise.
    """

    if servos is None:
        return not maestro.get_moving_state()
    elif isinstance(servos, Servo):
        maestro.get_position(servos)

        if servos.at_target():
            return True

        return False
    else:
        maestro.get_multiple_positions(servos)

        if all(servo.at_target() for servo in servos):
            return True

        return False


maestro = Maestro()

servo1 = Servo(0, 0, 270, 500, 2500, 180, bias=0, direction=1)
servo2 = Servo(1, -90, 90, 600, 2400, 140, bias=0, direction=1)

# Move all servos to 0.
servo1.set_target(radians(0))
servo2.set_target(radians(0))

# Have maestro execute the instructions.
# Have both servos end together. Reach the target in 1000 milliseconds (1 second).
maestro.end_together((servo1, servo2), t=1000, update=True)

# Wait until the servos get their target.
while not is_at_target(maestro, (servo1, servo2)):
    time.sleep(0.001)

# Move each servo to 90 degrees and then 0 degrees 5 times.
for i in range(5):
    # Set each servo to 90 degrees.
    servo1.set_target(radians(90))
    servo2.set_target(radians(90))
    maestro.set_target(servo2)

    # Have maestro execute the instructions.
    # Have both servos end together. Reach the target in 1000 milliseconds (1 second).
    maestro.end_together((servo1, servo2), t=1000, update=True)

    # Wait until the servos get their target.
    while not is_at_target(maestro, (servo1, servo2)):
        time.sleep(0.001)

    # Set each servo to 0 degrees.
    servo1.set_target(radians(0))
    servo2.set_target(radians(0))

    # Have maestro execute the instructions.
    # Have both servos end together. Reach the target in 1000 milliseconds (1 second).
    maestro.end_together((servo1, servo2), t=1000, update=True)

    # Wait until the servos get their target.
    while not is_at_target(maestro, (servo1, servo2)):
        time.sleep(0.001)

# Move all servos to 0.
servo1.set_target(radians(0))
servo2.set_target(radians(0))

# Have maestro execute the instructions.
# Have both servos end together. Reach the target in 1000 milliseconds (1 second).
maestro.end_together((servo1, servo2), t=1000, update=True)

# Start servo2 300 milliseconds after servo1 ends. Both will go to 90 degrees.
servo1.set_target(radians(90))
servo2.set_target(radians(90))

# Have maestro execute the instructions for servo1 only.
# Get servo1 to 90 degrees in 1 second.
maestro.end_in(servo1, t=1000, update=True)

# Wait until servo1 reaches its target.
while not is_at_target(maestro, servo1):
    time.sleep(0.001)

# Wait 300 milliseconds.
time.sleep(0.300)

# Have maestro execute the instructions for servo2.
maestro.end_in(servo2, t=1000, update=True)

# Wait until servo2 reaches its target.
while not is_at_target(maestro, servo2):
    time.sleep(0.001)
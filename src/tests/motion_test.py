from agility import Servo, Maestro, usc
from math import *


maestro = Maestro()
servo1 = Servo(0, 0, 270, 500, 2500, 160, -4, 1)

servo1.set_target(radians(270 / 2))
maestro.set_target(servo1)
from agility import Servo, Arm
from kinematics import solve_fk, solve_ik


class Android:
    servo1 = Servo(0, -135, 135, 500, 2500, 180, bias=3, direction=1, period=4)
    servo2 = Servo(1, -135, 135, 500, 2500, 180, bias=-3, direction=-1, period=4)
    servo3 = Servo(2, -110, 160, 500, 2500, 180, bias=3, direction=1, period=4)
    servo4 = Servo(3, -120, 60, 800, 2700, 120, bias=6, direction=-1, period=4, multiplier=5)

    lengths = (18.0, 18.0, 0.8, 5.6)

    arm = Arm(servo1, servo2, servo3, servo4, lengths, solve_fk, solve_ik)
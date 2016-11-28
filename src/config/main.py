from agility import Servo, Arm
from kinematics import solve_fk, solve_ik


class Android:
    servo1 = Servo(0, -135, 135, 500, 2500, 180, bias=1, direction=1)
    servo2 = Servo(1, -135, 135, 500, 2500, 180, bias=5, direction=-1)
    servo3 = Servo(2, -110, 160, 500, 2500, 180, bias=3, direction=1)
    servo4 = Servo(3, -120, 60, 800, 2700, 120, bias=0, direction=-1)

    diameter = 1
    height = 5.05

    lengths = (18.0, 18.0, 0.82 + diameter / 2, 2.5 + height)

    arm = Arm(servo1, servo2, servo3, servo4, lengths, solve_fk, solve_ik)


class Crossbar:
    ip = '127.0.0.1'
    realm = 'realm1'
    authid = 'arm'
    ticket = '5gRZ_E4YCE4!E$jX'
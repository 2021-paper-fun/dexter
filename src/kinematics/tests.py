from kinematics.ik import solve_ik
from kinematics.fk import solve_fk
from math import *
import time


def test_system(lengths, constraints):
    target = constraints[:3]

    try:
        angles = solve_ik(lengths, constraints)
        for angle in angles:
            print(angle)
    except (ValueError, ZeroDivisionError):
        angles = []

    if len(angles) > 0:
        for angle in angles:
            points = solve_fk(lengths, angle)

            if all(round(points[-1][i], 5) == round(target[i], 5) for i in range(3)):
                print('Test passed!')
            else:
                print('Test failed!')


def test_speed(lengths, constraints):
    print(solve_ik(lengths, constraints))
    start = time.time()
    for i in range(100000):
        solve_ik(lengths, constraints)
    print((time.time() - start))

#
test_system((10.0, 10.0, 2.0, 5.0), (-1.06, 2.16, 13.66, pi/2))
test_system((10.0, 10.0, 2.0, 5.0), (0.0, 15.0, 2.0, pi))
test_system((10.0, 10.0, 2.0, 5.0), (4.0, 3.0, 2.0, -1.3))
test_system((10.0, 10.0, 2.0, 5.0), (5.0, 0.0, 10.0, pi / 2))

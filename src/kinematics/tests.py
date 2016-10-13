from kinematics.main import *
import time


def test_system(lengths, constraints):
    target = constraints[:3]

    angles = IK.solve(lengths, constraints)
    print('Number of solutions: {}.'.format(len(angles)))

    if len(angles) > 0:
        for angle in angles:
            points = FK.solve(lengths, angle)

            if all(round(points[-1][i], 5) == round(target[i], 5) for i in range(3)):
                print('Test passed!')
            else:
                print('Test failed!')


def test_speed(lengths, constraints):
    start = time.time()
    for i in range(10000):
        IK.solve(lengths, constraints)
    print((time.time() - start))


test_speed((10, 10, 2, 5), [10, 2, -3, pi])
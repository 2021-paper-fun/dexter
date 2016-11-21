from math import *


def jacobian(lengths, angles):
    l1, l2, l3, l4 = lengths
    theta1, theta2, theta3, theta4 = angles

    j = [[-l1 * sin(theta1) * sin(theta2) - l2 * sin(theta1) * sin(theta2 + theta3) - l3 * cos(theta1)
          - l4 * sin(theta1) * sin(theta2 + theta3 + theta4),
          l1 * cos(theta1) * cos(theta2) + l2 * cos(theta1) * cos(theta2 + theta3) + l4 * cos(theta1)
          * cos(theta2 + theta3 + theta4),
          l2 * cos(theta1) * cos(theta2 + theta3) + l4 * cos(theta1) * cos(theta2 + theta3 + theta4),
          l4 * cos(theta1) * cos(theta2 + theta3 + theta4)],

         [l1 * sin(theta2) * cos(theta1) + l2 * sin(theta2 + theta3) * cos(theta1) - l3 * sin(theta1)
          + l4 * sin(theta2 + theta3 + theta4) * cos(theta1),
          l1 * sin(theta1) * cos(theta2) + l2 * sin(theta1) * cos(theta2 + theta3) + l4 * sin(theta1)
          * cos(theta2 + theta3 + theta4),
          l2 * sin(theta1) * cos(theta2 + theta3) + l4 * sin(theta1) * cos(theta2 + theta3 + theta4),
          l4 * sin(theta1) * cos(theta2 + theta3 + theta4)],

         [0,
          -l1 * sin(theta2) - l2 * sin(theta2 + theta3) - l4 * sin(theta2 + theta3 + theta4),
          -l2 * sin(theta2 + theta3) - l4 * sin(theta2 + theta3 + theta4), -l4
          * sin(theta2 + theta3 + theta4)],

         [0, 0, 0, 0],

         [0, 1, 1, 1],

         [1, 0, 0, 0]]

    return j
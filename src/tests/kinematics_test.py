from kinematics import solve_fk, solve_ik
from math import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def test_system(lengths, constraints):
    target = constraints[:3]
    angles = solve_ik(lengths, constraints)
    # print(angles)

    for angle in angles:
        points = solve_fk(lengths, angle)

        if any([round(points[-1][i], 5) != round(target[i], 5) for i in range(3)]):
            return False

    return True

def generate_sphere(lengths, phi, d, n):
    p = np.random.uniform(0, 2 * pi, n)
    costheta = np.random.uniform(-1, 1, n)
    u = np.random.uniform(0, 1, n)

    theta = np.arccos(costheta)
    r = d * u ** (1 / 3)
    x = r * np.sin(theta) * np.cos(p)
    y = r * np.sin(theta) * np.sin(p)
    z = r * np.cos(theta)

    x_good, x_bad = [], []
    y_good, y_bad = [], []
    z_good, z_bad = [], []

    for i in range(len(x)):
        if test_system(lengths, (x[i], y[i], z[i], phi)):
            x_good.append(x[i])
            y_good.append(y[i])
            z_good.append(z[i])
        else:
            x_bad.append(x[i])
            y_bad.append(y[i])
            z_bad.append(z[i])

    return x_good, y_good, z_good, x_bad, y_bad, z_bad


def graph(points, d):
    x_good, y_good, z_good, x_bad, y_bad, z_bad = points

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal')
    ax.scatter([-d, d, 0, 0, 0, 0], [0, 0, -d, d, 0, 0], [0, 0, 0, 0, -d, d], alpha=0)
    ax.scatter(x_good, y_good, z_good, c='g', alpha=0.1)
    ax.scatter(x_bad, y_bad, z_bad, c='r')
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')

    fig2 = plt.figure(figsize=(25, 7.5), dpi=80)
    fig2.suptitle('Null Space Analysis', fontsize=16, fontweight='bold')

    ax1 = fig2.add_subplot(131)
    ax1.set_aspect('equal')
    ax1.scatter(x_good, y_good, c='g', alpha=0.2)
    ax1.scatter(x_bad, y_bad, c='r')
    ax1.set_xlabel('X Axis')
    ax1.set_ylabel('Y Axis')

    ax2 = fig2.add_subplot(132)
    ax2.set_aspect('equal')
    ax2.scatter(x_good, z_good, c='g', alpha=0.2)
    ax2.scatter(x_bad, z_bad, c='r')
    ax2.set_xlabel('X Axis')
    ax2.set_ylabel('Z Axis')

    ax3 = fig2.add_subplot(133)
    ax3.set_aspect('equal')
    ax3.scatter(y_good, z_good, c='g', alpha=0.2)
    ax3.scatter(y_bad, z_bad, c='r')
    ax3.set_xlabel('Y Axis')
    ax3.set_ylabel('Z Axis')

    fig2.savefig('analysis.png', bbox_inches='tight')

    plt.show()


lengths = (10.0, 10.0, 10.0, 5.0)
phi = radians(135)

# test_system(lengths, (15, 0, 5, phi))

d = lengths[0] + lengths[1] + sin(phi) * lengths[3] - 1
points = generate_sphere(lengths, phi, d, 10000)
graph(points, d)
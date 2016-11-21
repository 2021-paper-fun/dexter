from kinematics.jacobian import *
from kinematics import solve_ik
from config import Android
import numpy as np
import matplotlib.pyplot as plt


lengths = Android.lengths
torques = np.array([13.5, 13.5, 13.5, 1.8])

nx, ny = (100, 100)
x_lim = (-21.59 / 2, 21.59 / 2)
y_lim = (6, 27.94 + 6)
xs = np.linspace(*x_lim, nx)
ys = np.linspace(*y_lim, nx)
z = -7.6

data = []

for x in xs:
    for y in ys:
        angles = solve_ik(lengths, (x, y, z, np.pi))[0]
        j = jacobian(lengths, angles)
        forces = np.dot(j, torques)
        data.append((x, y, forces[0], forces[1], forces[2]))

data = np.array(data)
data[:, 2:] = np.abs(data[:, 2:])

data[:, 2] /= np.max(data[:, 2])
data[:, 3] /= np.max(data[:, 3])
data[:, 4] /= np.max(data[:, 4])

fig = plt.figure(dpi=80, figsize=(12, 7.5))

ax1 = fig.add_subplot(121)
ax1.set_aspect('equal')
ax1.set_title('Force along X axis', color='white', fontsize=20, y=1.05)
ax1.set_xlim(x_lim)
ax1.set_ylim(y_lim)
ax1.tick_params(axis='x', colors='white')
ax1.tick_params(axis='y', colors='white')
ax1.scatter(data[:, 0], data[:, 1], c=data[:, 2], s=25)

ax2 = fig.add_subplot(122)
ax2.set_aspect('equal')
ax2.set_title('Force along Y axis', color='white', fontsize=20, y=1.05)
ax2.set_xlim(x_lim)
ax2.set_ylim(y_lim)
ax2.tick_params(axis='x', colors='white')
ax2.tick_params(axis='y', colors='white')
ax2.scatter(data[:, 0], data[:, 1], c=data[:, 3], s=25)

plt.savefig('jacobian.png', bbox_inches='tight', transparent=True)
plt.show()
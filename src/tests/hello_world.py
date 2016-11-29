from config import Android
from agility import Agility
from math import pi


agility = Agility(Android.arm)

char = {
    'd': (17.2, 4.6),
    'e': (19.2, 5.3),
    'h': (17, -1.4),
    'l': (16.2, -7.2),
    'o': (18.0, -6.8),
    'r': (18.8, 3.6),
    'w': (19.0, 7.6),
    ' ': (13.4, -1)
}

depth = -6
text = 'hello world'
constraints = []

for c in text:
    constraints.append(char[c] + (0, pi))
    constraints.append(char[c] + (depth, pi))
    constraints.append(char[c] + (0, pi))

for constraint in constraints:
    agility.move_absolute(constraint, 3)
from drawing import Drawing
from config import Android
from agility import Agility, Maestro
from util import logger

agility = Agility(Android.arm)
agility.zero()
print(agility.get_position())

landscape = (11.0 * 96, 8.5 * 96)
portrait = (landscape[1], landscape[0])

drawing = Drawing('../svg/weather/2.svg', portrait, center=True, resize=True, dx=20)

if drawing.preview():
    angles, dts = agility.draw(drawing, 10, 6, -7.4, 4)
    agility.execute(angles, dts)
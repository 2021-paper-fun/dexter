from drawing import Drawing
from config import Android
from agility import Agility, Maestro
from util import logger

agility = Agility(Android.arm)
agility.zero()

landscape = (11.0 * 96, 8.5 * 96)
portrait = (landscape[1], landscape[0])

drawing = Drawing('../svg/weather/23.svg', portrait, center=True, resize=True, dx=5)

if drawing.preview():
    angles, dts = agility.draw(drawing, 15, 6, -7.4, 4)
    agility.execute(angles, dts)
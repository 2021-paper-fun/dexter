from drawing import Drawing
from config import Android
from agility import Agility, Maestro
from util import logger

agility = Agility(Android.arm)

landscape = (11.0 * 96, 8.5 * 96)
portrait = (8.5 * 96, 11.0 * 96)

drawing = Drawing('svg/apple.svg', portrait, center=True, resize=True, dx=5)

if drawing.preview():
    angles, dts = agility.draw(drawing, 10, 6, -8, 4)
    agility.execute(angles, dts)
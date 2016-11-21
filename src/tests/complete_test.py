from drawing import Drawing
from config import Android
from agility import Agility, Maestro
from util import logger

agility = Agility(Android.arm)

landscape = (11.0 * 96, 8.5 * 96)
portrait = (landscape[1], landscape[0])

drawing = Drawing('svg/git-cat.svg', portrait, center=True, resize=True, dx=5)

if drawing.preview():
    angles, dts = agility.draw(drawing, 10, 6, -7.6, 4)
    agility.execute(angles, dts)
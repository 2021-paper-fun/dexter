from drawing import Drawing
from config import Android
from agility import Agility, Maestro
from util import logger

landscape = (11.0 * 96, 8.5 * 96)
portrait = (8.5 * 96, 11.0 * 96)

drawing = Drawing('svg/odie.svg', portrait, center=True, resize=True)
agility = Agility(Android.arm)

if drawing.preview():
    constraints, dts = agility.draw(drawing, 10, 8, -8, 4)
    agility.execute(constraints, dts)
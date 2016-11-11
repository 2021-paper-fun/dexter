from drawing import Drawing
from config import Android
from agility import Agility, Maestro
from util import logger

landscape = (11.0 * 96, 8.5 * 96)
portrait = (8.5 * 96, 11.0 * 96)

drawing = Drawing('svg/circuit_diagram.svg', portrait, center=True, resize=True)
drawing.preview()
agility = Agility(Android.arm)
angles, dts = agility.draw(drawing, 10, 6, -2, 2)
agility.execute(angles, dts)
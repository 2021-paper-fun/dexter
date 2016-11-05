from drawing import Drawing
from util import logger

landscape = (11.0 * 96, 8.5 * 96)
portrait = (8.5 * 96, 11.0 * 96)

drawing = Drawing('svg/xkcd.svg', landscape, center=True, resize=True)
print(drawing.preview())
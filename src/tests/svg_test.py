from drawing import Drawing
from util import logger

viewport = (11.0 * 96, 8.5 * 96)
drawing = Drawing('tnr.svg', viewport, center=True, resize=True)
print(drawing.preview())
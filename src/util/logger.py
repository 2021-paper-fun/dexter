import logging
from logging import StreamHandler


logger = logging.getLogger('universe')
handler = StreamHandler()
logger.setLevel(logging.INFO)
formatter = logging.Formatter(fmt='%(asctime)s.%(msecs).03d | %(levelname)s | %(message)s', datefmt='%H:%M:%S')
handler.setFormatter(formatter)
logger.addHandler(handler)
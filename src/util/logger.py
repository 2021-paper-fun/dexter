import logging
from logging import StreamHandler


logger = logging.getLogger('universe')
handler = StreamHandler()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(fmt='%(asctime)s | %(levelname)s | %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
handler.setFormatter(formatter)
logger.addHandler(handler)
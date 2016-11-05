import logging
from logging import StreamHandler


logger = logging.getLogger('universe')
handler = StreamHandler()
logger.setLevel(logging.INFO)
formatter = logging.Formatter(fmt='%(asctime)s | %(levelname)s | %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
handler.setFormatter(formatter)
logger.addHandler(handler)
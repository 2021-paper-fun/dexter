import logging
from logging import StreamHandler
from logging.handlers import QueueHandler
from queue import Queue
import time


class Handler(QueueHandler):
    """
    An asynchronous logger using a proxy-based queue.
    """

    def prepare(self, record):
        # msg = self.format(record)
        msg = (time.time(), record.msg)
        return msg


logger = logging.getLogger('universe')
logger.setLevel(logging.INFO)

handler = StreamHandler()
formatter = logging.Formatter(fmt='%(asctime)s.%(msecs).03d | %(levelname)s | %(message)s', datefmt='%H:%M:%S')
handler.setFormatter(formatter)
logger.addHandler(handler)

logging_queue = Queue()
handler2 = Handler(logging_queue)
logger.addHandler(handler2)

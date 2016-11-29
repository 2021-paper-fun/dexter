import asyncio
import ssl
from threading import Event, Lock
import os
from math import pi
import pyowm
import requests

from util import logger, logging_queue
from drawing import Drawing
from agility import Agility
from config import Android, Crossbar

from autobahn import wamp
from autobahn.asyncio.wamp import ApplicationSession
from util import ApplicationRunner
from concurrent.futures import ThreadPoolExecutor


class Numeric:
    @staticmethod
    def to_float(num):
        try:
            return float(num)
        except ValueError:
            return None

    @staticmethod
    def to_int(num):
        try:
            return int(num)
        except ValueError:
            return None


class Weather:
    def __init__(self):
        self.icons = {
            '01d': '2.svg',
            '01n': '3.svg',
            '02d': '8.svg',
            '02n': '9.svg',
            '03d': '14.svg',
            '03n': '14.svg',
            '04d': '25.svg',
            '04n': '25.svg',
            '09d': '18.svg',
            '09n': '18.svg',
            '10d': '17.svg',
            '10n': '17.svg',
            '11d': '27.svg',
            '11n': '27.svg',
            '13d': '23.svg',
            '13n': '23.svg',
            '50d': '10.svg',
            '50n': '11.svg'
        }

        self.owm = pyowm.OWM('5234e8b7ef96dbc75fad653baba4f7d0')

        r = requests.get('http://ipinfo.io/json')
        ip_info = r.json()
        self.location = '{}, {}, {}'.format(ip_info['city'], ip_info['region'], ip_info['country'])

        logger.info('Weather system initialized.')

    def get_icon(self):
        observation = self.owm.weather_at_place(self.location)
        w = observation.get_weather()
        icon = w.get_weather_icon_name()

        return 'weather/' + self.icons[icon]


class Cerebral(ApplicationSession):
    def __init__(self, *args, **kwargs):
        self.root = os.path.dirname(__file__)
        self.loop = asyncio.get_event_loop()
        self.executor = ThreadPoolExecutor(10)

        self.weather = None
        self.agility = None

        self.initialized = False
        self.work_lock = Lock()
        self.event = Event()

        self.points = [None, None, None]
        self.params = {
            'lift': 4.0,
            'speed': 10.0,
            'offset': 6.0
        }

        super().__init__(*args, **kwargs)

    def onConnect(self):
        logger.info('Connected to server.')
        self.join(self.config.realm, ['ticket'], Crossbar.authid)

    def onChallenge(self, challenge):
        logger.debug('Challenge received.')
        return Crossbar.ticket

    async def onJoin(self, details):
        logger.info('Joined "{}" realm.'.format(self.config.realm))

        await self.register(self)
        self.run(self.watch_logging)
        self.run(self.initialize)

    def onDisconnect(self):
        logger.info('Connection lost!')

    ####################
    # Special functions.
    ####################

    def run(self, fn, *args, **kwargs):
        return asyncio.wrap_future(self.executor.submit(fn, *args, **kwargs))

    def initialize(self):
        self.weather = Weather()
        self.agility = Agility(Android.arm)

        self.initialized = True
        logger.info('Initialization complete.')

    def watch_logging(self):
        while True:
            message = logging_queue.get()
            self.publish('arm.log', *message)

    ########################
    # Main remote functions.
    ########################

    @wamp.register('arm.ready')
    async def ready(self):
        if self.initialized:
            message = 'I\'m all good to go.'
        else:
            message = 'Give me a moment. I\'m currently initializing.'

        self.call('controller.speak', message)

    def _draw_weather(self):
        svg = self.weather.get_icon()

        landscape = (11.0 * 96, 8.5 * 96)
        portrait = (landscape[1], landscape[0])
        drawing = Drawing(os.path.join(self.root, 'svg', svg).replace('\\', '/'), portrait,
                          center=True, resize=True, dx=5)

        angles, dts = self.agility.draw(drawing, self.params['speed'], self.params['offset'], -7.6, self.params['lift'])
        self.event.clear()
        completed = self.agility.execute(angles, dts, event=self.event)

        if not completed:
            self.agility.zero()

    @wamp.register('arm.draw_weather')
    async def draw_weather(self):
        if not self.initialized:
            return self.call('controller.speak', 'Please wait. System is not initialized.')

        if self.work_lock.acquire(blocking=False):
            try:
                self.call('controller.speak', 'Drawing the weather.')
                await self.run(self._draw_weather)
            finally:
                self.work_lock.release()
        else:
            self.call('controller.speak', 'I am currently busy.')

    def _zero(self):
        self.agility.zero()

    @wamp.register('arm.zero')
    async def zero(self):
        if not self.initialized:
            return self.call('controller.speak', 'Please wait. System is not initialized.')

        if self.work_lock.acquire(blocking=False):
            try:
                self.call('controller.speak', 'Zeroing the arm.')
                await self.run(self._zero)
            finally:
                self.work_lock.release()
        else:
            self.call('controller.speak', 'I am currently busy.')

    @wamp.register('arm.stop')
    async def stop(self):
        if not self.work_lock.acquire(blocking=False):
            self.event.set()
            self.call('controller.speak', 'Stopping.')
        else:
            self.call('controller.speak', 'I cannot stop doing nothing.')

    @wamp.register('arm.info')
    async def info(self):
        text = 'The current configuration is as follows. ' \
               'Linear velocity: {}. ' \
               'X-offset: {}. ' \
               'Lift height: {}.'.format(self.params['speed'], self.params['offset'], self.params['lift'])

        self.call('controller.speak', text)

    def _set_parameter(self, param, value):
        value = Numeric.to_float(value)

        if value is None:
            return self.call('controller.speak', 'I don\'t recognize that number.')

        if param not in self.params:
            return self.call('controller.speak', 'I don\'t recognize that parameter.')

        self.params[param] = value

    @wamp.register('arm.set_parameter')
    async def set_parameter(self, param, value):
        self.call('controller.speak', 'Setting parameter.')
        await self.run(self._set_parameter, param, value)

    def _relative_move(self, direction, delta):
        delta = Numeric.to_float(delta)

        if delta is None:
            return self.call('controller.speak', 'I don\'t recognize that number.')

        dx = 0
        dy = 0
        dz = 0

        if direction == 'left':
            dy += delta
        elif direction == 'right':
            dy -= delta
        elif direction == 'forward':
            dx += delta
        elif direction == 'backward':
            dx -= delta
        elif direction == 'up':
            dz += delta
        elif direction == 'down':
            dz -= delta
        else:
            return self.call('controller.speak', 'I don\'t recognize that direction.')

        self.agility.move_relative((dx, dy, dz), pi, self.params['speed'])

    @wamp.register('arm.move_relative')
    async def relative_move(self, direction, delta):
        if not self.initialized:
            return self.call('controller.speak', 'Please wait. System is not initialized.')

        if self.work_lock.acquire(blocking=False):
            try:
                self.call('controller.speak', 'Moving.')
                await self.run(self._relative_move, direction, delta)
            finally:
                self.work_lock.release()
        else:
            self.call('controller.speak', 'I am currently busy.')

    def _absolute_move(self, x, y, z):
        x = Numeric.to_float(x)
        y = Numeric.to_float(y)
        z = Numeric.to_float(z)

        if x is None or y is None or z is None:
            return self.call('controller.speak', 'I don\'t recognize that coordinate.')

        self.agility.move_absolute((x, y, z), pi, self.params['speed'])

    @wamp.register('arm.move_absolute')
    async def relative_move(self, x, y, z):
        if not self.initialized:
            return self.call('controller.speak', 'Please wait. System is not initialized.')

        if self.work_lock.acquire(blocking=False):
            try:
                self.call('controller.speak', 'Moving.')
                await self.run(self._absolute_move, x, y, z)
            finally:
                self.work_lock.release()
        else:
            self.call('controller.speak', 'I am currently busy.')

    def _save_point(self, num):
        num = Numeric.to_int(num)

        if num is None or num < 1 or num > 3:
            return self.call('controller.speak', 'Invalid index.')

        self.points[num - 1] = self.agility.arm.get_position()

    @wamp.register('arm.save_point')
    async def save_point(self, num):
        if not self.initialized:
            return self.call('controller.speak', 'Please wait. System is not initialized.')

        self.call('controller.speak', 'Saving current location.')
        await self.run(self._save_point, num)

if __name__ == '__main__':
    # Configure SSL.
    context = ssl.SSLContext(ssl.PROTOCOL_SSLv23)
    context.check_hostname = False
    pem = ssl.get_server_certificate((Crossbar.ip, 443))
    context.load_verify_locations(cadata=pem)

    # Create application runner.
    runner = ApplicationRunner(url='wss://{}/ws'.format(Crossbar.ip), realm=Crossbar.realm,
                               ssl=context)

    # Run forever.
    runner.run(Cerebral)
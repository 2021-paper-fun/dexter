import asyncio
from cleverbot import Cleverbot
from datetime import datetime, timedelta
from functools import partial
from io import BytesIO
from math import pi
import os
import pyowm
import requests
import ssl
import threading

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


class Chatbot:
    def __init__(self):
        self.bot = Cleverbot()

        logger.info('Chatbot initialized.')

    def ask(self, text):
        return self.bot.ask(text)


class Image:
    def __init__(self):
        self.api_key = '3885957-fe1000226f3034f8bdbf7b9bf'
        self.url = 'https://pixabay.com/api/'

        logger.info('Image search system initialized.')

    def get_svg_url(self, q, index):
        payload = {
            'key': self.api_key,
            'q': q,
            'response_group': 'high_resolution',
            'image_type': 'vector',
        }

        r = requests.get(self.url, params=payload)

        try:
            images = r.json()
        except ValueError:
            return None

        hits = images['hits']

        for image in hits:
            if not image['vectorURL'].endswith('.svg'):
                hits.remove(image)

        if len(hits) <= index:
            return None

        return hits[index]['vectorURL']

    def get_all_url(self, q, index):
        payload = {
            'key': self.api_key,
            'q': q,
            'response_group': 'high_resolution'
        }

        r = requests.get(self.url, params=payload)

        try:
            images = r.json()
        except ValueError:
            return None

        hits = images['hits']

        if len(hits) <= index:
            return None

        return hits[index]['largeImageURL']

    def get_image(self, url):
        r = requests.get(url)
        return BytesIO(r.content)


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

    def get_now(self):
        observation = self.owm.weather_at_place(self.location)
        w = observation.get_weather()
        icon = w.get_weather_icon_name()

        return 'weather/' + self.icons[icon]

    def get_forecast(self, value, units):
        dt = datetime.now()

        if units == 'minutes' or units == 'minute':
            dt += timedelta(minutes=value)
        elif units == 'hours' or units == 'hour':
            dt += timedelta(hours=value)
        elif units == 'days' or units == 'day':
            dt += timedelta(days=value)
        elif units == 'weeks' or units == 'week':
            dt += timedelta(weeks=value)

        forecast = self.owm.daily_forecast(self.location)
        w = forecast.get_weather_at(dt)
        icon = w.get_weather_icon_name()

        return 'weather/' + self.icons[icon]


class Cerebral(ApplicationSession):
    def __init__(self, *args, **kwargs):
        self.root = os.path.dirname(__file__)
        self.loop = asyncio.get_event_loop()
        self.id = threading.get_ident()

        self.executor = ThreadPoolExecutor(10)

        self.agility = None
        self.chatbot = None
        self.image = None
        self.weather = None

        self.initialized = False
        self.work_lock = threading.Lock()
        self.event = threading.Event()

        self.points = {}
        self.params = {
            'lift': 4.0,
            'speed': 10.0,
            'offset': 6.0,
            'depth': -7.5
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

    #########################
    # Thread safe WAMP calls.
    #########################

    def publish(self, *args, **kwargs):
        if threading.get_ident() == self.id:
            super().publish(*args, **kwargs)
        else:
            self.loop.call_soon_threadsafe(partial(super().publish, *args, **kwargs))

    def call(self, *args, **kwargs):
        if threading.get_ident() == self.id:
            super().call(*args, **kwargs)
        else:
            self.loop.call_soon_threadsafe(partial(super().call, *args, **kwargs))

    ####################
    # Special functions.
    ####################

    def run(self, fn, *args, **kwargs):
        return asyncio.wrap_future(self.executor.submit(fn, *args, **kwargs))

    def initialize(self):
        self.agility = Agility(Android.arm)
        self.chatbot = Chatbot()
        self.image = Image()
        self.weather = Weather()

        self.initialized = True
        logger.info('Initialization complete.')

    def speak(self, message):
        logger.info('Speaking "{}"'.format(message))
        self.call('controller.speak', message)

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

        self.speak(message)

    def _draw(self, svg):
        self.speak('Executing draw.')

        landscape = (11.0 * 96, 8.5 * 96)
        portrait = (landscape[1], landscape[0])

        if type(svg) == str:
            svg = os.path.join(self.root, 'svg', svg).replace('\\', '/')

        drawing = Drawing(svg, portrait, center=True, resize=True, dx=20)

        angles, dts = self.agility.draw(drawing, self.params['speed'], self.params['offset'],
                                        self.params['depth'], self.params['lift'])
        self.event.clear()
        completed = self.agility.execute(angles, dts, event=self.event)

        if not completed:
            self.agility.zero()

    def _trace_image(self, q, index):
        index = Numeric.to_int(index)

        if index is None:
            return self.speak('I don\'t recognize that index.')

        url = self.image.get_all_url(q, index)

        if url is None:
            return self.speak('I am unable to find an image with those specifications.')

        logger.info('Image URL: {}.'.format(url))

        svg = self.image.get_image(url)
        self._draw(svg)

    @wamp.register('arm.trace_image')
    async def trace_image(self, q, index=0):
        if not self.initialized:
            return self.speak('Please wait. System is not initialized.')

        if self.work_lock.acquire(blocking=False):
            try:
                self.speak('Fetching image from Pixabay.')
                await self.run(self._trace_image, q, index)
            finally:
                self.work_lock.release()
        else:
            self.speak('I am currently busy.')

    def _draw_image(self, q, index):
        index = Numeric.to_int(index)

        if index is None:
            return self.speak('I don\'t recognize that index.')

        url = self.image.get_svg_url(q, index)

        if url is None:
            return self.speak('I am unable to find an image with those specifications.')

        logger.info('Image URL: {}.'.format(url))

        svg = self.image.get_image(url)
        self._draw(svg)

    @wamp.register('arm.draw_image')
    async def draw_image(self, q, index=0):
        if not self.initialized:
            return self.speak('Please wait. System is not initialized.')

        if self.work_lock.acquire(blocking=False):
            try:
                self.speak('Fetching image from Pixabay.')
                await self.run(self._draw_image, q, index)
            finally:
                self.work_lock.release()
        else:
            self.speak('I am currently busy.')

    def _draw_weather(self):
        svg = self.weather.get_now()
        self._draw(svg)

    @wamp.register('arm.draw_weather')
    async def draw_weather(self):
        if not self.initialized:
            return self.speak('Please wait. System is not initialized.')

        if self.work_lock.acquire(blocking=False):
            try:
                self.speak('Drawing the weather.')
                await self.run(self._draw_weather)
            finally:
                self.work_lock.release()
        else:
            self.speak('I am currently busy.')

    def _draw_forecast(self, value, units):
        value = Numeric.to_float(value)

        if value is None:
            return self.speak('I don\'t recognize that number.')

        if units not in ('minute', 'minutes', 'hour', 'hours', 'day', 'days', 'week', 'weeks'):
            return self.speak('I don\'t recognize that unit.')

        try:
            svg = self.weather.get_forecast(value, units)
        except Exception:
            return self.speak('Forecast out of range.')

        self._draw(svg)

    @wamp.register('arm.draw_forecast')
    async def draw_forecast(self, value, units):
        if not self.initialized:
            return self.speak('Please wait. System is not initialized.')

        if self.work_lock.acquire(blocking=False):
            try:
                self.speak('Drawing the forecast.')
                await self.run(self._draw_forecast, value, units)
            finally:
                self.work_lock.release()
        else:
            self.speak('I am currently busy.')

    def _home(self):
        self.agility.zero()

    @wamp.register('arm.home')
    async def home(self):
        if not self.initialized:
            return self.speak('Please wait. System is not initialized.')

        if self.work_lock.acquire(blocking=False):
            try:
                self.speak('Homing the arm.')
                await self.run(self._home)
            finally:
                self.work_lock.release()
        else:
            self.speak('I am currently busy.')

    @wamp.register('arm.stop')
    async def stop(self):
        if not self.work_lock.acquire(blocking=False):
            self.event.set()
            self.speak('Stopping.')
        else:
            self.work_lock.release()
            self.speak('I cannot stop doing nothing.')

    @wamp.register('arm.info')
    async def info(self):
        text = 'The current configuration is as follows. ' \
               'Linear velocity: {:.2f}. ' \
               'X-offset: {:.2f}. ' \
               'Lift height: {:.2f}.' \
               'Z-depth: {:.2f}.'.format(self.params['speed'], self.params['offset'],
                                     self.params['lift'], self.params['depth'])

        self.speak(text)

    def _set_parameter(self, param, value):
        value = Numeric.to_float(value)

        if value is None:
            return self.speak('I don\'t recognize that number.')

        if param not in self.params:
            return self.speak('I don\'t recognize that parameter.')

        self.params[param] = value

    @wamp.register('arm.set_parameter')
    async def set_parameter(self, param, value):
        self.speak('Setting parameter.')
        await self.run(self._set_parameter, param, value)

    def _relative_move(self, direction, delta):
        delta = Numeric.to_float(delta)

        if delta is None:
            return self.speak('I don\'t recognize that number.')

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
            return self.speak('I don\'t recognize that direction.')

        self.agility.move_relative((dx, dy, dz), pi, self.params['speed'])

    @wamp.register('arm.move_relative')
    async def relative_move(self, direction, delta):
        if not self.initialized:
            return self.speak('Please wait. System is not initialized.')

        if self.work_lock.acquire(blocking=False):
            try:
                self.speak('Moving.')
                await self.run(self._relative_move, direction, delta)
            finally:
                self.work_lock.release()
        else:
            self.speak('I am currently busy.')

    def _absolute_move(self, x, y, z):
        x = Numeric.to_float(x)
        y = Numeric.to_float(y)
        z = Numeric.to_float(z)

        if x is None or y is None or z is None:
            return self.speak('I don\'t recognize that coordinate.')

        self.agility.move_absolute((x, y, z), pi, self.params['speed'])

    @wamp.register('arm.move_absolute')
    async def absolute_move(self, x, y, z):
        if not self.initialized:
            return self.speak('Please wait. System is not initialized.')

        if self.work_lock.acquire(blocking=False):
            try:
                self.speak('Moving.')
                await self.run(self._absolute_move, x, y, z)
            finally:
                self.work_lock.release()
        else:
            self.speak('I am currently busy.')

    def _save_point(self, name):
        self.points[name] = self.agility.get_position()

    @wamp.register('arm.save_point')
    async def save_point(self, name):
        if not self.initialized:
            return self.speak('Please wait. System is not initialized.')

        self.speak('Saving current location.')
        await self.run(self._save_point, name)

    @wamp.register('arm.load_point')
    async def load_point(self, name):
        if name not in self.points:
            return self.speak('I am unable to find that point.')

        await self._absolute_move(*self.points[name])

    def _calibrate(self):
        pos = self.agility.get_position()
        self.params['depth'] = pos[2]

    @wamp.register('arm.calibrate')
    async def calibrate(self):
        self.speak('Calibrate depth using current position.')
        await self.run(self._calibrate)

    @wamp.register('arm.get_position')
    async def get_position(self):
        position = await self.run(self.agility.get_position)
        self.speak('The current position is {:.2f}, {:.2f}, {:.2f}.'.format(*position))

    @wamp.register('arm.chat')
    async def chat(self, input):
        response = await self.run(self.chatbot.ask, input)
        self.speak(response)


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
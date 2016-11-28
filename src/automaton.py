import asyncio
import ssl
from threading import Event
import os
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
        self.working = False
        self.event = Event()

        self.params = {
            'lift': 4,
            'speed': 10,
            'forward': 6
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

    def _draw_weather(self):
        svg = self.weather.get_icon()

        landscape = (11.0 * 96, 8.5 * 96)
        portrait = (landscape[1], landscape[0])
        drawing = Drawing(os.path.join(self.root, 'svg', svg).replace('\\', '/'), portrait,
                          center=True, resize=True, dx=5)

        angles, dts = self.agility.draw(drawing, self.params['speed'], self.params['forward'], -7.6, self.params['lift'])
        self.event.clear()
        self.agility.execute(angles, dts, event=self.event)

        self.working = False

    @wamp.register('arm.draw_weather')
    async def draw_weather(self):
        if self.initialized:
            if not self.working:
                self.working = True
                self.run(self._draw_weather)
                return 'Give me a moment.'
            else:
                return 'I am currently busy.'
        else:
            return 'System is not initialized.'

    @wamp.register('arm.stop')
    async def stop(self):
        self.event.set()
        return 'Stopping.'

    @wamp.register('arm.info')
    async def info(self):
        text = 'The current configuration is as follows. ' \
               'Linear velocity: {}. ' \
               'X-offset: {}. ' \
               'Lift height: {}.'.format(self.params['speed'], self.params['forward'], self.params['lift'])

        return text

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
from util import logger
from agility import Agility
from config import Android
from tkinter import Tk, Frame
from threading import Thread
import time


# Obtain configuration.
arm = Android.arm

# Set up agility.
agility = Agility(arm)
agility.zero()


# Define control class.
class Control:
    def __init__(self, arm, dx):
        self.dx = dx
        self.x, self.y, self.z = (10, 0, 5)

    def incr_x(self, event):
        self.x += self.dx

    def decr_x(self, event):
        self.x -= self.dx

    def incr_y(self, event):
        self.y += self.dx

    def decr_y(self, event):
        self.y -= self.dx

    def incr_z(self, event):
        self.z += self.dx

    def decr_z(self, event):
        self.z -= self.dx

    def get_target(self):
        return self.x, self.y, self.z


control = Control(arm, 0.1)


# Define threaded function.
def run():
    last_target = arm.zero

    while True:
        target = control.get_target()

        if target != last_target:
            logger.info('Moving arm to ({:.2f}, {:.2f}, {:.2f}).'.format(*target))
            agility.move_to(target, 5)
            last_target = target

        time.sleep(0.001)


# Spawn thread.
thread = Thread(target=run)
thread.start()

# Set up keyboard control.
main = Tk()
frame = Frame(main, width=100, height=100)

frame.bind('<w>', control.incr_x)
frame.bind('<s>', control.decr_x)
frame.bind('<a>', control.incr_y)
frame.bind('<d>', control.decr_y)
frame.bind('<r>', control.incr_z)
frame.bind('<f>', control.decr_z)

frame.pack()
frame.focus_set()
main.mainloop()


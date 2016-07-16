import time
import collections
import numpy as np
import pyglet
from vizbot.core import Agent
from vizbot.preprocess import Grayscale, Downsample, FrameSkip
from vizbot.utility import AttrDict, clamp


class Keyboard(Agent):

    def __init__(self, env, fps=30, sensitivity=0.3):
        # env = Grayscale(env)
        # env = Downsample(env, 2)
        # env = FrameSkip(env, 4)
        # env = Grayscale(env)
        super().__init__(env)
        self._viewer = Viewer(fps=fps)
        self._fps = fps
        self._time = None
        self._sensitivity = sensitivity

    def __del__(self):
        try:
            self._viewer.close()
        except AttributeError:
            pass

    def start(self):
        super().start()
        self._time = time.time()

    def perform(self, state):
        super().perform(state)
        self._viewer(state)
        action = self._noop()
        action = self._apply_keyboard(action, self._viewer.pressed_keys())
        delta = self._viewer.delta()
        delta = self._sensitivity * delta[0], self._sensitivity * delta[1]
        action = self._apply_mouse(action, delta)
        return action

    def _apply_keyboard(self, action, pressed):
        return action

    def _apply_mouse(self, action, delta):
        return action


class KeyboardDoom(Keyboard):

    KEYMAP = AttrDict(
        d='right', a='left', s='backward', w='forward', mouse_1='fire',
        enter='fire', shift='run', space='jump', ctrl='crouch',
        number_1='weapon_1', number_2='weapon_2', number_3='weapon_3',
        number_4='weapon_4', number_5='weapon_5', number_6='weapon_6',
        number_7='weapon_7')

    COMMANDS = (
        'fire turn_180 speed circle right left backward forward turn_left '
        'turn_right weapon_1 weapon_2 weapon_3 weapon_4 weapon_5 weapon_6 '
        'weapon_7 rotate_y rotate_x'.split())

    AttrDict(
        fire=0, run=1, right=2, left=3, bachward=4, forward=5, weapon_1=6,
        weapon_2=7, weapon_3=8, weapon_4=9, weapon_5=10, weapon_6=11,
        weapon_7=12, rotate_x=38, rotate_y=39)

    def __init__(self, env, fps=30, sensitivity=0.3, render_state=True):
        super().__init__(env, fps, sensitivity)
        self._render_state = render_state

    def _apply_keyboard(self, action, pressed):
        for key in pressed:
            command = self.KEYMAP.get(key)
            if command in self.COMMANDS:
                action[self.COMMANDS.index(command)] = 1
        return action

    def _apply_mouse(self, action, delta):
        action[self.COMMANDS.index('rotate_x')] = int(clamp(delta[0], -10, 10))
        action[self.COMMANDS.index('rotate_y')] = int(clamp(delta[1], -10, 10))
        return action


class Viewer:

    KEY_CODES = {
        100: 'd', 97: 'a', 115: 's', 119: 'w', 65293: 'enter', 65505:
        'shift', 65507: 'ctrl', 32: 'space'}

    def __init__(self, width=800, height=600, fps=30):
        self._dx = 0
        self._dy = 0
        self._window = pyglet.window.Window(width, height)
        self._window.on_mouse_motion = self._handle_mouse
        self._window.on_mouse_drag = self._handle_mouse
        self._window.on_close = self._handle_close
        self._window.set_exclusive_mouse()
        self._frame_time = 1 / fps
        self._frame_times = collections.deque(maxlen=100)
        self._time = None

    def __call__(self, image=None):
        self._update_fps()
        self._window.clear()
        self._window.switch_to()
        self._window.dispatch_events()
        if image is not None:
            image = self._load(image)
            x, y, width, height = self._center(image)
            image.blit(x, y, 0, width, height)
        self._window.flip()

    def close(self):
        try:
            self._window.close()
        except Exception:
            pass

    def pressed_keys(self):
        for number in range(1, 10):
            self.KEY_CODES[number + 48] = 'number_{}'.format(number)
        keys = set()
        for code in self._window.pressed_keys:
            if code in self.KEY_CODES:
                keys.add(self.KEY_CODES[code])
        for index, pressed in enumerate(self._window._mouse_buttons):
            if pressed:
                keys.add('mouse_{}'.format(index))
        return keys

    def delta(self):
        delta = self._dx, self._dy
        self._dx = 0
        self._dy = 0
        return delta

    def __del__(self):
        self.close()

    def _update_fps(self):
        last, self._time = self._time, time.time()
        if not last:
            return
        time.sleep(max(0, self._frame_time - (self._time - last)))
        self._frame_times.append(time.time() - last)
        fps = len(self._frame_times) / sum(self._frame_times)
        self._window.set_caption('FPS {}'.format(round(fps)))

    def _handle_mouse(self, x, y, dx, dy, *args):
        self._dx += dx
        self._dy += dy

    def _handle_close(self):
        self.close()
        raise KeyboardInterrupt

    def _load(self, image):
        image = image.astype(np.uint8)
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1).copy()
        image = pyglet.image.ImageData(
            image.shape[0], image.shape[1], 'RGB', image.tobytes(),
            pitch=image.shape[1] * -3)
        width, height = image.width, image.height
        image.width = height
        image.height = width
        return image

    def _center(self, image):
        aspect_width = self._window.width / image.width
        aspect_height = self._window.height / image.height
        if aspect_width > aspect_height:
            scale_width = aspect_height * image.width
            scale_height = aspect_height * image.height
        else:
            scale_width = aspect_width * image.width
            scale_height = aspect_width * image.height
        x = (self._window.width - scale_width) / 2
        y = (self._window.height - scale_height) / 2
        return x, y, scale_width, scale_height

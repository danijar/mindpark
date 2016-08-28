import time
import collections
import numpy as np
import pyglet
from vizbot.core import Algorithm, Policy, Sequential
from vizbot.step import (
    Grayscale, Subsample, Maximum, Skip, History, Normalize, ClampReward,
    Delta, EpsilonRandom)
from vizbot.utility import AttrDict, merge_dicts


class Keyboard(Algorithm, Policy):

    @classmethod
    def defaults(cls):
        frameskip = 3
        fps = 30 / frameskip
        sensitivity = 0.3
        viewer = Viewer
        return merge_dicts(super().defaults(), locals())

    def __init__(self, task, config):
        Algorithm.__init__(self, task, config)
        self._preprocess = self._create_preprocess()
        Policy.__init__(self, self._preprocess.above_task)
        self._viewer = self.config.viewer(fps=self.config.fps)
        self._time = None

    def __del__(self):
        try:
            self._viewer.close()
        except AttributeError:
            pass

    def begin_episode(self, episode, training):
        super().begin_episode(episode, training)
        self._time = time.time()

    def observe(self, observ):
        super().observe(observ)
        actions = self._preprocess.above_task.actions
        self._viewer(observ)
        action = np.zeros(actions.n)
        action = self._apply_keyboard(action, self._viewer.pressed_keys())
        delta = self._viewer.delta()
        sensitivity = self.config.sensitivity
        delta = sensitivity * delta[0], sensitivity * delta[1]
        action = self._apply_mouse(action, delta)
        action = action.argmax()
        return action

    def receive(self, reward, final):
        super().receive(reward, final)
        if reward:
            print('Receive reward', reward)

    @property
    def policy(self):
        policy = Sequential(self.task)
        policy.add(self._preprocess)
        policy.add(self)
        return policy

    def _create_preprocess(self):
        policy = Sequential(self.task)
        # Network preprocessing.
        # policy.add(Skip, self.config.frameskip)
        # policy.add(Maximum, 2)
        # policy.add(Grayscale)
        # # policy.add(Subsample, (2, 2))
        # # policy.add(Delta)
        # policy.add(History, 3)
        # policy.add(ClampReward)
        # policy.add(EpsilonRandom, from_=0.5, to=0.5, test=0.5)
        policy.add(Normalize)
        return policy

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

    COMMANDS = ((
        'fire right left backward forward turn_left turn_right').split())

    def _apply_keyboard(self, action, pressed):
        for key in pressed:
            command = self.KEYMAP.get(key)
            if command in self.COMMANDS:
                action[self.COMMANDS.index(command)] = 1
        return action

    def _apply_mouse(self, action, delta):
        if delta[0] > 0:
            action[self.COMMANDS.index('turn_left')] = 1
        elif delta[0] < 0:
            action[self.COMMANDS.index('turn_right')] = 1
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
        image = (image * 255).astype(np.uint8)
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

import time
import datetime
import os
import collections
import numpy as np
import gym
import pyglet
from doom_py import ScreenResolution
from vizbot.utility import AttrDict, ensure_directory, clamp


class InteractiveViewer:

    KEY_CODES = {
        100: 'd', 97: 'a', 115: 's', 119: 'w', 65293: 'enter', 65505:
        'shift', 65507: 'ctrl', 32: 'space'}

    def __init__(self, width=800, height=600):
        self._dx = 0
        self._dy = 0
        self._window = pyglet.window.Window(width, height)
        self._window.on_mouse_motion = self._handle_mouse
        self._window.on_mouse_drag = self._handle_mouse
        self._window.on_close = self._handle_close
        self._window.set_exclusive_mouse()
        self._fps = collections.deque(maxlen=100)
        self._last = None

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

    def mouse_delta(self):
        delta = self._dx, self._dy
        self._dx = 0
        self._dy = 0
        return delta

    def __del__(self):
        self.close()

    def _update_fps(self):
        self._fps.append(1 / (time.time() - self._last) if self._last else 0)
        fps = sum(self._fps) / len(self._fps)
        self._window.set_caption('FPS {}'.format(round(fps)))
        self._last = time.time()

    def _handle_mouse(self, x, y, dx, dy, *args):
        self._dx += dx
        self._dy += dy

    def _handle_close(self):
        self.close()
        raise KeyboardInterrupt

    def _load(self, image):
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



class KeyboardAgent:

    KEYMAP = AttrDict(
        d='right', a='left', s='backward', w='forward', mouse_1='fire',
        enter='fire', shift='run', space='jump', ctrl='crouch',
        number_1='weapon_1', number_2='weapon_2', number_3='weapon_3',
        number_4='weapon_4', number_5='weapon_5', number_6='weapon_6',
        number_7='weapon_7')

    ACTIONS = {}

    def __init__(self):
        self._viewer = InteractiveViewer()

    def __call__(self, state):
        self._viewer(state)
        action = np.zeros(43, dtype=int)
        action = self._apply_actions(action)
        return action

    def close(self):
        self._viewer.close()

    def _apply_actions(self, action):
        for key in self._viewer.pressed_keys():
            index = self.ACTIONS.get(self.KEYMAP.get(key))
            if index is not None:
                action[index] = 1
        return action


class DeathmatchKeyboardAgent(KeyboardAgent):

    ACTIONS = AttrDict(
        right=10, left=11, backward=12, forward=13, rotate_y=38, rotate_x=39,
        fire=0, switch=31, run=8, jump=2, crouch=3, weapon_1=21, weapon_2=22,
        weapon_3=23, weapon_4=24, weapon_5=25, weapon_6=26, weapon_7=27)

    def __init__(self, sensitivity=0.3):
        super().__init__()
        self._sensitivity = sensitivity

    def _apply_actions(self, action):
        action = super()._apply_actions(action)
        delta = self._viewer.mouse_delta()
        rotate_x = clamp(self._sensitivity * delta[0], -10, 10)
        rotate_y = clamp(self._sensitivity * delta[1], -10, 10)
        action[self.ACTIONS.rotate_x] = int(rotate_x)
        action[self.ACTIONS.rotate_y] = int(rotate_y)
        return action


def control_episode(env, agent, fps=30):
    state = env.reset()
    done = False
    states, actions, rewards = [], [], []
    while not done:
        start = time.time()
        action = agent(state)
        next_state, reward, done, _ = env.step(action)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        state = next_state
        duration = time.time() - start
        time.sleep(max(0, 1 / fps - duration))
    states = np.array(states, dtype=float)
    actions = np.array(actions, dtype=float)
    rewards = np.array(rewards, dtype=float)
    return states, actions, rewards


def store_episode(directory, env, states, actions, rewards):
    ensure_directory(directory)
    name = env.spec.id
    timestamp = datetime.datetime.utcnow().isoformat()
    filename = '{}-{}-{}.npz'.format(name, timestamp, rewards.sum())
    filepath = os.path.join(directory, filename)
    size = (states.nbytes + actions.nbytes + rewards.nbytes) / 1024 / 1024
    message = 'Store {} transitions of raw size {}mb to {}'
    print(message.format(len(states), round(size), filepath))
    arrays = {'states': states, 'actions': actions, 'rewards': rewards}
    np.savez_compressed(filepath, **arrays)
    print('Thank you :)')


def main():
    directory = os.path.join(os.path.dirname(__file__), 'recordings')
    directory = os.path.abspath(directory)
    env = gym.make('DoomDeathmatch-v0')
    env.configure(screen_resolution=ScreenResolution.RES_160X120)
    agent = DeathmatchKeyboardAgent()
    states, actions, rewards = control_episode(env, agent)
    agent.close()
    store_episode(directory, env, states, actions, rewards)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass

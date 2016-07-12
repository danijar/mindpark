import time
import datetime
import os
import numpy as np
import gym
import pyglet
from doom_py import ScreenResolution
from vizbot.utility import AttrDict, ensure_directory, clamp


class ImageViewer(object):

    KEY_CODES = {
        100: 'd', 97: 'a', 115: 's', 119: 'w', 65293: 'enter', 65505:
        'shift', 65507: 'ctrl', 32: 'space'}

    def __init__(self, size, zoom=1):
        self._size = size
        self._zoom = zoom
        self._window = pyglet.window.Window(
            width=size[0] * zoom, height=size[1] * zoom)
        self._window.set_vsync(True)
        self._window.set_exclusive_mouse()

    def __call__(self, image=None):
        self._window.clear()
        self._window.switch_to()
        self._window.dispatch_events()
        if image is not None:
            assert image.shape == (self._size[1], self._size[0], 3)
            sprite = self._load_sprite(image)
            sprite.draw()
        self._window.flip()

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
        center = self._size[0] / 2, self._size[1] / 2
        mouse = self._window._mouse_x, self._window._mouse_y
        mouse = mouse[0] / self._zoom, mouse[1] / self._zoom
        delta = int(mouse[0] - center[0]), int(mouse[1] - center[1])
        return delta

    def __del__(self):
        self._window.close()

    def _load_sprite(self, image):
        image = pyglet.image.ImageData(
            self._size[0], self._size[1], 'RGB', image.tobytes(),
            pitch=self._size[0] * -3)
        sprite = pyglet.sprite.Sprite(image, 0, 0)
        sprite.scale = self._zoom
        return sprite



class KeyboardAgent:

    KEYMAP = AttrDict(
        d='right', a='left', s='backward', w='forward', mouse_1='fire',
        enter='fire', shift='run', space='jump', ctrl='crouch',
        number_1='weapon_1', number_2='weapon_2', number_3='weapon_3',
        number_4='weapon_4', number_5='weapon_5', number_6='weapon_6',
        number_7='weapon_7')

    ACTIONS = AttrDict(
        right=10, left=11, backward=12, forward=13, rotate_y=38, rotate_x=39,
        fire=0, switch=31, run=8, jump=2, crouch=3, weapon_1=21, weapon_2=22,
        weapon_3=23, weapon_4=24, weapon_5=25, weapon_6=26, weapon_7=27)

    def __init__(self, env, sensitivity=(0.5, 0.3), size=(160, 120), zoom=3):
        self._env = env
        self._sensitivity = sensitivity
        self._size = size
        self._zoom = zoom
        self._viewer = ImageViewer(size, zoom)

    def __call__(self, state):
        self._viewer(state)
        action = np.zeros(self._env.action_space.shape, dtype=int)
        delta = self._viewer.mouse_delta()
        rotate_x = clamp(self._sensitivity[0] * delta[0], -10, 10)
        rotate_y = clamp(self._sensitivity[1] * delta[1], -10, 10)
        action[self.ACTIONS.rotate_x] = int(rotate_x)
        action[self.ACTIONS.rotate_y] = int(rotate_y)
        for key in self._viewer.pressed_keys():
            index = self.ACTIONS.get(self.KEYMAP.get(key))
            if index is not None:
                action[index] = 1
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
        time.sleep(max(0, time.time() - start))
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


def main():
    directory = os.path.join(os.path.dirname(__file__), 'recordings')
    directory = os.path.abspath(directory)
    env = gym.make('DoomDeathmatch-v0')
    env.configure(screen_resolution=ScreenResolution.RES_160X120)
    agent = KeyboardAgent(env)
    states, actions, rewards = control_episode(env, agent)
    # store_episode(directory, env, states, actions, rewards)


if __name__ == '__main__':
    main()

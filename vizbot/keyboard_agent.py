import time
import sys
import errno
import datetime
import os
import numpy as np
import gym
from doom_py import ScreenResolution, Mode


class AttrDict(dict):

    def __getattr__(self, key):
        if key not in self:
            raise AttributeError
        return self[key]

    def __setattr__(self, key, value):
        if key not in self:
            raise AttributeError
        self[key] = value


def clamp(value, min_, max_):
    return max(min_, min(value, max_))


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

    def __init__(self, env, sensitivity=(0.5, 0.3)):
        env.reset()
        env.render()
        self._space = env.action_space
        self._window = env.viewer.window
        self._window.set_exclusive_mouse()
        self._sensitivity = sensitivity
        self._enable_key_events()

    def __call__(self, state):
        delta = self._mouse_delta()
        action = np.zeros(self._space.shape, dtype=int)
        rotate_x = clamp(self._sensitivity[0] * delta[0], -10, 10)
        rotate_y = clamp(self._sensitivity[1] * delta[1], -10, 10)
        action[self.ACTIONS.rotate_x] = int(rotate_x)
        action[self.ACTIONS.rotate_y] = int(rotate_y)
        for key in self.pressed:
            index = self.ACTIONS.get(self.KEYMAP.get(key))
            if index is not None:
                action[index] = 1
        return action

    def reward(self, reward):
        pass

    @property
    def pressed(self):
        return self._pressed_mouse | self._pressed_keyboard

    @property
    def _pressed_mouse(self):
        keys = set()
        for index, pressed in enumerate(self._window._mouse_buttons):
            if pressed:
                keys.add('mouse_{}'.format(index))
        return keys

    @property
    def _pressed_keyboard(self):
        key_codes = {
            100: 'd', 97: 'a', 115: 's', 119: 'w', 65293: 'enter', 65505:
            'shift', 65507: 'ctrl', 32: 'space'}
        for number in range(1, 10):
            key_codes[number + 48] = 'number_{}'.format(number)
        keys = set()
        for code in self._window.pressed_keys:
            if code in key_codes:
                keys.add(key_codes[code])
            else:
                # print('Unknown key code', code)
                pass
        return keys

    def _enable_key_events(self):
        self._window.on_key_press = lambda x, y: None
        self._window.on_key_release = lambda x, y: None

    def _mouse_delta(self):
        # We don't set back the mouse to the center of the window since that is
        # done by VizDoom automatically.
        center = self._window.width // 2, self._window.height // 2
        mouse = self._window._mouse_x, self._window._mouse_y
        delta = mouse[0] - center[0], mouse[1] - center[1]
        return delta


def ensure_directory(directory):
    directory = os.path.expanduser(directory)
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e


def control_episode(env, agent, fps=30):
    state = env.reset()
    done = False
    states, actions, rewards = [], [], []
    while not done:
        start = time.time()
        env.render()
        action = agent(state)
        next_state, reward, done, _ = env.step(action)
        agent.reward(reward)
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

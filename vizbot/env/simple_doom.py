import numpy as np
import gym
from gym.core import Env
from gym.spaces import Discrete
from doom_py import ScreenResolution
from vizbot.utility import lazy_property


class SimpleDoom(Env):

    AVAILABLE_ACTIONS = sorted([
        0,                           # Fire
        # 4,                           # Turn 180
        # 8,                           # Speed
        # 9,                           # Move sideways in a circle
        10, 11, 12, 13,              # Move
        14, 15,                      # Turn
        # 21, 22, 23, 24, 25, 26, 27,  # Select weapon
        # 38, 39,                      # Mouse look
    ])

    def __init__(self, env):
        assert env.startswith('Doom')
        super().__init__()
        self._env = gym.make(env)
        self._env.configure(screen_resolution=ScreenResolution.RES_160X120)

    @property
    def observation_space(self):
        return self._env.observation_space

    @lazy_property
    def action_space(self):
        return Discrete(len(self.AVAILABLE_ACTIONS))

    @property
    def metadata(self):
        return self._env.metadata

    @property
    def monitor(self):
        return self._env.monitor

    def _step(self, action):
        box = np.zeros(self._env.action_space.num_rows)
        box[self.AVAILABLE_ACTIONS[action]] = 1
        state, reward, done, info = self._env._step(box)
        return state, reward, done, info

    def _close(self):
        return self._env._close()

    def _seed(self, seed=None):
        return self._env._seed(seed)

    def _reset(self):
        return self._env._reset()

    def _render(self, mode='human', close=False):
        return self._env._render(mode, close)

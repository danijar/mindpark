import numpy as np
import gym
from gym.core import Env
from gym.spaces import HighLow
from doom_py import ScreenResolution


class SimpleDoom(Env):

    AVAILABLE_ACTIONS = sorted([
        0,                           # Fire
        4,                           # Turn 180
        8,                           # Speed
        9,                           # Move sideways in a circle
        10, 11, 12, 13,              # Move
        14, 15,                      # Turn
        21, 22, 23, 24, 25, 26, 27,  # Select weapon
        38, 39,                      # Mouse look
    ])

    def __init__(self, env='DoomDeathmatch-v0'):
        assert env.startswith('Doom')
        super().__init__()
        self._env = gym.make(env)
        self._env.configure(screen_resolution=ScreenResolution.RES_160X120)

    @property
    def action_space(self):
        matrix = self._env.action_space.matrix[self.AVAILABLE_ACTIONS]
        return HighLow(matrix)

    def _step(self, action):
        assert len(action) == len(self.AVAILABLE_ACTIONS)
        full_action = np.zeros(self._env.action_space.num_rows)
        for index, value in zip(self.AVAILABLE_ACTIONS, action):
            full_action[index] = value
        # Work around drop weapon bug in parent class.
        full_action[33] = 0
        return self._env._step(full_action)

    def __getattr__(self, name):
        print('Access', name)
        return getattr(self._env, name)

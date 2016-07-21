import numpy as np
import gym
from gym.core import Env
from gym.spaces import HighLow
from vizbot.utility import lazy_property


class SimpleAtari(Env):


    def __init__(self, env):
        super().__init__()
        self._env = gym.make(env)
        # self._env.configure()

    @property
    def observation_space(self):
        return self._env.observation_space

    @lazy_property
    def action_space(self):
        matrix = np.zeros((self._env.action_space.n, 3))
        matrix[:, 1] = 1
        return HighLow(matrix)

    @property
    def metadata(self):
        return self._env.metadata

    @property
    def monitor(self):
        return self._env.monitor

    def _step(self, action):
        # assert len(action) == len(self.AVAILABLE_ACTIONS)
        # full_action = np.zeros(self._env.action_space.num_rows)
        # for index, value in zip(self.AVAILABLE_ACTIONS, action):
        #     full_action[index] = value
        # Work around drop weapon bug in parent class.
        # full_action[33] = 0
        choice = np.array(action).argmax()
        state, reward, done, info = self._env._step(choice)
        # if done:
        #     reward = -1
        return state, reward, done, info

    def _close(self):
        return self._env._close()

    def _seed(self, seed=None):
        return self._env._seed(seed)

    def _reset(self):
        return self._env._reset()

    def _render(self, mode='human', close=False):
        return self._env._render(mode, close)

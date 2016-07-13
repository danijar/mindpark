import numpy as np
from vizbot.core import Agent


class Noop(Agent):

    def __init__(self, env):
        self._env = env

    def __call__(self, reward, state):
        return np.zeros(self._env.action_space.shape)

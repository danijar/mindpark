import numpy as np
from vizbot.core import Preprocess


class NormalizeReward(Preprocess):

    def __init__(self, env):
        super().__init__(env)

    @property
    def states(self):
        return self._env.states

    @property
    def actions(self):
        return self._env.actions

    def reset(self):
        state = self._env.reset()
        return state

    def step(self, action):
        state, reward = self._env.step(action)
        reward = np.sign(reward)
        return state, reward

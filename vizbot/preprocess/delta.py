import numpy as np
from gym.spaces import Box
from vizbot.core import Preprocess


class Delta(Preprocess):

    def __init__(self, env):
        super().__init__(env)
        low, high = self._env.states.low, self._env.states.high
        self._low, self._high = low.flatten()[0], high.flatten()[0]
        assert (low == self._low).all()
        assert (high == self._high).all()
        self._zeros = np.zeros(self._env.states.shape)
        self._previous = None

    @property
    def states(self):
        low, high = self._low - self._high, self._high - self._low
        return Box(low, high, self._env.states.shape)

    @property
    def actions(self):
        return self._env.actions

    def reset(self):
        self._previous = self._env.reset()
        return self._zeros

    def step(self, action):
        state, reward = self._env.step(action)
        delta = state - self._previous
        self._previous = state
        return delta, reward

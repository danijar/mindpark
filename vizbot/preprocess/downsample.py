import numpy as np
from gym.spaces import Box
from vizbot.core import Preprocess


class Downsample(Preprocess):

    def __init__(self, env, factor=2):
        if not self._is_power_of_two(factor):
            raise ValueError('factor must be a power of two')
        super().__init__(env)
        self._factor = factor
        low, high = self._env.states.low, self._env.states.high
        self._low, self._high = low.flatten()[0], high.flatten()[0]
        assert (low == self._low).all()
        assert (high == self._high).all()

    @property
    def states(self):
        states = self._env.states
        shape = list(states.shape)
        shape[0] //= self._factor
        shape[1] //= self._factor
        return Box(self._low, self._high, shape)

    @property
    def actions(self):
        return self._env.actions

    def reset(self):
        state = self._env.reset()
        state = self._apply(state)
        return state

    def step(self, action):
        state, reward = self._env.step(action)
        state = self._apply(state)
        return state, reward

    def _apply(self, state):
        (width, height), factor = state.shape[:2], self._factor
        shape = width // factor, factor, height // factor, factor, -1
        state = state.reshape(shape).mean(3).mean(1)
        state = np.squeeze(state)
        return state

    @staticmethod
    def _is_power_of_two(value):
        if not isinstance(value, int) or value < 1:
            return False
        return value & (value - 1) == 0

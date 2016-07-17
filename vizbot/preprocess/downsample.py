import numpy as np
from gym.spaces import Box
from vizbot.core import Preprocess


class Downsample(Preprocess):

    def __init__(self, env, factor=2):
        if not self._is_power_of_two(factor):
            raise ValueError('factor must be a power of two')
        super().__init__(env)
        self.__factor = factor

    @property
    def states(self):
        shape = list(self._env.states.shape)
        shape[0] //= self.__factor
        shape[1] //= self.__factor
        return Box(0, 255, shape)

    @property
    def actions(self):
        return self._env.actions

    def perform(self, state):
        super().step()
        super().perform(state)
        (width, height), factor = state.shape[:2], self.__factor
        shape = width // factor, factor, height // factor, factor, -1
        state = state.reshape(shape).mean(3).mean(1)
        state = np.squeeze(state).astype(np.uint8)
        return self._agent.perform(state)

    def feedback(self, action, reward):
        super().feedback(action, reward)
        self._agent.feedback(action, reward)

    @staticmethod
    def _is_power_of_two(value):
        if not isinstance(value, int) or value < 1:
            return False
        return value & (value - 1) == 0

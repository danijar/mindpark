import numpy as np
from gym.spaces import Box
from vizbot.core import Preprocess


class Downsample(Preprocess):

    def __init__(self, env, factor=2):
        super().__init__(env)
        self.__factor = factor

    @property
    def states(self):
        shape = list(self._env.states.shape)
        shape[0] //= self._factor
        shape[1] //= self._factor
        return Box(0, 255, shape)

    @property
    def actions(self):
        return self._env.actions

    def perform(self, state):
        super().perform(state)
        (width, height), factor = state.shape[:2], self.__factor
        shape = width // factor, factor, height // factor, factor, -1
        state = np.squeeze(state.reshape(shape).mean(3).mean(1)).astype(np.uint8)
        return self._agent.perform(state)

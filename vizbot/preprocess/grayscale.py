import numpy as np
from gym.spaces import Box
from vizbot.core import Preprocess


class Grayscale(Preprocess):

    @property
    def states(self):
        shape = self._env.states.shape[: -1]
        return Box(0, 255, shape)

    @property
    def actions(self):
        return self._env.actions

    def perform(self, state):
        super().perform(state)
        state = state.mean(-1).astype(np.uint8)
        return self._agent.perform(state)

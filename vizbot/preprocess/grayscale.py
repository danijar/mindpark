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

    def reset(self):
        state = self._env.reset()
        state = self._apply(state)
        return state

    def step(self, action):
        state, reward = self._env.step(action)
        state = self._apply(state)
        return state, reward

    @staticmethod
    def _apply(state):
        return state.mean(-1).astype(np.uint8)

import collections
import numpy as np
from gym.spaces import Box
from vizbot.core import Preprocess


class FrameWindow(Preprocess):

    def __init__(self, env, width):
        super().__init__(env)
        self._width = width
        self._frames = np.empty((self._amount,) + self._env.states.shape)
        self._noop = np.zeros(self._env.actions.shape)
        self._offset = None

    @property
    def states(self):
        shape = self._env.states.shape + (self.__amount,)
        return Box(0, 255, shape)

    @property
    def actions(self):
        return self._env.actions

    def reset(self):
        self._offset = 0
        self._push_frame(self._env.reset())
        for _ in range(1, self._width):
            state, _ = self._env.step(self._noop)
            self._push_frame(state)
        return self._window()

    def step(self, action):
        state, reward = self._env.step(action)
        self._push_frame(state)
        return self._window(), reward

    def _push_frame(self, state):
        self._frames[self._offset] = state
        self._offset += 1

    def _window(self):
        order = [(x + self._offset) % self._width for x in range(self._width)]
        state = np.moveaxis(self._frames[order], 0, -1)
        return state

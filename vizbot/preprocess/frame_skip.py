import numpy as np
from gym.spaces import Box
from vizbot.core import Preprocess


class FrameSkip(Preprocess):

    def __init__(self, env, amount):
        super().__init__(env)
        self._amount = amount
        self._frames = np.empty((self._amount,) + self._env.states.shape)
        self._noop = np.zeros(self._env.actions.shape)

    @property
    def states(self):
        shape = self._env.states.shape + (self.__amount,)
        return Box(0, 255, shape)

    @property
    def actions(self):
        return self._env.actions

    def reset(self):
        self._frames[0] = self._env.reset()
        for index in range(1, self._amount):
            self._frames[index] = self._env.step(self._noop)
        return np.moveaxis(self._frames, 0, -1)

    def step(self, action):
        for index in range(self._amount):
            self._frames[index] = self._env.step(action)
        return np.moveaxis(self._frames, 0, -1)

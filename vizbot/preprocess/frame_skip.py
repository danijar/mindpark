import numpy as np
from gym.spaces import Box
from vizbot.core import Preprocess


class FrameSkip(Preprocess):

    def __init__(self, env, amount):
        super().__init__(env)
        self.__amount = amount
        self.__action = None
        self.__frames = None
        self.__index = None

    @property
    def states(self):
        shape = self._env.states.shape + (self.__amount,)
        return Box(0, 255, shape)

    @property
    def actions(self):
        return self._env.actions

    def start(self):
        super().start()
        self.__action = self._agent._noop()
        self.__frames = np.empty((self.__amount,) + self._env.states.shape)
        self.__index = 0

    def perform(self, state):
        super().perform(state)
        self.__frames[self.__index % self.__amount] = state
        self.__index += 1
        if self.__index and self.__index % self.__amount:
            frames = np.moveaxis(self.__frames, 0, -1)
            self.__action = self._agent.perform(frames)
        return self.__action

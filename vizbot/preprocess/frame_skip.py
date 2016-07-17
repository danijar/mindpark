import numpy as np
from gym.spaces import Box
from vizbot.core import Agent, Preprocess


class FrameSkip(Preprocess):

    def __init__(self, env, amount):
        super().__init__(env)
        self.__amount = amount
        self.__action = None
        self.__frames = None
        self.__timestep = None

    @property
    def states(self):
        shape = self._env.states.shape + (self.__amount,)
        return Box(0, 255, shape)

    @property
    def actions(self):
        return self._env.actions

    def start(self):
        super().start()
        self.__action = self._noop()
        self.__frames = np.empty((self.__amount,) + self._env.states.shape)
        self.__timestep = 0

    def perform(self, state):
        super().perform(state)
        self.__frames[self.__timestep % self.__amount] = state
        if self._batch_collected():
            super().step()
            frames = np.moveaxis(self.__frames, 0, -1)
            self.__action = self._agent.perform(frames)
        self.__timestep += 1
        return self.__action

    def feedback(self, action, reward):
        super().feedback(action, reward)
        if self._batch_collected():
            self._agent.feedback(action, reward)

    def _batch_collected(self):
        return self.__timestep and not self.__timestep % self.__amount

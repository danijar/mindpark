import numpy as np


class Agent:

    def __init__(self, env):
        self._env = env
        self._env.register(self)
        self.__state = None
        self.__action = None
        self.__reward = None

    def begin(self):
        self.__state = None
        self.__action = None
        self.__reward = None

    def perform(self, state):
        self._env.episode += 1
        if self._env.episode:
            self._experience(self.__state, self.__action, self.__reward, state)
        self.__state = state

    def feedback(self, action, reward):
        self.__action = action
        self.__reward = reward

    def end(self):
        if self._env.episode:
            self._experience(self.__state, self.__action, self.__reward, None)

    def _experience(self, state, action, reward, successor):
        pass

    def _noop(self):
        return np.zeros(self._env.actions.shape)

    def _decay(self, start, end, over):
        progress = min(self._env.episode, over) / over
        return (1 - progress) * start + progress * end

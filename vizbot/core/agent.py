import numpy as np


class Agent:

    def __init__(self, states, actions, seed=0):
        self._states = states
        self._actions = actions
        self._episode = -1
        self._random = np.random.RandomState(seed)
        self.__state = None
        self.__action = None
        self.__reward = None

    def begin(self):
        self.__state = None
        self.__action = None
        self.__reward = None

    def step(self, state):
        self._episode += 1
        if self._episode:
            self._experience(self.__state, self.__action, self.__reward, state)
        self.__state = state

    def feedback(self, action, reward):
        self.__action = action
        self.__reward = reward

    def end(self):
        if self._episode:
            self._experience(self.__state, self.__action, self.__reward, None)

    def _experience(self, state, action, reward, successor):
        pass

    def _noop(self):
        return np.zeros(self._actions.shape)

    def _decay(self, start, end, over):
        progress = min(self._episode, over) / over
        return (1 - progress) * start + progress * end

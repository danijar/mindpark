import numpy as np


class Agent:

    def __init__(self, env):
        self._env = env
        self._env.register(self)
        self._random = np.random.RandomState(seed=0)
        self.__timestep = None
        self.__state = None
        self.__action = None
        self.__reward = None

    def start(self):
        self.__timestep = 0
        self.__state = None
        self.__action = None
        self.__reward = None

    def perform(self, state):
        if self.__timestep > 0:
            self._experience(self.__state, self.__action, self.__reward, state)
        self.__timestep += 1
        self.__state = state

    def feedback(self, action, reward):
        assert action is not None
        assert reward is not None
        self.__action = action
        self.__reward = reward

    def stop(self):
        if self.__timestep > 0:
            self._experience(self.__state, self.__action, self.__reward, None)

    def _experience(self, state, action, reward, successor):
        assert state is not None
        assert action is not None
        assert reward is not None

    def _noop(self):
        return np.zeros(self._env.actions.shape)

    def _decay(self, start, end, over):
        progress = min(self._env.timestep, over) / over
        return (1 - progress) * start + progress * end

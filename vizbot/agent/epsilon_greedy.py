import numpy as np
from vizbot.core import Agent


class EpsilonGreedy(Agent):

    def __init__(self, env, start, stop, over):
        super().__init__(env)
        self.__start = start
        self.__stop = stop
        self.__over = over

    def perform(self, state):
        super().perform(state)
        epsilon = self._decay(self.__start, self.__stop, self.__over)
        if self._random.rand() < epsilon:
            # TODO: Compare performance to single actions:
            # action = self._noop()
            # action[self._random.choice(self._env.actions.shape)] = 1
            # return action
            return np.array(self._env.actions.sample())
        return self._perform(state)

    def _perform(self, state):
        raise NotImplementedError

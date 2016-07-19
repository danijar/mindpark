import numpy as np
from vizbot.core import Agent


class EpsilonGreedy(Agent):

    def __init__(self, trainer, start, stop, over):
        super().__init__(trainer)
        self._start = start
        self._stop = stop
        self._over = over
        self._was_greedy = None

    def step(self, state):
        epsilon = self._decay(self._start, self._stop, self._over)
        if self._random.rand() < epsilon:
            self._was_greedy = True
            return np.array(self.actions.sample())
        self._was_greedy = False
        return self._step(state)

    def _step(self, state):
        raise NotImplementedError

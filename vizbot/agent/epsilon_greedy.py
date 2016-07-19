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
        if not self._trainer.timestep % 1000:
            print('Epsilon', round(epsilon, 4))
        if self._random.rand() < epsilon:
            self._was_greedy = True
            # TODO: Compare performance to single actions:
            # action = self._noop()
            # action[self._random.choice(self._env.actions.shape)] = 1
            # return action
            return np.array(self.actions.sample())
        self._was_greedy = False
        return self._step(state)

    def _step(self, state):
        raise NotImplementedError

import numpy as np
from vizbot.core import Agent
from vizbot.utility import Decay, merge_dicts


class EpsilonGreedy(Agent):

    @classmethod
    def defaults(cls):
        epsilon_from = 1.0
        epsilon_to = 0.1
        epsilon_duration = 5e5
        return merge_dicts(super().defaults(), locals())

    def __init__(self, trainer, config):
        super().__init__(trainer, config)
        self._was_greedy = None
        self._epsilon = Decay(
            config.epsilon_from, config.epsilon_to, config.epsilon_duration)

    def step(self, state):
        if self._random.rand() < self._epsilon(self.timestep):
            self._was_greedy = True
            return self._env.sample()
        self._was_greedy = False
        return self._step(state)

    def _step(self, state):
        raise NotImplementedError

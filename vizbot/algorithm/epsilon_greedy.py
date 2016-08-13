import numpy as np
from vizbot.core import Agent
from vizbot.utility import Decay, merge_dicts


class EpsilonGreedy(Agent):

    @classmethod
    def defaults(cls):
        epsilon_after = 1e5
        epsilon_from = 1.0
        epsilon_to = 0.1
        epsilon_duration = 5e5
        test_epsilon = 0.05
        return merge_dicts(super().defaults(), locals())

    def __init__(self, trainer, config):
        super().__init__(trainer, config)
        self._was_greedy = None
        self.config.epsilon_after = int(float(self.config.epsilon_after))
        self._epsilon = Decay(
            config.epsilon_from, config.epsilon_to, config.epsilon_duration)

    def step(self, state):
        super().step(state)
        if self.training:
            timestep = max(0, self.timestep - self.config.epsilon_after)
            epsilon = self._epsilon(timestep)
        else:
            epsilon = self.config.test_epsilon
        if self._random.rand() < epsilon:
            self._was_greedy = True
            return self.actions.sample()
        self._was_greedy = False
        return self._step(state)

    def _step(self, state):
        raise NotImplementedError

import numpy as np
from vizbot.core import Agent


class Noop(Agent):

    def step(self, state):
        super().step(state)
        return np.zeros(self._env.action_space.shape)

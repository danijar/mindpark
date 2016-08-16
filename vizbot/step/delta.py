import numpy as np
from gym.spaces import Box
from vizbot.step.identity import Identity


class Delta(Identity):

    def __init__(self, interface):
        super().__init__(interface)
        self._last = None
        self._empty = np.zeros(self.observations.shape)

    @property
    def interface(self):
        low = self.observations.low - self.observations.high
        high = self.observations.high - self.observations.low
        return Box(low, high), self.actions

    def begin_episode(self, training):
        super().begin_episode(training)
        self._last = None

    def perform(self, observation):
        super().perform(observation)
        if self._last is None:
            delta = self._empty
        else:
            delta = observation - self._last
        self._last = observation
        return self.above.observe(delta)

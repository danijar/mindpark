import numpy as np
from gym.spaces import Box
from vizbot.core import Policy


class Delta(Policy):

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

    def step(self, observation):
        super().step(observation)
        if self._last is None:
            delta = self._empty
        else:
            delta = observation - self._last
        self._last = observation
        return self.above.step(delta)

    def experience(self, *transition):
        super().experience(*transition)
        self.above.experience(*transition)

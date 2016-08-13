import numpy as np
from gym.spaces import Box
from vizbot.core import Policy


class History(Policy):

    def __init__(self, interface, amount=4):
        super().__init__(interface)
        self._amount = amount
        self._buffer = np.empty((self._amount,) + self.observations.shape)
        self._offset = None

    def begin_episode(self, training):
        super().begin_episode(training)
        self._offset = 0

    @property
    def interface(self):
        low = self._repeat(self.observations.low)
        high = self._repeat(self.observations.high)
        return Box(low, high), self.actions

    def step(self, observation):
        super().step(observation)
        self._push(observation)
        return self.above.step(self._history())

    def experience(self, *transition):
        super().experience(*transition)
        self.above.experience(*transition)

    def _push(self, observation):
        self._buffer[self._offset % self._amount] = observation
        self._offset += 1

    def _history(self):
        last = self._offset - self._amount
        order = [max(0, last + x) % self._amount for x in range(self._amount)]
        assert len(order) == self._amount
        return np.moveaxis(self._buffer[order], 0, -1)

    def _repeat(self, limit):
        return np.ones(limit.shape + (self._amount,)) * limit[..., np.newaxis]

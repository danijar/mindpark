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

    def observe(self, observation):
        super().observe(observation)
        self._push(observation)
        observation = self._history()
        return self.above.observe(observation)

    def receive(self, reward, final):
        super().receive(reward, final)
        self.above.receive(reward, final)

    def _push(self, observation):
        self._buffer[self._offset % self._amount] = observation
        self._offset += 1

    def _history(self):
        last = self._offset - self._amount
        order = [max(0, last + x) % self._amount for x in range(self._amount)]
        return np.moveaxis(self._buffer[order], 0, -1)

    def _repeat(self, array):
        return np.ones(array.shape + (self._amount,)) * array[..., np.newaxis]

import numpy as np
from gym.spaces import Box
from vizbot.core import Partial


class History(Partial):

    def __init__(self, interface, amount=4):
        super().__init__(interface)
        self._amount = amount
        self._buffer = np.empty((self._amount,) + self.task.observs.shape)
        self._offset = None

    @property
    def above_observs(self):
        low = self._repeat(self.task.observs.low)
        high = self._repeat(self.task.observs.high)
        return Box(low, high)

    @property
    def above_actions(self):
        return self.task.actions

    def begin_episode(self, episode, training):
        super().begin_episode(episode, training)
        self._offset = 0

    def observe(self, observ):
        super().observe(observ)
        self._push(observ)
        observ = self._history()
        return self.above.observe(observ)

    def receive(self, reward, final):
        super().receive(reward, final)
        self.above.receive(reward, final)

    def _push(self, observ):
        self._buffer[self._offset % self._amount] = observ
        self._offset += 1

    def _history(self):
        last = self._offset - self._amount
        order = [max(0, last + x) % self._amount for x in range(self._amount)]
        return np.moveaxis(self._buffer[order], 0, -1)

    def _repeat(self, array):
        return np.ones(array.shape + (self._amount,)) * array[..., np.newaxis]

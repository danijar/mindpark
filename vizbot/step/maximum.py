import numpy as np
from vizbot.core import Partial


class Maximum(Partial):

    """
    Running maximum over the specified number of previous the observations.
    """

    def __init__(self, task, amount=2):
        super().__init__(task)
        self._amount = amount
        self._buffer = np.empty((self._amount,) + self.task.observs.shape)
        self._offset = None

    @property
    def above_observs(self):
        return self.task.observs

    @property
    def above_actions(self):
        return self.task.actions

    def begin_episode(self, episode, training):
        super().begin_episode(episode, training)
        self._offset = 0

    def observe(self, observ):
        super().observe(observ)
        self._push(observ)
        observ = self._buffer[:min(self._offset, self._amount)].max(0)
        return self.above.observe(observ)

    def receive(self, reward, final):
        super().receive(reward, final)
        self.above.receive(reward, final)

    def _push(self, observ):
        self._buffer[self._offset % self._amount] = observ
        self._offset += 1

    def _repeat(self, array):
        return np.ones((self._amount,) + array.shape) * array[np.newaxis, ...]

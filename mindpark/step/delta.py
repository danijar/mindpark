import numpy as np
from gym.spaces import Box
from mindpark.core import Partial


class Delta(Partial):

    def __init__(self, task):
        super().__init__(task)
        self._last = None
        self._empty = np.zeros(self.task.observs.shape)

    @property
    def above_observs(self):
        low = self.task.observs.low - self.task.observs.high
        high = self.task.observs.high - self.task.observs.low
        return Box(low, high)

    @property
    def above_actions(self):
        return self.task.actions

    def begin_episode(self, episode, training):
        super().begin_episode(episode, training)
        self._last = None

    def observe(self, observ):
        super().observe(observ)
        if self._last is None:
            delta = self._empty
        else:
            delta = observ - self._last
        self._last = observ
        return self.above.observe(delta)

    def receive(self, reward, final):
        super().receive(reward, final)
        self.above.receive(reward, final)

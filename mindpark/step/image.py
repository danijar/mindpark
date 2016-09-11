import numpy as np
from gym.spaces import Box, Discrete
from mindpark.core import Partial


class Image(Partial):

    def __init__(self, task):
        super().__init__(task)

    @property
    def above_observs(self):
        if isinstance(self.task.observs, Box):
            low = self._expand_dims(self.task.observs.low)
            high = self._expand_dims(self.task.observs.high)
        elif isinstance(self.task.observs, Discrete):
            low = self._expand_dims(0)
            high = self._expand_dims(self.task.observs.n)
        else:
            raise NotImplementedError
        return Box(low, high)

    @property
    def above_actions(self):
        return self.task.actions

    def observe(self, observ):
        super().observe(observ)
        observ = self._expand_dims(observ)
        assert self.above_task.observs.contains(observ)
        return self.above.observe(observ)

    def receive(self, reward, final):
        super().receive(reward, final)
        self.above.receive(reward, final)

    def _expand_dims(self, observ):
        observ = np.array(observ)
        if len(observ.shape) > 3:
            raise ValueError('observations already have too many dimensions')
        for _ in range(3 - len(observ.shape)):
            observ = np.expand_dims(observ, -1)
        return observ

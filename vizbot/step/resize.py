from gym.spaces import Box
from vizbot.core import Partial


class Resize(Partial):

    def __init__(self, task, shape):
        super().__init__(task)
        self._shape = shape

    @property
    def interface(self):
        low = self._apply(self.task.observs.low)
        high = self._apply(self.task.observs.high)
        return Box(low, high), self.actions

    def step(self, observ):
        super().step(observ)
        observ = self._apply(observ)
        return self.above.step(observ)

    def experience(self, *transition):
        super().experience(*transition)
        self.above.experience(*transition)

    def _apply(self, observ):
        assert NotImplementedError

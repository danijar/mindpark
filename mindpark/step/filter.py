from abc import abstractmethod
from gym.spaces import Box
from mindpark.core import Partial


class Filter(Partial):

    """
    Base class for steps that modify individual the observations. Forward
    rewards and actions without changing them. If your filter changes the range
    of the observation space, override `above_observs` accordingly.
    """

    def __init__(self, task):
        super().__init__(task)

    @property
    def above_observs(self):
        low = self.filter(self.task.observs.low)
        high = self.filter(self.task.observs.high)
        return Box(low, high)

    @property
    def above_actions(self):
        return self.task.actions

    def observe(self, observ):
        super().observe(observ)
        observ = self.filter(observ)
        assert self.above_task.observs.contains(observ)
        return self.above.observe(observ)

    def receive(self, reward, final):
        super().receive(reward, final)
        self.above.receive(reward, final)

    @abstractmethod
    def filter(self, observ):
        pass

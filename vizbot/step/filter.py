from gym.spaces import Box
from vizbot.core import Policy


class Filter(Policy):

    """
    Base class for steps that modify individual the observations. Forward
    rewards and actions without changing them. If your filter chances the range
    of observation space, override `interface` accordingly.
    """

    def __init__(self, interface):
        super().__init__(interface)

    @property
    def interface(self):
        low = self.filter(self.observations.low)
        high = self.filter(self.observations.high)
        return Box(low, high), self.actions

    def observe(self, observation):
        super().observe(observation)
        observation = self.filter(observation)
        assert self.interface[0].contains(observation)
        return self.above.observe(observation)

    def receive(self, reward, final):
        super().receive(reward, final)
        self.above.receive(reward, final)

    def filter(self, observation):
        raise NotImplementedError

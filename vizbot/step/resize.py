from gym.spaces import Box
from vizbot.core import Policy


class Resize(Policy):

    def __init__(self, interface, shape):
        super().__init__(interface)
        self._shape = shape

    @property
    def interface(self):
        low = self._apply(self.observations.low)
        high = self._apply(self.observations.high)
        return Box(low, high), self.actions

    def step(self, observation):
        super().step(observation)
        observation = self._apply(observation)
        return self.above.step(observation)

    def experience(self, *transition):
        super().experience(*transition)
        self.above.experience(*transition)

    def _apply(self, observation):
        assert NotImplementedError

import numpy as np
from gym.spaces import Box
from vizbot.core import Policy


class Subsample(Policy):

    def __init__(self, interface, amount=(2, 2, 1)):
        super().__init__(interface)
        if not all(isinstance(x, int) for x in amount):
            raise ValueError('amount must be an int for each dimension')
        self._amount = amount

    @property
    def interface(self):
        low = self._apply(self.observations.low)
        high = self._apply(self.observations.high)
        return Box(low, high), self.actions

    def step(self, observation):
        super().step(observation)
        observation = self._apply(observation)
        assert self.interface[0].contains(observation)
        return self.above.step(observation)

    def experience(self, *transition):
        super().experience(*transition)
        self.above.experience(*transition)

    def _apply(self, observation):
        for amount in self._amount:
            observation = observation[::amount]
            observation = np.moveaxis(observation, 1, -1)
        return observation

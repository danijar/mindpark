import numpy as np
from vizbot.step.filter import Filter


class Subsample(Filter):

    def __init__(self, interface, amount=(2, 2, 1)):
        super().__init__(interface)
        if not len(amount) == len(self.observations.shape):
            raise ValueError('amount must be a number for each dimension')
        if not all(isinstance(x, int) for x in amount):
            raise ValueError('can only sub sample by integer amounts')
        self._amount = amount

    def filter(self, observation):
        for amount in self._amount:
            observation = observation[::amount]
            observation = np.moveaxis(observation, 0, -1)
        return observation

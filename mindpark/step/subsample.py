import numpy as np
from mindpark.step.filter import Filter


class Subsample(Filter):

    def __init__(self, task, amount=(2, 2, 1)):
        super().__init__(task)
        if not len(amount) == len(self.task.observs.shape):
            raise ValueError('amount must be a number for each dimension')
        if not all(isinstance(x, int) for x in amount):
            raise ValueError('can only sub sample by integer amounts')
        self._amount = amount

    def filter(self, observ):
        for amount in self._amount:
            observ = observ[::amount]
            observ = np.moveaxis(observ, 0, -1)
        return observ

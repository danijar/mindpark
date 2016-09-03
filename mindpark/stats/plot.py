import numpy as np


class Plot:

    def __call__(self, ax, data):
        raise NotImplementedError

    def _aggregate(self, values, borders, reducer):
        groups = []
        for start, stop in zip(borders[:-1], borders[1:]):
            groups.append(reducer(values[start: stop]))
        groups = np.array(groups)
        return groups

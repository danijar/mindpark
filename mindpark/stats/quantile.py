import numpy as np
from mindpark.stats.plot import Plot


class Quantile(Plot):

    AREA = dict(alpha=0.2)

    def __init__(self, legend, resolution=20):
        super().__init__(legend)
        self._resolution = resolution

    def _plot(self, ax, domain, line, label, color, marker):
        order = np.argsort(domain)
        domain, line = domain[order], line[order]

        borders = np.linspace(domain[0], domain[-1], self._resolution)
        borders = np.digitize(borders, domain)
        domain = np.linspace(domain[0], domain[-1], len(borders) - 1)
        lower_ = self._aggregate(
            line, borders, lambda x: np.percentile(x, 10, axis=0)[0])
        middle = self._aggregate(
            line, borders, lambda x: np.percentile(x, 50, axis=0)[0])
        upper_ = self._aggregate(
            line, borders, lambda x: np.percentile(x, 90, axis=0)[0])

        ax.fill_between(
            domain, upper_, lower_, facecolor=color, edgecolor=color,
            **self.AREA)
        ax.plot(
            domain, middle, c=color, label=label)

import numpy as np
from matplotlib.ticker import MaxNLocator
from mindpark.stats.plot import Plot


class Scatter(Plot):

    def __call__(self, ax, value, ticks):
        domain = np.linspace(0, ticks + 1, len(value))
        ax.scatter(domain, value, alpha=0.1, lw=0)
        ax.set_xlim(domain.min(), domain.max())
        padding = 0.05 * (value.max() - value.min())
        padding = padding or np.abs(np.log10(value[0])) / 100
        ax.set_ylim(value.min() - padding, value.max() + padding)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

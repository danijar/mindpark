import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mindpark.stats.plot import Plot


class Histogram(Plot):

    def __init__(self, resolution=10, normalize=False):
        self._resolution = resolution
        self._normalize = normalize

    def __call__(self, ax, counts, ticks):
        borders = np.linspace(0, len(counts), self._resolution * ticks)
        borders = borders.astype(int)
        groups = self._aggregate(counts, borders, lambda x: np.mean(x, axis=0))
        domain = np.linspace(0, ticks + 1, self._resolution * len(groups))
        bar = self._plot_grid(ax, domain, groups)
        if self._normalize:
            bar.set_ticks([])
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    def _plot_grid(self, ax, domain, cells):
        extent = [domain.min(), domain.max(), -.5, cells.shape[1] - .5]
        kwargs = dict(cmap='viridis')
        img = ax.matshow(
            cells.T, extent=extent, origin='lower', aspect='auto', **kwargs)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='7%', pad=0.1)
        bar = plt.colorbar(img, cax=cax)
        ax.xaxis.set_ticks_position('bottom')
        return bar

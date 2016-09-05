import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable


class Histogram:

    def __init__(self, resolution=25, normalize=False):
        super().__init__()
        self._resolution = resolution
        self._normalize = normalize

    def __call__(self, ax, domain, count):
        assert len(domain) == len(count)
        assert domain[0] < domain[-1]
        order = np.argsort(domain)
        domain, count = domain[order], count[order]
        borders = np.linspace(domain[0], domain[-1], self._resolution)
        borders = np.digitize(borders, domain)
        groups = self._aggregate(count, borders, lambda x: np.mean(x, axis=0))
        bar = self._plot_grid(ax, domain[borders - 1], groups)
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

    def _aggregate(self, values, borders, reducer):
        groups = []
        for start, stop in zip(borders[:-1], borders[1:]):
            groups.append(reducer(values[start: stop]))
        groups = np.array(groups)
        return groups

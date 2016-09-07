import numpy as np
from matplotlib import cm
from mindpark.utility import aggregate, add_color_bar


class Histogram:

    def __init__(self, resolution=100, normalize=False):
        self._resolution = resolution
        self._normalize = normalize

    def __call__(self, ax, domain, count):
        assert len(domain) == len(count)
        assert domain[0] < domain[-1]
        ax.set_axis_bgcolor(cm.get_cmap('viridis')(0))
        order = np.argsort(domain)
        domain, count = domain[order], count[order]
        resolution = min(len(domain), self._resolution)
        borders = np.linspace(domain[0], domain[-1], resolution)
        borders = np.digitize(borders, domain)
        groups = aggregate(count, borders, lambda x: np.mean(x, axis=0))
        self._plot_grid(ax, domain[borders - 1], groups)
        ax.set_yticks(np.arange(count.shape[1]))

    def _plot_grid(self, ax, domain, cells):
        extent = [domain.min(), domain.max(), -.5, cells.shape[1] - .5]
        img = ax.imshow(
            cells.T, extent=extent, origin='lower', aspect='auto',
            interpolation='nearest', cmap='viridis')
        bar = add_color_bar(ax, img)
        if self._normalize:
            bar.set_ticks([])

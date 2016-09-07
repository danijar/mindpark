import numpy as np
from mindpark.utility import add_color_bar


class Scalar:

    HEXBIN = dict(gridsize=100, cmap='viridis', bins='log')

    def __call__(self, ax, domain, line):
        extent = self._get_limits(domain, line)
        img = ax.hexbin(domain, line, extent=extent, **self.HEXBIN)
        add_color_bar(ax, img).set_ticks([])

    def _get_limits(self, domain, line):
        xmin, xmax = domain.min(), domain.max()
        ymin, ymax = line.min(), line.max()
        padding = 0.1 * (ymax - ymin) or np.abs(np.log10(ymin)) / 100
        ymin, ymax = ymin - padding, ymax + padding
        return xmin, xmax, ymin, ymax

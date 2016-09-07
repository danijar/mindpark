import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


class Scalar:

    HEXBIN = dict(gridsize=100, cmap='viridis', bins='log')

    def __call__(self, ax, domain, line):
        extent = self._get_limits(domain, line)
        mappable = ax.hexbin(domain, line, extent=extent, **self.HEXBIN)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='7%', pad=0.1)
        bar = plt.colorbar(mappable, cax=cax)
        bar.set_ticks([])

    def _get_limits(self, domain, line):
        xmin, xmax = domain.min(), domain.max()
        ymin, ymax = line.min(), line.max()
        padding = 0.1 * (ymax - ymin) or np.abs(np.log10(ymin)) / 100
        ymin, ymax = ymin - padding, ymax + padding
        return xmin, xmax, ymin, ymax

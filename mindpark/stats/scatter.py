import numpy as np
from matplotlib.ticker import MaxNLocator


class Scatter:

    COLORS = ('#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33',
              '#a65628', '#f781bf')
    MARKERS = 'so^Dp>d<'
    SCATTER = dict(alpha=0.1, lw=0)
    LEGEND = dict(loc='best', fontsize='medium', labelspacing=0, numpoints=1)

    def __call__(self, ax, lines, ticks):
        if isinstance(lines, dict):
            max_length = max(len(x) for x in lines.values())
            domain = np.linspace(0, ticks + 1, max_length)
            lines = sorted(lines.items(), key=lambda x: -np.nanmean(x[1]))
            for index, (label, line) in enumerate(lines):
                self._plot(ax, domain, line, label, index)
            self._plot_legend(ax)
        else:
            domain = np.linspace(0, ticks + 1, len(lines))
            self._plot(ax, domain, lines, None, 1)
        ax.set_xlim(domain.min(), domain.max())
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    def _plot(self, ax, domain, line, label, index):
        color, marker = self.COLORS[index], self.MARKERS[index]
        ax.scatter(
            domain, line, c=color, marker=marker, label=label, **self.SCATTER)
        padding = 0.1 * (line.max() - line.min())
        padding = padding or np.abs(np.log10(line[0])) / 100
        ax.set_ylim(line.min() - padding, line.max() + padding)

    def _plot_legend(self, ax):
        leg = ax.legend(**self.LEGEND)
        leg.get_frame().set_edgecolor('white')
        for line in leg.get_lines():
            line.set_alpha(1)

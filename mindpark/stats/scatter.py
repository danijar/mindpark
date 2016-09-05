import numpy as np
from matplotlib.ticker import MaxNLocator


class Scatter:

    COLORS = ('#377eb8', '#e41a1c', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33',
              '#a65628', '#f781bf')
    MARKERS = 'os^Dp>d<'
    SCATTER = dict(alpha=0.1, lw=0)
    LEGEND = dict(
        loc='best', fontsize='medium', labelspacing=0, numpoints=1,
        scatterpoints=1)

    def __init__(self, legend, scatter={}):
        self._legend = legend
        self._scatter = self.SCATTER.copy()
        self._scatter.update(scatter)

    def __call__(self, ax, domains, lines):
        if not isinstance(lines, dict):
            domains, lines = {'Line': domains}, {'Line': lines}
        assert domains.keys() == lines.keys()
        labels = sorted(lines.keys(), key=lambda x: -np.nanmean(lines[x]))
        for index, label in enumerate(labels):
            self._plot(ax, domains[label], lines[label], label, index)
        if self._legend:
            self._plot_legend(ax)
        min_ = min(x.min() for x in domains.values())
        max_ = max(x.max() for x in domains.values())
        ax.set_xlim(min_, max_)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    def _plot(self, ax, domain, line, label, index):
        color, marker = self.COLORS[index], self.MARKERS[index]
        ax.scatter(domain, line, c=color, marker=marker, **self._scatter)
        legend = self._scatter.copy()
        legend['alpha'] = 1.0
        ax.scatter([], [], c=color, marker=marker, label=label, **legend)
        padding = 0.1 * (line.max() - line.min())
        padding = padding or np.abs(np.log10(line[0])) / 100
        ax.set_ylim(line.min() - padding, line.max() + padding)

    def _plot_legend(self, ax):
        leg = ax.legend(**self.LEGEND)
        leg.get_frame().set_edgecolor('white')
        for line in leg.get_lines():
            line.set_alpha(1)

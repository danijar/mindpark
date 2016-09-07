import numpy as np
from mindpark.utility import aggregate


class Lines:

    COLORS = ('#377eb8', '#e41a1c', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33',
              '#a65628', '#f781bf')
    MARKERS = 'os^Dp>d<'
    LEGEND = dict(loc='best', fontsize='medium', labelspacing=0, numpoints=1)
    AREA = dict(alpha=0.2)

    def __init__(self, resolution=20, legend=True):
        self._resolution = resolution
        self._legend = legend

    def __call__(self, ax, domains, lines):
        assert domains.keys() == lines.keys()
        labels = sorted(lines.keys(), key=lambda x: -np.nanmean(lines[x]))
        for index, label in enumerate(labels):
            color, marker = self.COLORS[index], self.MARKERS[index]
            domain, line = domains[label], lines[label]
            self._plot_line(ax, domain, line, label, color, marker)
        if self._legend:
            self._plot_legend(ax)

        min_ = min(x.min() for x in domains.values())
        max_ = max(x.max() for x in domains.values())
        ax.set_xlim(min_, max_)

        min_ = max(x.min() for x in lines.values())
        max_ = max(x.max() for x in lines.values())
        padding = 0.1 * (max_ - min_) or np.abs(np.log10(min_)) / 100
        ax.set_ylim(min_ - padding, max_ + padding)

    def _plot_line(self, ax, domain, line, label, color, marker):
        order = np.argsort(domain)
        domain, line = domain[order], line[order]

        borders = np.linspace(domain[0], domain[-1], self._resolution)
        borders = np.digitize(borders, domain)
        domain = np.linspace(domain[0], domain[-1], len(borders) - 1)
        lower_ = aggregate(line, borders, lambda x: np.percentile(x, 10, 0)[0])
        middle = aggregate(line, borders, lambda x: np.percentile(x, 50, 0)[0])
        upper_ = aggregate(line, borders, lambda x: np.percentile(x, 90, 0)[0])

        ax.fill_between(
            domain, upper_, lower_, facecolor=color, edgecolor=color,
            **self.AREA)
        ax.plot(
            domain, middle, c=color, label=label)

    def _plot_legend(self, ax):
        leg = ax.legend(**self.LEGEND)
        leg.get_frame().set_edgecolor('white')
        for line in leg.get_lines():
            line.set_alpha(1)

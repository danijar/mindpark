import numpy as np


class Plot:

    COLORS = ('#377eb8', '#e41a1c', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33',
              '#a65628', '#f781bf')
    MARKERS = 'os^Dp>d<'
    LEGEND = dict(
        loc='best', fontsize='medium', labelspacing=0, numpoints=1,
        scatterpoints=1)  # handletextpad=0

    def __init__(self, legend):
        self._legend = legend

    def __call__(self, ax, domains, lines):
        if not isinstance(lines, dict):
            domains, lines = {'Line': domains}, {'Line': lines}
        assert domains.keys() == lines.keys()
        labels = sorted(lines.keys(), key=lambda x: -np.nanmean(lines[x]))
        for index, label in enumerate(labels):
            color, marker = self.COLORS[index], self.MARKERS[index]
            self._plot(ax, domains[label], lines[label], label, color, marker)
        if self._legend:
            self._plot_legend(ax)

        min_ = min(x.min() for x in domains.values())
        max_ = max(x.max() for x in domains.values())
        ax.set_xlim(min_, max_)

        min_ = max(x.min() for x in lines.values())
        max_ = max(x.max() for x in lines.values())
        padding = 0.1 * (max_ - min_) or np.abs(np.log10(min_)) / 100
        ax.set_ylim(min_ - padding, max_ + padding)

        # self._locate_ticks(ax)

    def _aggregate(self, values, borders, reducer):
        groups = []
        for start, stop in zip(borders[:-1], borders[1:]):
            groups.append(reducer(values[start: stop]))
        groups = np.array(groups)
        return groups

    def _plot_legend(self, ax):
        leg = ax.legend(**self.LEGEND)
        leg.get_frame().set_edgecolor('white')
        for line in leg.get_lines():
            line.set_alpha(1)

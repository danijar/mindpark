import os
import numpy as np
import matplotlib.pyplot as plt
from vizbot.utility.other import ensure_directory


class DeviationFigure:

    """
    A figure of multiple plots, each containing labelled lines and their
    standard deviation.
    """

    COLORS = (
        '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33',
        '#a65628', '#f781bf')
    MARKERS = 'so^Dp>d<'
    AREA = dict(alpha=0.15)
    LINE = dict(markersize=5, markeredgewidth=0)
    LEGEND = dict(loc='best', fontsize='medium', labelspacing=0, numpoints=1)

    def __init__(self, ncols, title, offset=0):
        """
        Create a figure that can hold a certain amount of plots.
        """
        self._ncols = ncols
        self._fig = plt.figure(figsize=(12, 4))
        self._fig.suptitle(title, fontsize=16)
        self._index = 1
        self._offset = offset

    def add(self, title, xlabel, ylabel, **lines):
        """
        Add a plot of certain lines. Lines are passed as a mapping from labels
        to data. Data is an array of multiple runs over which to compute the
        standard deviation.
        """
        lines = sorted(lines.items(), key=lambda x: -np.sum(x[1]))
        ax = self._next_plot()
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        for index, (label, line) in enumerate(lines):
            color = self.COLORS[index]
            marker = self.MARKERS[index]
            self._plot(ax, label, line, color, marker)
        leg = ax.legend(**self.LEGEND)
        leg.get_frame().set_edgecolor('white')
        return ax

    def save(self, filepath):
        """
        Render the figure and save it to disk.
        """
        ensure_directory(os.path.dirname(filepath))
        self._fig.tight_layout(rect=[0, 0, 1, .93])
        self._fig.savefig(filepath, dpi=300)

    def _next_plot(self):
        if self._index > self._ncols:
            raise RuntimeError
        ax = self._fig.add_subplot(1, self._ncols, self._index)
        self._index += 1
        return ax

    def _plot(self, ax, label, line, color, marker):
        means = line.mean(axis=0)
        stds = line.std(axis=0)
        domain = np.arange(self._offset, self._offset + line.shape[1])
        ax.fill_between(
            domain, means - stds, means + stds, color=color, **self.AREA)
        ax.plot(
            domain, means, label=label, color=color, marker=marker,
            **self.LINE)

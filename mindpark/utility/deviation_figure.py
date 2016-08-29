import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from mindpark.utility.other import ensure_directory


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

    def __init__(self, ncols, title):
        """
        Create a figure that can hold a certain amount of plots.
        """
        self._ncols = ncols
        self._fig = plt.figure(figsize=(12, 4))
        self._fig.suptitle(title, fontsize=16)
        self._index = 1

    def add(self, title, xlabel, ylabel, domain, lines):
        """
        Add a plot of certain lines. Lines are passed as a mapping from labels
        to data. Data is an array of multiple runs over which to compute the
        standard deviation.
        """
        ax = self._next_plot()
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            lines = sorted(lines.items(), key=lambda x: -np.nanmean(x[1]))
        for index, (label, line) in enumerate(lines):
            assert len(domain) == len(line)
            color = self.COLORS[index]
            marker = self.MARKERS[index]
            self._plot(ax, label, domain, line, color, marker)
        ax.set_xlim(domain.min(), domain.max())
        leg = ax.legend(**self.LEGEND)
        leg.get_frame().set_edgecolor('white')
        return ax

    def save(self, filepath):
        """
        Render the figure and save it to disk. The file extension determines
        the image format.
        """
        ensure_directory(os.path.dirname(filepath))
        self._fig.tight_layout(rect=[0, 0, 1, .93])
        self._fig.savefig(filepath, dpi=300)

    def close(self):
        plt.close(self._fig)

    def _next_plot(self):
        if self._index > self._ncols:
            raise RuntimeError
        ax = self._fig.add_subplot(1, self._ncols, self._index)
        self._index += 1
        return ax

    def _plot(self, ax, label, domain, line, color, marker):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            means = np.nanmean(line, axis=1)
            stds = np.nanstd(line, axis=1)
        below, above = means - stds, means + stds
        ax.fill_between(domain, below, above, color=color, **self.AREA)
        ax.plot(
            domain, means, label=label, color=color, marker=marker,
            **self.LINE)

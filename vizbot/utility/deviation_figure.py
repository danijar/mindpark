import os
import numpy as np
import matplotlib.pyplot as plt
from vizbot.utility.attrdict import AttrDict
from vizbot.utility.other import ensure_directory


class DeviationFigure:

    """
    A figure of multiple plots, each containing labelled lines and their
    standard deviation.
    """

    COLORS = ('green', 'blue', 'red', 'yellow')

    LABEL_ARGS = AttrDict(
        loc='best', frameon=False, fontsize='medium', labelspacing=0)

    def __init__(self, ncols, title):
        """
        Create a figure that can hold a certain amount of plots.
        """
        self._ncols = ncols
        self._fig = plt.figure(figsize=(12, 4))
        self._fig.suptitle(title, fontsize=16)
        self._index = 1

    def add(self, title, xlabel, ylabel, **lines):
        """
        Add a plot of certain lines. Lines are passed as a mapping from labels
        to data. Data is an array of multiple runs over which to compute the
        standard deviation.
        """
        lines = sorted(lines.items(), key=lambda x: -x[1].sum())
        ax = self._next_plot()
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        for index, (label, line) in enumerate(lines):
            color = self.COLORS[index]
            self._plot(ax, label, line, color)
        ax.legend(**self.LABEL_ARGS)

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

    def _plot(self, ax, label, line, color):
        means = line.mean(axis=0)
        stds = line.std(axis=0)
        area = np.arange(len(means)), means - stds, means + stds
        ax.fill_between(*area, color=color, alpha=0.15)
        ax.plot(means, label=label, color=color)

import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
from vizbot.utility.deviation_figure import DeviationFigure


class EpochFigure(DeviationFigure):

    """
    A deviation figure where time axes are grouped based on an epoch size and
    duration information for the lines.
    """

    def __init__(self, ncols, title, epochs):
        super().__init__(ncols, title)
        self._epochs = epochs

    def add(self, title, xlabel, ylabel, lines, durations):
        starts = self._compute_starts(durations)
        lines = self._average_lines(lines, starts)
        ax = super().add(title, xlabel, ylabel, **lines)
        ax.set_xlim(0, max(x.shape[1] for x in lines.values()) - 1)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_formatter(FuncFormatter(self._formatter))

    def _formatter(self, value, position):
        return str(int(value + 1))

    def _compute_starts(self, durations):
        starts = {}
        for label, duration in durations.items():
            starts[label] = [[0] + np.cumsum(x[:-1]) for x in duration]
        return starts

    def _average_lines(self, lines, starts):
        averaged = {}
        for label, line in lines.items():
            start = starts[label]
            last = max(x[-1] for x in start)
            line = [self._average(x, y) for x, y in zip(line, start)]
            averaged[label] = np.array(line, dtype=float)
        return averaged

    def _average(self, values, starts):
        sums, counts = np.zeros(self._epochs), np.zeros(self._epochs)
        clamped = 0
        for value, start in zip(values, starts):
            epoch = start // starts[-1]
            if epoch >= self._epochs:
                clamped += 1
                epoch = self._epochs - 1
            sums[epoch] += value
            counts[epoch] += 1
        empty = (counts == 0)
        averages = sums / np.maximum(counts, 1)
        averages[empty] = np.nan
        if clamped:
            print('Clamped', clamped, 'eposides to last epoch.')
        return averages

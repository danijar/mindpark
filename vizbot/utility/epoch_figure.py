import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
from vizbot.utility.deviation_figure import DeviationFigure


class EpochFigure(DeviationFigure):

    """
    A deviation figure where time axes are grouped based on an epoch size and
    duration information for the lines.
    """

    def __init__(self, ncols, title, epochs, epoch_size):
        super().__init__(ncols, title)
        self._epoch_size = epoch_size
        self._epochs = epochs

    def add(self, title, xlabel, ylabel, lines, durations):
        starts = self._compute_starts(durations)
        lines = self._average_lines(lines, starts)
        ax = super().add(title, xlabel, ylabel, **lines)
        ax.set_xlim(0, self._epochs - 1)
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
            line = [self._average(x, y) for x, y in zip(line, start)]
            averaged[label] = np.array(line, dtype=float)
        return averaged

    def _average(self, values, starts):
        sums, counts = np.zeros(self._epochs), np.zeros(self._epochs)
        for value, start in zip(values, starts):
            epoch = start // self._epoch_size
            if epoch >= self._epochs:
                print('Skip episode after last epoch in the diagram')
                continue
            sums[epoch] += value
            counts[epoch] += 1
        averages = sums / counts
        return averages

import warnings
import numpy as np
from matplotlib.ticker import FuncFormatter, MaxNLocator
from vizbot.utility.deviation_figure import DeviationFigure


class EpochFigure(DeviationFigure):

    """
    A deviation figure where time axes are grouped based on an epoch size and
    duration information for the lines.
    """

    def __init__(self, ncols, title, resolution, epochs, epoch_length):
        super().__init__(ncols, title, resolution)
        self._epochs = epochs + 1
        self._bins = resolution * self._epochs
        self._bin_size = epoch_length / resolution

    def add(self, title, xlabel, ylabel, lines, durations):
        lines = self._average_lines(lines, durations)
        ax = super().add(title, xlabel, ylabel, **lines)
        ax.set_xlim(0, self._epochs)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.xaxis.set_major_formatter(FuncFormatter(self._formatter))

    def _formatter(self, value, position):
        return str(int(value))

    def _average_lines(self, lines, durations):
        starts = self._compute_starts(durations)
        averaged = {}
        for label, line in lines.items():
            start, duration = starts[label], durations[label]
            last = max(x[-1] + y[-1] for x, y in zip(start, duration))
            size = max(self._bin_size, last / self._bins)
            line = [self._average(x, y, size) for x, y in zip(line, start)]
            averaged[label] = np.array(line, dtype=float)
        return averaged

    def _average(self, values, starts, size):
        assert len(values) == len(starts)
        sums, counts = np.zeros(self._bins), np.zeros(self._bins)
        clamped = 0
        for value, start in zip(values, starts):
            epoch = int(start / size)
            if epoch >= self._bins:
                clamped += 1
                epoch = self._bins - 1
            sums[epoch] += value
            counts[epoch] += 1
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            averages = sums / counts
        if clamped:
            print('Clamped', clamped, 'eposides to last epoch.')
        return averages

    def _compute_starts(self, durations):
        starts = {}
        for label, duration in durations.items():
            duration = [np.array(x) for x in duration]
            start = [np.cumsum(x[:-1]) for x in duration]
            start = [np.insert(x, 0, 0) for x in start]
            starts[label] = start
        return starts

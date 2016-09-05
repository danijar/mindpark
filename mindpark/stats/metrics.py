import numpy as np
import matplotlib.pyplot as plt
from mindpark.stats.scatter import Scatter
from mindpark.stats.histogram import Histogram


class Metrics:

    def __init__(self, type_):
        self._type = type_
        self._plot_scalar = Scatter()
        self._plot_counts = Histogram()
        self._plot_distribution = Histogram(normalize=True)

    def __call__(self, metrics, title, filepath):
        fig, ax = self._create_subplots(2, len(metrics))
        fig.suptitle(title, fontsize=16)
        for index, (name, rows) in enumerate(metrics):
            ax[0, index].set_title(name)
            ax[1, index].set_title(name)
            self._process_metric(ax[0, index], rows[rows.T[4] == 1])
            self._process_metric(ax[1, index], rows[rows.T[4] == 0])
        self._label_rows(ax, ['Training', 'Evaluation'])
        fig.tight_layout(rect=[0, 0, 1, .94])
        fig.savefig(filepath)

    def _create_subplots(self, rows, cols, **kwargs):
        size = [4 * cols, 3 * rows]
        fig, ax = plt.subplots(ncols=cols, nrows=rows, figsize=size, **kwargs)
        if cols == 1:
            ax = np.array([ax]).T
        return fig, ax

    def _label_rows(self, ax, labels):
        for index, label in enumerate(labels):
            ax[index, 0].set_ylabel(label, fontsize=16)
            ax[index, 0].yaxis.labelpad = 16

    def _process_metric(self, ax, rows):
        if not rows.size:
            ax.tick_params(colors=(0, 0, 0, 0))
            return
        values = rows[:, 6:].astype(float)
        ticks = rows.T[3].max() or 1
        categorical = self._is_categorical(values)
        if values.shape[1] == 1 and not categorical:
            self._plot_scalar(ax, values[:, 0], ticks)
        elif values.shape[1] == 1:
            indices = values[:, 0].astype(int)
            min_, max_ = indices.min(), indices.max()
            histograms = np.eye(max_ - min_ + 1)[indices - min_]
            self._plot_distribution(ax, histograms, ticks)
        elif values.shape[1] > 1:
            self._plot_counts(ax, values, ticks)

    def _is_categorical(self, values):
        if not np.allclose(values, values.astype(int)):
            return False
        if np.unique(values.astype(int)).size > 10:
            return False
        return True

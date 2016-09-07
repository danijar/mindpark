import numpy as np
import matplotlib.pyplot as plt
from mindpark.stats.reader import Metric


class Figure:

    def _save(self, fig, filepath):
        fig.tight_layout(rect=[0, 0, 1, .94])
        fig.savefig(filepath)

    def _domain(self, line):
        display = line.episode  # line.epoch
        # Choose two points to linearly fit display qunatity.
        display_1, display_2 = np.unique(display)[[1, -1]]
        index_1 = np.argmax(display == display_1)
        index_2 = np.argmax(display == display_2)
        step_1, step_2 = line.step[index_1], line.step[index_2]
        # Linearly transform steps onto display quantity.
        slope = (display_2 - display_1) / (step_2 - step_1)
        offset = display_1 - slope * step_1
        domain = slope * line.step + offset
        # Start counting at index one.
        domain += 1
        return domain

    def _label_columns(self, ax, labels):
        for index, label in enumerate(labels):
            ax[0, index].set_title(label, fontsize=16, y=1.07)

    def _label_rows(self, ax, labels):
        for index, label in enumerate(labels):
            ax[index, 0].set_ylabel(label, fontsize=16)
            ax[index, 0].yaxis.set_label_coords(-.15, 0.5)

    def _create_subplots(self, rows, cols, **kwargs):
        size = [4 * cols, 3 * rows]
        fig, ax = plt.subplots(ncols=cols, nrows=rows, figsize=size, **kwargs)
        if cols == 1:
            ax = np.array([ax]).T
        return fig, ax

    def _concat_metrics(self, metrics):
        keys = [list(x.keys()) for x in metrics]
        keys = set(keys[0]).union(*keys[1:])
        assert all((y in x for y in keys) for x in metrics)
        metric = {x: np.concatenate([y[x] for y in metrics]) for x in keys}
        return Metric(metric)

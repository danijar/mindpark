import numpy as np
import matplotlib.pyplot as plt


class Figure:

    def _save(self, fig, filepath):
        fig.tight_layout(rect=[0, 0, 1, .94])
        fig.savefig(filepath)

    def _domain(self, metric, ticks):
        domain = (metric.step - metric.step.min()) / (metric.step.max() or 1)
        domain = domain * (ticks + 1)
        return domain

    def _label_columns(self, ax, labels):
        for index, label in enumerate(labels):
            ax[0, index].set_title(label, fontsize=16, y=1.07)

    def _label_rows(self, ax, labels):
        for index, label in enumerate(labels):
            ax[index, 0].set_ylabel(label, fontsize=16)
            ax[index, 0].yaxis.labelpad = 16

    def _create_subplots(self, rows, cols, **kwargs):
        size = [4 * cols, 3 * rows]
        fig, ax = plt.subplots(ncols=cols, nrows=rows, figsize=size, **kwargs)
        if cols == 1:
            ax = np.array([ax]).T
        return fig, ax

import numpy as np
from mindpark.stats.figure import Figure
from mindpark.stats.scatter import Scatter
from mindpark.stats.histogram import Histogram


class Metrics(Figure):

    """
    Generate figures showing multiple metrics as plots. Choose the plot type
    automatically, based on the data.
    """

    def __init__(self):
        self._plot_scalar = Scatter(legend=False)
        self._plot_counts = Histogram()
        self._plot_distribution = Histogram(normalize=True)

    def __call__(self, metrics, title, filepath):
        self._validate_input(metrics)
        fig, ax = self._create_subplots(2, len(metrics))
        fig.suptitle(title, fontsize=16)
        names, metrics = zip(*metrics)
        self._label_columns(ax, names)
        self._label_rows(ax, ['Training', 'Evaluation'])
        for index, metric in enumerate(metrics):
            ticks = metric.episode.max()
            train = metric[metric.training == 1]
            test = metric[metric.training == 0]
            self._process_metric(ax[0, index], train, ticks)
            self._process_metric(ax[1, index], test, ticks)
        self._save(fig, filepath)

    def _validate_input(self, metrics):
        assert all(len(x) == 2 for x in metrics)
        for name, metric in metrics:
            assert isinstance(name, str)
            assert all(isinstance(x, np.ndarray) for x in metric.values())

    def _process_metric(self, ax, metric, ticks):
        if not metric.data.size:
            ax.tick_params(colors=(0, 0, 0, 0))
            return
        domain = self._domain(metric, ticks)
        categorical = self._is_categorical(metric.data)
        if metric.data.shape[1] == 1 and not categorical:
            self._plot_scalar(ax, domain, metric.data[:, 0])
        elif metric.data.shape[1] == 1:
            indices = metric.data[:, 0].astype(int)
            min_, max_ = indices.min(), indices.max()
            count = np.eye(max_ - min_ + 1)[indices - min_]
            self._plot_distribution(ax, domain, count)
        elif metric.data.shape[1] > 1:
            self._plot_counts(ax, domain, metric.data)

    def _is_categorical(self, data):
        if not np.allclose(data, data.astype(int)):
            return False
        if np.unique(data.astype(int)).size > 8:
            return False
        return True

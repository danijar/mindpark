import os
import collections
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import sqlalchemy as sql
from mindpark.stats.scatter import Scatter
from mindpark.stats.histogram import Histogram
from mindpark.utility import get_subdirs, natural_sorted, read_yaml


Run = collections.namedtuple(
    'Run', 'experiment name env algorithm repeat stats')


class Stats:

    def __init__(self, type_, metrics=None):
        self._type = type_
        self._metrics = metrics
        self._plot_scalar = Scatter()
        self._plot_counts = Histogram()
        self._plot_distribution = Histogram(normalize=True)

    def __call__(self, experiment):
        for run in self._collect_runs(experiment):
            self._process(run)

    def _collect_runs(self, experiment):
        print('Read experiment', experiment)
        name = os.path.basename(experiment).title()
        for env_dir in get_subdirs(experiment):
            env = os.path.basename(env_dir)
            for directory in natural_sorted(get_subdirs(env_dir)):
                repeat = int(directory.rsplit('-', 1)[-1])
                algorithm = read_yaml(directory, 'algorithm.yaml').name
                stats = os.path.join(directory, 'stats.db')
                yield Run(experiment, name, env, algorithm, repeat, stats)

    def _process(self, run):
        title = '{} on {} (Repeat {})'.format(
            run.algorithm, run.env, run.repeat)
        filepath = '{}-{}-{}-{}.{}'.format(
            run.name, run.env, run.algorithm, run.repeat, self._type)
        filepath = os.path.join(run.experiment, filepath)
        print(' Plot run', title)
        self._figure(run.stats, title, filepath)

    def _figure(self, stats, title, filepath):
        engine = sql.create_engine('sqlite:///{}'.format(stats))
        tables = self._get_tables(engine)
        if not tables:
            print('  No metrics found.')
            return
        tables = self._select_metrics(tables)
        fig, ax = self._subplots(2, len(tables))
        fig.suptitle(title, fontsize=16)
        for index, (title, table) in enumerate(tables):
            rows = self._collect_stats(engine, table)
            if rows is None:
                continue
            test, train = rows[rows.T[4] == 0], rows[rows.T[4] == 1]
            if train.size:
                self._plot(ax[0, index], train)
                ax[0, index].set_title(title)
            else:
                ax[0, index].tick_params(colors=(0, 0, 0, 0))
            if test.size:
                self._plot(ax[1, index], test)
                ax[1, index].set_title(title)
            else:
                ax[1, index].tick_params(colors=(0, 0, 0, 0))
        ax[0, 0].set_ylabel('Training', fontsize=16)
        ax[0, 0].yaxis.labelpad = 16
        ax[1, 0].set_ylabel('Testing', fontsize=16)
        ax[1, 0].yaxis.labelpad = 16
        fig.tight_layout(rect=[0, 0, 1, .94])
        fig.savefig(filepath)

    def _select_metrics(self, tables):
        if not self._metrics:
            return natural_sorted(tables.items(), key=lambda x: x[0])
        selected = []
        for metric in self._metrics:
            matches = [x for x in tables.keys() if metric in x]
            if not matches:
                print("  '{}' matches no metrics.".format(metric))
                continue
            if len(matches) > 1:
                print("  '{}' matches multiple metrics.".format(metric))
            selected += [(x, tables[x]) for x in matches]
        return selected

    def _collect_stats(self, engine, table):
        result = engine.execute(sql.select([table]))
        columns = np.array([x for x in result]).T
        if not len(columns) or not columns.shape[1]:
            return None
        id_, timestamp, step, epoch, training, episode = columns[:6]
        id_ = [int(x, 16) for x in id_]
        order = np.lexsort([id_, step, episode, training, epoch])
        rows = columns.T
        rows = rows[order]
        return rows

    def _plot(self, ax, rows):
        _, _, _, epoch, training, _ = rows.T[:6]
        values = rows[:, 6:].astype(float)
        categorical = np.allclose(values, values.astype(int))
        categorical = categorical and np.unique(values.astype(int)).size <= 10
        if values.shape[1] == 1 and not categorical:
            self._plot_scalar(ax, values[:, 0], epoch.max())
        elif values.shape[1] == 1:
            indices = values[:, 0].astype(int)
            histograms = np.eye(indices.max() + 1)[indices]
            self._plot_distribution(ax, histograms, epoch.max())
        elif values.shape[1] > 1:
            self._plot_counts(ax, values, epoch.max())

    def _subplots(self, rows, cols, **kwargs):
        size = [4 * cols, 3 * rows]
        fig, ax = plt.subplots(ncols=cols, nrows=rows, figsize=size, **kwargs)
        if cols == 1:
            ax = np.array([ax]).T
        return fig, ax

    def _get_tables(self, engine):
        metadata = sql.MetaData()
        metadata.reflect(engine)
        return metadata.tables

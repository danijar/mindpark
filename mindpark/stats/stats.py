import os
import collections
import matplotlib.pyplot as plt
import numpy as np
import sqlalchemy as sql
# from mindpark.plot.scatter import Scatter
from mindpark.utility import get_subdirs, natural_sorted, read_yaml


Run = collections.namedtuple(
    'Run', 'experiment name env algorithm repeat stats')


class Metrics:

    def __init__(self, type_):
        self._type = type_

    def __call__(self, experiment):
        for run in self._collect_runs(experiment):
            self._process(run)

    def _collect_runs(self, experiment):
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
            run.algorithm.title(), run.env.title(), run.repeat)
        filepath = '{}-{}-{}-{}.{}'.format(
            run.name, run.env, run.algorithm, run.repeat, self._type)
        filepath = os.path.join(run.experiment, filepath.lower())
        self._figure(run.stats, title, filepath)

    def _figure(self, stats, title, filepath):
        engine = sql.create_engine('sqlite:///{}'.format(stats))
        tables = self._get_tables(engine)
        fig, ax = self._subplots(len(tables))
        fig.suptitle(title, fontsize=16)
        tables = natural_sorted(tables.items(), key=lambda x: x[0])
        for axes, (title, table) in zip(ax, tables):
            axes.set_title(title)
            rows = self._collect_stats(engine, table)
            print('Plot', title)
            if rows is None:
                print('No rows.')
                continue
            self._plot(axes, rows)
        fig.tight_layout(rect=[0, 0, 1, .94])
        fig.savefig(filepath)

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
        domain = np.linspace(0, epoch.max(), len(values))
        if values.shape[1] == 1:  # Scalar
            value = values[:, 0]
            ax.scatter(domain, value, c=training, alpha=0.1, lw=0)
        elif np.allclose(values, values.astype(int)):
            values = values.astype(int)
            # TODO: Grid image.
            pass

    def _subplots(self, amount, **kwargs):
        cols, rows = 3, int(np.ceil(amount / 3))
        size = [4 * cols, 3 * rows]
        fig, ax = plt.subplots(ncols=cols, nrows=rows, figsize=size, **kwargs)
        if not hasattr(ax, '__len__'):
            ax = [ax]
        if hasattr(ax[0], '__len__'):
            ax = [y for x in ax for y in x]
        return fig, ax

    def _get_tables(self, engine):
        metadata = sql.MetaData()
        metadata.reflect(engine)
        return metadata.tables

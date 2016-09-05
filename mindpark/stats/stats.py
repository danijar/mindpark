import os
import collections
import numpy as np
import sqlalchemy as sql
from mindpark.stats.metrics import Metrics
from mindpark.utility import natural_sorted
from mindpark.utility import get_subdirs, read_yaml


Run = collections.namedtuple(
    'Run', 'experiment name env algorithm repeat stats')


class Stats:

    def __init__(self, type_, selectors=None):
        self._type = type_
        self._selectors = selectors
        self._plot_metrics = Metrics(type_)

    def __call__(self, experiment):
        # TODO: Plot scores here.
        for run in self._collect_runs(experiment):
            self._process_run(run)

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

    def _process_run(self, run):
        title = '{} on {} (Repeat {})'.format(
            run.algorithm, run.env, run.repeat)
        print(' Plot run', title)
        metrics = list(self._collect_metrics(run))
        if not metrics:
            print('  No metrics found.')
            return
        filepath = '{}-{}-{}-{}.{}'.format(
            run.name, run.env, run.algorithm, run.repeat, self._type)
        filepath = os.path.join(run.experiment, filepath)
        self._plot_metrics(metrics, title, filepath)

    def _collect_metrics(self, run):
        engine = sql.create_engine('sqlite:///{}'.format(run.stats))
        metadata = sql.MetaData()
        metadata.reflect(engine)
        for name in self._select_metrics(metadata.tables.keys()):
            rows = self._collect_rows(engine, metadata.tables[name])
            if rows is None:
                continue
            yield name, rows

    def _select_metrics(self, metrics):
        if not self._selectors:
            return natural_sorted(metrics)
        selected = []
        for metric in self._selectors:
            matches = [x for x in metrics if metric in x]
            if not matches:
                print("  '{}' matches no metrics.".format(metric))
                continue
            if len(matches) > 1:
                print("  '{}' matches multiple metrics.".format(metric))
            selected += matches
        return selected

    def _collect_rows(self, engine, table):
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

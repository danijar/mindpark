import os
import numpy as np
import sqlalchemy as sql
from mindpark.utility import natural_sorted


class Metric(dict):

    def __getattr__(self, key):
        if key not in self:
            raise AttributeError("unknown key '{}'".format(key))
        return self[key]

    def __setattr__(self, key, value):
        if key not in self:
            raise AttributeError("unknown key '{}'".format(key))
        self[key] = value

    def __getitem__(self, key):
        if isinstance(key, str):
            return super().__getitem__(key)
        return Metric({k: v[key] for k, v in self.items()})


class Reader:

    """
    Read a stats database file and iterate over its metrics. Optionally,
    metrics and their order can be selected by partial names.
    """

    def __init__(self, selectors=None):
        self._selectors = selectors

    def __call__(self, filename):
        assert os.path.isfile(filename)
        engine = sql.create_engine('sqlite:///{}'.format(filename))
        metadata = sql.MetaData()
        metadata.reflect(engine)
        for name in self._select_metrics(metadata.tables.keys()):
            columns = self._collect_columns(engine, metadata.tables[name])
            if columns is None:
                continue
            yield name, columns

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

    def _collect_columns(self, engine, table):
        result = engine.execute(sql.select([table]))
        columns = np.array([x for x in result]).T
        if not len(columns) or not columns.shape[1]:
            return None
        columns = Metric(
            id=np.array([int(x, 16) for x in columns[0]]),
            timestamp=columns[1],
            step=columns[2].astype(int),
            epoch=columns[3].astype(int),
            training=columns[4].astype(bool),
            episode=columns[5].astype(int),
            data=columns[6:].T.astype(float))
        columns = self._sort_columns(columns)
        return columns

    def _sort_columns(self, columns):
        c = columns
        order = np.lexsort([c.id, c.step, c.episode, c.training, c.epoch])
        columns = columns[order]
        return columns

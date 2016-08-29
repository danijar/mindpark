import pytest
import sqlalchemy as sql
from mindpark.core import Metric
from test.fixtures import *


@pytest.fixture(params=[1, 2, ['first'], ['first', 'second']])
def metric(request, task):
    return Metric(task, 'metric', request.param)


class TestMetric:

    def test_meta_data_from_task(self, task, metric):
        values = [0.0] * len(metric.columns)
        reference = []
        while task.epoch < task.epochs:
            for _ in range(10):
                reference.append((task.epoch.value, task.training))
                metric(*values)
            task.epoch.increment()
        metric.flush()
        rows = self._select_all(task.directory, metric.name)
        actual = [(x['epoch'], x['training']) for x in rows]
        assert actual == reference

    def test_correct_values(self, task, metric):
        reference = []
        for index in range(10):
            values = [index + i for i, _ in enumerate(metric.columns)]
            reference.append(values)
            metric(*values)
        metric.flush()
        rows = self._select_all(task.directory, metric.name)
        actual = [[x[y] for y in metric.columns] for x in rows]
        assert actual == reference

    def test_need_at_least_one_column(self, task):
        with pytest.raises(ValueError):
            Metric(task, 'metric', 0)
        with pytest.raises(ValueError):
            Metric(task, 'metric', [])

    def test_need_distinct_column_names(self, task):
        with pytest.raises(KeyError):
            Metric(task, 'metric', ['foo', 'foo'])

    def test_cannot_use_reserved_column(self, task):
        with pytest.raises(KeyError):
            Metric(task, 'metric', ['id'])
        with pytest.raises(KeyError):
            Metric(task, 'metric', ['step', 'foo'])
        with pytest.raises(KeyError):
            Metric(task, 'metric', ['foo', 'step', 'foo'])

    @staticmethod
    def _select_all(directory, table):
        filepath = 'sqlite:///{}/stats.db'.format(directory.rstrip('/'))
        engine = sql.create_engine(filepath)
        meta = sql.MetaData()
        table = sql.Table(table, meta, autoload=True, autoload_with=engine)
        rows = engine.execute(sql.select([table]))
        return rows

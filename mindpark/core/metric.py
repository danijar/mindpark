from datetime import datetime
import time
import uuid
import sqlalchemy as sql
from mindpark.utility import Uuid


class Metric:

    def __init__(self, task, name, columns, flush_interval=5):
        self.columns = self._parse_columns(columns)
        self.name = name
        self._task = task
        self._flush_interval = flush_interval
        self._last_flush = time.time()
        self._buffer = []
        self._engine = self._get_engine()
        self._table = self._create_table(self.columns)
        self._insert = self._table.insert()

    def __call__(self, *values):
        values = self._parse_values(values)
        row = dict(
            step=self._task.step.value,
            epoch=self._task.epoch.value,
            training=self._task.training,
            episode=self._task.episode.value)
        row.update(values)
        self._buffer.append(row)
        if time.time() >= self._last_flush + self._flush_interval:
            self.flush()

    def flush(self):
        if not self._buffer:
            return
        self._last_flush = time.time()
        insert = self._insert.values()
        with self._engine.begin() as connection:
            connection.execute(insert, self._buffer)
        self._buffer = []

    def _parse_columns(self, columns):
        if not isinstance(columns, (int, list, tuple)):
            raise ValueError('columns must be a number or list of names')
        if isinstance(columns, int):
            columns = ['value_{}'.format(x) for x in range(columns)]
        if len(set(columns)) != len(columns):
            raise KeyError('column names must be unique')
        reserved = 'id timestamp step epoch training episode'.split()
        for column in columns:
            if column in reserved:
                message = "can't use reserved column '{}'"
                raise KeyError(message.format(column))
        if not columns:
            raise ValueError('need at least one column')
        return columns

    def _parse_values(self, values):
        if len(values) != len(self.columns):
            message = 'need one value for each column, expected {} got {}'
            raise ValueError(message.format(len(self.columns), len(values)))
        values = [float(x) for x in values]
        values = {k: v for k, v in zip(self.columns, values)}
        return values

    def _create_table(self, names):
        metadata = sql.MetaData()
        columns = [
            sql.Column('id', Uuid, primary_key=True, default=uuid.uuid4),
            sql.Column('timestamp', sql.DateTime, default=datetime.now),
            sql.Column('step', sql.Integer),
            sql.Column('epoch', sql.Integer),
            sql.Column('training', sql.Boolean),
            sql.Column('episode', sql.Integer)]
        columns += [sql.Column(x, sql.Float) for x in names]
        table = sql.Table(self.name, metadata, *columns)
        metadata.create_all(self._engine)
        return table

    def _get_engine(self):
        if not self._task.directory:
            kwargs = dict(
                connect_args={'check_same_thread': False},
                poolclass=sql.pool.StaticPool)
            return sql.create_engine('sqlite://', **kwargs)
        filepath = self._task.directory.rstrip('/')
        return sql.create_engine('sqlite:///{}/stats.db'.format(filepath))

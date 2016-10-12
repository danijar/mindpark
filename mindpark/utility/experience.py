import os
import numpy as np
from mindpark.utility.other import ensure_directory


class Experience:

    def __init__(self, maxlen, shapes):
        self._maxlen = maxlen
        self._random = np.random.RandomState(seed=0)
        self._columns = [np.zeros((self._maxlen,) + x) for x in shapes]
        self._index = 0

    def __len__(self):
        if not self._columns:
            return 0
        return min(self._index, self._maxlen)

    @property
    def nbytes(self):
        return sum(x.nbytes for x in self._columns)

    def append(self, transition):
        assert len(transition) == 4
        for column, entry in zip(self._columns, transition):
            if entry is None:
                entry = np.empty(column.shape[1:])
                entry.fill(np.nan)
            entry = np.array(entry)
            if entry.shape != column.shape[1:]:
                message = 'experience shape {} does not match previous {}'
                raise ValueError(message.format(entry.shape, column.shape[1:]))
            column[self._index % self._maxlen] = entry
        self._index += 1

    def save(self, filepath):
        ensure_directory(os.path.dirname(filepath))
        names = ('states', 'actions', 'rewards', 'successors')
        data = {k: np.array(v) for k, v in zip(names, self._columns)}
        np.savez_compressed(filepath, data)

    def sample(self, amount):
        if amount > len(self):
            raise RuntimeError('not enough transitions to sample from')
        choices = self._random.choice(len(self), amount, replace=False)
        return (x[choices] for x in self._columns)

    def access(self):
        used = [x[:self._index] for x in self._columns]
        states, actions, rewards, successors = used
        return states, actions, rewards, successors

    def clear(self):
        self._index = 0

    def log_memory_size(self):
        print('Replay memory size', round(self.nbytes / (1024 ** 3), 2), 'GB')

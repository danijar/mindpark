import numpy as np
from mindpark.utility.other import ensure_directory


class Experience:

    def __init__(self, maxlen):
        self._maxlen = maxlen
        self._random = np.random.RandomState(seed=0)
        self._transitions = None
        self._index = None

    def __len__(self):
        if not self._transitions:
            return 0
        return min(self._index, self._maxlen)

    @property
    def nbytes(self):
        return sum(x.nbytes for x in self._transitions)

    def append(self, transition):
        assert len(transition) == 4
        if not self._transitions:
            self._initialize(transition)
        for store, example in zip(self._transitions, transition):
            store[self._index % self._maxlen] = example
        self._index += 1

    def save(self, filepath):
        ensure_directory(os.path.dirname(filepath))
        names = ('states', 'actions', 'rewards', 'successors')
        data = {k: np.array(v) for k, v in zip(names, self._transitions)}
        np.savez_compressed(filepath, data)

    def sample(self, amount):
        if amount > len(self):
            raise RuntimeError('not enough transitions to sample from')
        choices = self._random.choice(len(self), amount, replace=False)
        return (x[choices] for x in self._transitions)

    def access(self):
        used = [x[:self._index] for x in self._transitions]
        states, actions, rewards, successors = used
        return states, actions, rewards, successors

    def clear(self):
        self._index = 0

    def _initialize(self, transition):
        shapes = [x.shape if hasattr(x, 'shape') else tuple()
                  for x in transition]
        shapes = [(self._maxlen,) + x for x in shapes]
        self._transitions = [np.empty(x) for x in shapes]
        self._index = 0

import numpy as np


class RingBuffer:

    """
    Ring buffer holding tuples of Numpy matrices that are stored column wise.
    Supports advanced slicing and sliced assignment. Converts None values to
    arrays of the target column size holding nan values.
    """

    def __init__(self, capacity, shapes):
        self._capacity = int(capacity)
        self._shapes = tuple(tuple(x) for x in shapes)
        self._buffers = tuple(np.zeros((int(capacity),) + x)
                              for x in self._shapes)
        self._head = 0
        self._tail = 0

    @property
    def head(self):
        """
        Index of the first element hold in the buffer. Inclusive lower bound
        for slicing.
        """
        return self._head

    @property
    def tail(self):
        """
        Index of the next element to be inserted. Exclusive upper bound for
        slicing.
        """
        return self._tail

    def push(self, *transition):
        assert len(transition) == len(self._buffers)
        for element, shape in zip(transition, self._shapes):
            if element is not None:
                element = np.array(element)
                assert element.shape == shape
        for element, buffer in zip(transition, self._buffers):
            buffer[self.tail % self._capacity] = element
        self._tail += 1
        self._head = max(self.head, self.tail - self._capacity)

    def clear(self):
        self._head = 0
        self._tail = 0

    def log_memory_size(self):
        nbytes = sum(x.nbytes for x in self._buffers)
        print('Replay memory size', round(nbytes / (1024 ** 3), 2), 'GB')

    def __len__(self):
        return self.tail - self.head

    def __getitem__(self, key):
        key = self._wrap_key(key)
        return [np.array(x[key], dtype=x.dtype) for x in self._buffers]

    def __setitem__(self, key, transition):
        transition = [np.array(x) for x in transition]
        key = self._wrap_key(key)
        for shape, buffer in zip(self._shapes, self._buffers):
            if transition is None:
                transition = self._nans(shape)
            buffer[key] = transition

    def _wrap_key(self, key):
        if isinstance(key, slice):
            return self._wrap_slice(key)
        if isinstance(key, int):
            return self._wrap_index(int(key))
        if hasattr(key, '__iter__'):
            return [self._wrap_index(x) for x in key]

    def _wrap_slice(self, key):
        # Default slice values.
        start = self._head if key.start is None else key.start
        stop = self._tail if key.stop is None else key.stop
        step = 1 if key.step is None else key.step
        # Negative indices are relative to the end.
        start = self._tail + start if start < 0 else start
        stop = self._tail + stop if stop < 0 else stop
        # Use vector based indexing for easy wrapping.
        indices = range(start, stop, step)
        indices = [self._wrap_index(x) for x in indices]
        return indices

    def _wrap_index(self, key):
        key = self.tail + key if key < 0 else key
        if not (self.head <= key <= self.tail):
            message = 'Index {} must be in range {} to {}.'
            raise IndexError(message.format(key, self.head, self.tail))
        if key == self.tail:
            return (key - 1) % self._capacity + 1
        return key % self._capacity

    def _nans(self, shape):
        array = np.empty(shape)
        array.fill(np.nan)
        return array


class Sequential(RingBuffer):

    """
    Replay buffer with first in first out behavior. Obtaining elements frees
    them from the buffer. Exceeding the capacity also frees the oldest
    elements.
    """

    def __init__(self, capacity, shapes, random=None):
        super().__init__(capacity, shapes)
        self._random = random or np.random.RandomState()

    def batch(self, amount):
        if amount > len(self):
            raise RuntimeError('Not enough elements to form batch.')
        batch = self[self.head: self.head + amount]
        self._head += amount
        return batch

    def shuffle(self):
        if not len(self):
            return
        order = self._head + self._random.permutation(len(self))
        order = self._wrap_key(order)
        for buffer in self._buffers:
            buffer[:] = buffer[order]


class Random(RingBuffer):

    """
    Replay buffer where elements are obtained uniformly at random from the
    currently hold elements. Exceeding the capacity frees the oldest elements.
    """

    def __init__(self, capacity, shapes, random=None, replace=False):
        super().__init__(capacity, shapes)
        self._random = random or np.random.RandomState()
        self._replace = replace

    def batch(self, amount):
        if not self._replace and amount > len(self):
            message = "Can't sample {} from {} transitions without replacement"
            raise RuntimeError(message.format(amount, len(self)))
        selection = self._random.choice(len(self), amount, self._replace)
        return self[self._head + selection]

import functools
from threading import Lock


@functools.total_ordering
class Counter:

    def __init__(self):
        self._value = 0
        self._initial = True
        self._lock = Lock()

    @property
    def value(self):
        return self._value

    def increment(self):
        with self._lock:
            if self._initial:
                self._initial = False
            else:
                self._value += 1
            return self.value

    def __repr__(self):
        return str(self.value)

    def __eq__(self, other):
        return self.value == other

    def __lt__(self, other):
        return self.value < other

    def __add__(self, other):
        return self.value + other

    def __radd__(self, other):
        return self.value + other

    def __sub__(self, other):
        return self.value - other

    def __rsub__(self, other):
        return other - self.value

    def __truediv__(self, other):
        return self.value / other

    def __rtruediv__(self, other):
        return other / self.value

    def __int__(self):
        return self._value

    def __bool__(self):
        return bool(self.value)

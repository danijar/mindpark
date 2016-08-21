from threading import Lock


class Counter:

    def __init__(self):
        self._value = None
        self._lock = Lock()

    @property
    def value(self):
        return self._value

    def increment(self):
        with self._lock:
            if self._value is None:
                self._value = 0
            else:
                self._value += 1
            return self._value

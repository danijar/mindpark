class Every:

    def __init__(self, every):
        self._every = every
        self._last = None

    def __call__(self, current):
        if self._last is None:
            self._last = current
        if current - self._last < self._every:
            return False
        self._last += self._every
        return True

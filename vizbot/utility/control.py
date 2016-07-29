class Every:

    def __init__(self, every):
        self._every = int(float(every))
        self._last = None

    def __call__(self, timestep):
        if self._last is None:
            self._last = timestep
        if timestep - self._last < self._every:
            return False
        self._last += self._every
        return True


class Decay:

    def __init__(self, start, stop, steps):
        self._start = start
        self._stop = stop
        self._steps = int(float(steps))
        assert self._start >= self._stop
        assert self._steps

    def __call__(self, timestep):
        progress = min(timestep, self._steps) / self._steps
        mixed = (1 - progress) * self._start + progress * self._stop
        return mixed


class Statistic:

    def __init__(self, template, every=10000):
        self._template = template
        self._every = int(float(every))
        self._values = []

    def __call__(self, value):
        self._values.append(value)
        if len(self._values) > self._every:
            average = sum(self._values) / len(self._values)
            print(self._template.format(average))
            self._values = []

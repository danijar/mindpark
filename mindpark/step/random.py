from mindpark.core import Policy


class Random(Policy):

    def observe(self, observ):
        super().observe(observ)
        return self.task.actions.sample()

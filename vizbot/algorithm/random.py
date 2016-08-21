from vizbot.core import Algorithm, Policy


class Random(Algorithm, Policy):

    def __init__(self, task, config):
        Algorithm.__init__(self, task, config)
        Policy.__init__(self, self.task)

    def observe(self, observ):
        super().observe(observ)
        return self.task.actions.sample()

    def receive(self, reward, final):
        super().receive(reward, final)

    @property
    def policy(self):
        return self

from vizbot.core import Algorithm, Policy


class Random(Algorithm, Policy):

    def __init__(self, task, config):
        Algorithm.__init__(self, task, config)
        Policy.__init__(self, self.task.interface)

    def step(self, observation):
        return self.actions.sample()

    @property
    def policy(self):
        return self

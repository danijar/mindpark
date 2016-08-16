from vizbot.core import Algorithm, Policy


class Random(Algorithm, Policy):

    def __init__(self, task, config):
        Algorithm.__init__(self, task, config)
        Policy.__init__(self, self.task.interface)

    def observe(self, observation):
        super().observe(observation)
        return self.actions.sample()

    def receive(self, reward, final):
        super().receive(reward, final)

    @property
    def policy(self):
        return self

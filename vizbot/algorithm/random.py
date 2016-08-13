from vizbot.core import Policy


class Random(Policy):

    def step(self, observation):
        super().step(observation)
        return self.actions.sample()

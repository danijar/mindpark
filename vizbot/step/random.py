from vizbot.core import Policy


class Random(Policy):

    def observe(self, observation):
        super().observe(observation)
        return self.actions.sample()

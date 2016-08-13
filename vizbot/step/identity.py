from vizbot.core import Policy


class Identity(Policy):

    @property
    def interface(self):
        return self.observations, self.actions

    def step(self, observation):
        super().step(observation)
        return self.above.step(observation)

    def experience(self, *transition):
        super().experience(*transition)
        self.above.experience(*transition)

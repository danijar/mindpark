from vizbot.core import Step


class Identity(Step):

    def step(self, observation):
        super().step(observation)
        return self.above.step(observation)

    def experience(self, *transition):
        super().experience(*transition)
        self.above.experience(*transition)

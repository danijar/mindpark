from vizbot.core import Policy


class Identity(Policy):

    @property
    def interface(self):
        return self.observations, self.actions

    def observe(self, observation):
        super().observe(observation)
        action = self.above.observe(observation)
        assert self.actions.contains(action)
        return action

    def receive(self, reward, final):
        super().receive(reward, final)
        self.above.receive(reward, final)

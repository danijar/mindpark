from vizbot.core import Policy


class ClampReward(Policy):

    @property
    def interface(self):
        return self.observations, self.actions

    def observe(self, observation):
        super().observe(observation)
        return self.above.observe(observation)

    def receive(self, reward, final):
        super().receive(reward, final)
        reward = max(0, min(reward, 1))
        self.above.receive(reward, final)

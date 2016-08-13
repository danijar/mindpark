from vizbot.core import Policy, Input


class FrameSkip(Policy):

    def __init__(self, observations, actions, amount):
        super().__init__(observations, actions)
        self._amount = amount
        self._action = None

    @property
    def interface(self):
        return self.observations, self._actions

    def observe(self, reward, observation):
        super().observe(reward, observation)
        if self.timestep % self._amount == 0:
            raise Input(reward, observation)
        return self._action

    def perform(self, action):
        super().perform(action)
        self._action = action
        return action

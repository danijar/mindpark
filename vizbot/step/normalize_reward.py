import numpy as np
from vizbot.core import Policy


class NormalizeReward(Policy):

    @property
    def interface(self):
        return self.observations, self.actions

    def step(self, observation):
        super().step(observation)
        return self.above.step(observation)

    def experience(self, observation, action, reward, successor):
        super().experience(observation, action, reward, successor)
        reward = np.sign(reward)
        self.above.experience(observation, action, reward, successor)

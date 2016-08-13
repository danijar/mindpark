import numpy as np
from gym.spaces import Box
from vizbot.core import Policy


class Normalize(Policy):

    @property
    def interface(self):
        low = np.zeros(self.observations.shape)
        high = np.ones(self.observations.shape)
        return Box(low, high), self.actions

    def step(self, observation):
        super().step(observation)
        low, high = self.observations.low, self.observations.high
        observation = (observation.astype(float) - low) / high
        # low, high = self.interface[0].low, self.interface[0].high
        # observation = max(low, max(observation, high))
        return self.above.step(observation)

    def experience(self, *transition):
        super().experience(*transition)
        self.above.experience(*transition)

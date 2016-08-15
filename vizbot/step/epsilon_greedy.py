from vizbot.core import Policy
from vizbot.utility import Decay


class EpsilonGreedy(Policy):

    def __init__(self, interface,
                 from_=1, to=0.1, test=0.05, over=100, offset=0):
        super().__init__(interface)
        self._offset = offset
        self._test = test
        self._probability = Decay(from_, to, over)

    @property
    def interface(self):
        return self.observations, self.actions

    def step(self, observation):
        super().step(observation)
        if self.training:
            timestep = max(0, self.timestep - self._offset)
            epsilon = self._probability(timestep)
        else:
            epsilon = self._test
        if self.random.rand() < epsilon:
            return self.actions.sample()
        return self.above.step(observation)

    def experience(self, *transition):
        super().experience(*transition)
        self.above.experience(*transition)

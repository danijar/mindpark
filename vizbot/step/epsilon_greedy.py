from vizbot.core import Policy
from vizbot.utility import Decay


class EpsilonGreedy(Policy):

    def __init__(self, interface,
                 start=1, stop=0.1, over=100, test=0.1, after=0):
        super().__init__(interface)
        self._after = after
        self._test = test
        self._probability = Decay(start, stop, over)

    @property
    def interface(self):
        return self.observations, self.actions

    def step(self, observation):
        super().step(observation)
        if self.training:
            timestep = max(0, self.timestep - self._after)
            epsilon = self._probability(timestep)
        else:
            epsilon = self._test
        if self.random.rand() < epsilon:
            return self.actions.sample()
        return self.above.step(observation)

    def experience(self, *transition):
        super().experience(*transition)
        self.above.experience(*transition)

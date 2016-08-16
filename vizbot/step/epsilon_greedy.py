from vizbot.core import Policy
from vizbot.utility import Decay


class EpsilonGreedy(Policy):

    # TODO: Forward all observations and rewards, but do not let the next
    # policy to heavy work when not neccessary. How to do that?

    def __init__(self, interface,
                 from_=1, to=0.1, test=0.05, over=100, offset=0):
        super().__init__(interface)
        self._offset = offset
        self._test = test
        self._probability = Decay(from_, to, over)

    @property
    def interface(self):
        return self.observations, self.actions

    def observe(self, observation):
        super().observe(observation)
        if self.training:
            step = max(0, self.step - self._offset)
            epsilon = self._probability(step)
        else:
            epsilon = self._test
        if self.random.rand() < epsilon:
            return self.actions.sample()
        return self.above.observe(observation)

    def receive(self, reward, final):
        super().receive(reward, final)
        self.above.receive(reward, final)

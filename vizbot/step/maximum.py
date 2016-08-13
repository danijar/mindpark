from vizbot.core import Policy
from vizbot.step.history import History


class Maximum(History):

    """
    Running maximum over the specified number of previous the observations.
    """

    def __init__(self, interface, amount=2):
        super().__init__(interface, amount)

    @property
    def interface(self):
        return self.observations, self.actions

    def step(self, observation):
        Policy.step(self, observation)
        self._push(observation)
        observation = self._history().max(-1)
        return self.above.step(observation)

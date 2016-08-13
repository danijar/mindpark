from gym.spaces import Box
from vizbot.core import Policy


class Grayscale(Policy):

    """
    Convert observations to grayscale, dropping their last dimension. The
    default weighting of the RGB channels extracts the luminance.
    """

    def __init__(self, interface, weighting=(0.299, 0.587, 0.114)):
        super().__init__(interface)
        if len(weighting) != self.observations.shape[-1]:
            raise ValueError('weighting must match last axis of observations')
        self._weighting = weighting

    @property
    def interface(self):
        low = self._apply(self.observations.low)
        high = self._apply(self.observations.high)
        return Box(low, high), self.actions

    def step(self, observation):
        super().step(observation)
        observation = self._apply(observation)
        return self.above.step(observation)

    def experience(self, *transition):
        super().experience(*transition)
        self.above.experience(*transition)

    def _apply(self, observation):
        return (self._weighting * observation).sum(-1)

from vizbot.core import Agent
from vizbot.preprocess import Downsample


class Noop(Agent):

    def perform(self, state):
        super().perform(state)
        return self._noop()

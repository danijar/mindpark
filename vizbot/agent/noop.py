from vizbot.core import Agent


class Noop(Agent):

    def perform(self, state):
        super().perform(state)
        return self._noop()

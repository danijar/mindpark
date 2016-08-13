from vizbot.core import Agent


class Noop(Agent):

    def step(self, state):
        return self._noop()

from vizbot.core import Agent


class Noop(Agent):

    def step(self, state):
        super().step(state)
        return self._noop()

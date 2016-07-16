from vizbot.core import Agent


class Random(Agent):

    def step(self, state):
        super().step(state)
        return self._actions.sample()

from vizbot.core import Agent


class Random(Agent):

    def perform(self, state):
        super().step(state)
        return self._env.actions.sample()

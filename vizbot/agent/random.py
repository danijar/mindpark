from vizbot.core import Agent


class Random(Agent):

    def perform(self, state):
        super().perform(state)
        return self._env.actions.sample()

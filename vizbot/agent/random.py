from vizbot.core import Agent


class Random(Agent):

    def step(self, state):
        return self.actions.sample()

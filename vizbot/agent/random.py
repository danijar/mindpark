from vizbot.core import Agent


class Random(Agent):

    def __init__(self, actions, states):
        self.__actions = actions
        super().__init__(actions, states)

    def step(self, state):
        super().step(state)
        return self.__actions.sample()

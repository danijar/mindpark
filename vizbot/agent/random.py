from vizbot.core import Agent


class Random(Agent):

    def __init__(self, env):
        self._env = env

    def __call__(self, reward, state):
        return self._env.action_space.sample()

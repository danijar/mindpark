from vizbot.core.agent import Agent
from vizbot.core.env import Env


class Preprocess(Env, Agent):
    """
    Wrap environments and change their state and action spaces. Get triggered
    by self._source and should forward most actions to self._agent.
    """

    def __init__(self, env):
        Env.__init__(self)
        Agent.__init__(self, env)

    def start(self):
        Env.start(self)
        Agent.start(self)

    def stop(self):
        Env.stop(self)
        Agent.stop(self)

from vizbot.core.agent import Agent
from vizbot.core.env import Env


class Preprocess(Agent, Env):
    """
    Wrap environments and change their state and action spaces. Get triggered
    by self._source and should forward most actions to self._agent.
    """

    def __init__(self, env):
        Agent.__init__(self, env)
        Env.__init__()
        self._env.register(self)

from vizbot.core.agent import Agent
from vizbot.core.env import Env


class Preprocess(Env, Agent):
    """
    Wrap environments and change their state and action spaces. A preprocessing
    step is handled like an agent from the outside. It is responsibility of the
    subclass to treat the agent according to its interface and call
    self._agent.perform(state) and self._agent.feedback().
    """

    def __init__(self, env):
        Env.__init__(self)
        Agent.__init__(self, env)

    def start(self):
        Env.start(self)
        Agent.start(self)

    def stop(self):
        Agent.stop(self)
        Env.stop(self)

from vizbot.core.env import Env


class Preprocess(Env):
    """
    Wrap an environment to change their states, actions, or behavior.
    Typically, methods calls should be forwarded to the inner environment.
    """

    def __init__(self, env):
        self._env = env

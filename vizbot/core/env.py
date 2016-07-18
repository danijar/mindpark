class StopEpisode(Exception):

    def __init__(self, env):
        super().__init__(self)
        self.env = env


class Env:

    @property
    def states(self):
        raise NotImplementedError

    @property
    def actions(self):
        raise NotImplementedError

    def reset(self):
        """
        Initialize the environment from new or used state. Return an initial
        state.
        """
        raise NotImplementedError

    def step(self, action):
        """
        Apply the action and simulate one time step in the environment. Return
        the new state and reward. Raise StopEpisode(self) when the episode
        ends.
        """
        raise NotImplementedError

    def close(self):
        """
        Optional hook for cleanup before the object gets destoyed.
        """
        pass


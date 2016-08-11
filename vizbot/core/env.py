class Env:

    @property
    def observations(self):
        """
        The space of observations.
        """
        raise NotImplementedError

    @property
    def actions(self):
        """
        The space of actions.
        """
        raise NotImplementedError

    def reset(self):
        """
        Initialize or reinitialize the environment. Return an initial
        observation.
        """
        raise NotImplementedError

    def step(self, action):
        """
        Apply the action and simulate one time step in the environment. Return
        the next observation and reward. May return None as observation to stop
        the episode.
        """
        raise NotImplementedError

    def close(self):
        """
        Optional hook for cleanup before the object gets destoyed.
        """
        pass

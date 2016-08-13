class Env:

    @property
    def interface(self):
        """
        A tuple of the observation space and the action space.
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

class Policy:
    """
    A possibly partial behavior to interact with an environment. This is the
    base class for policies, sequential policies, and preprocesses.
    """

    def __init__(self, observations, actions):
        """
        Construct a new policy to act in the observation and action space. It
        is the respondibility of the caller to update the `training` and
        `timestep` properties.
        """
        self.observations = observations
        self.actions = actions
        self.training = None
        self.timestep = -1

    @property
    def interface(self):
        """
        The spaces of forwarded observations and expected actions. Must be
        implemented if forwarding observations to the next behavior.
        """
        raise NotImplementedError

    def begin_episode(self, training):
        """
        Notify the begin of an episode.
        """
        self.training = training

    def end_episode(self):
        """
        Notify the end of an episode.
        """
        self.training = None

    def observe(self, reward, observation):
        """
        Receive a reward and an observation from the previous behavior. Return
        an action or raise a tuple of a reward and an observation to process by
        the next behavior. The first reward and last observation of an episode
        are None. Do not call this method directly.
        """
        if self.training:
            self.timestep += 1

    def perform(self, action):
        """
        Receive an action from the next behavior. Return an action or raise a
        tuple of a reward and an observation to process by the next behavior.
        """
        return action

    def experience(self, observation, action, reward, successor):
        """
        Optional hook to process the transitions. Successor is None when the
        episode ended after the reward. All other arguments are never None.
        """
        pass

class Policy:

    """
    A behavior to interact with an environment. Call super when overriding
    methods.
    """

    def __init__(self, interface):
        """
        Construct a policy that interacts with an environment using the
        specified interface.
        """
        observations, actions = interface
        self.observations = observations
        self.actions = actions
        self.training = None
        self.timestep = None

    def begin_episode(self, training):
        """
        Optional hook at the beginning of an episode. The `training` parameter
        specifies whether the algorithm should learn during the episode or the
        algortihm is just evaluated.
        """
        self.training = training

    def end_episode(self):
        """
        Optional hook at the end of an episode.
        """
        self.training = None

    def step(self, observation):
        """
        Receive an observation from the environment an choose an action to
        perform next.
        """
        assert self.observations.contains(observation)
        if self.timestep is None:
            self.timestep = 0
        elif self.training:
            self.timestep += 1

    def experience(self, observation, action, reward, successor):
        """
        Optional hook to process the current transition. Successor is None when
        the episode ended after the reward. All other arguments are never None.
        """
        assert all(x is not None for x in (observation, action, reward))

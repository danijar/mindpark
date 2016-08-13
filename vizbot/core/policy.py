class Policy:

    """
    A behavior to interact with an environment. Call super when overriding
    methods.

    The behavior can be partial and delegate decisions to the next policy. In
    this case, override `interface` and use `above` to access the next policy.
    Consider forwarding calls to `step()` and `experience()` to the next
    policy.
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
        self.above = None

    @property
    def interface(self):
        """
        The interface for higher policies to interact with this one. The
        interface is a tuple of the observation and the action space. For final
        steps, the interface is None.
        """
        return None

    def set_above(self, above):
        """
        The above policy will be set through this setter. Its observation and
        action spaces will match the this policy's interface.
        """
        if self.interface is None:
            raise RuntimeError('cannot set above of final policy')
        assert self.interface[0] == above.observations
        assert self.interface[1] == above.actions
        self.above = above

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
        When overriding, do not forget to forward the experience of terminal
        states, where `successor` is None.
        """
        assert all(x is not None for x in (observation, action, reward))

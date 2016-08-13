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
        self.timestep = None

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
        an action or raise an Input object to forward to the next behavior. The
        first reward and last observation of an episode are None.
        """
        if self.timestep is None:
            self.timestep = 0
        elif self.training:
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


class Input(Exception):

    def __init__(self, reward, observation):
        if reward is None and observation is None:
            raise ValueError('must specify reward or observation')
        if reward is not None and not isinstance(reward, (float, int)):
            raise ValueError('reward must be a number')
        self.reward = reward
        self.observation = observation


class ExperienceProxy:
    """
    Wrapper for the policy class that collects transition tuples and calls the
    experience method automatically.
    """

    def __init__(self, policy):
        if isinstance(policy, ExperienceProxy):
            raise ValueError('policy is already wrapped')
        self._policy = policy
        self._last_observation = None
        self._last_action = None

    def __getattr__(self, name):
        return getattr(self._policy, name)

    def observe(self, reward, observation):
        if reward is not None:
            assert self._last_observation is not None
            assert self._last_action is not None
            self._policy.experience(
                self._last_observation, self._last_action, reward, observation)
        self._last_observation = observation
        action = self._policy.observe(reward, observation)
        if action is None:
            raise ValueError('must return an action or raise an input')
        self._last_action = action
        return action

    def perform(self, action):
        action = self._policy.perform(action)
        if action is None:
            raise ValueError('must return an action or raise an input')
        self._last_action = action
        return action

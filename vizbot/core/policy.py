class Policy:

    """
    A possibly partial behavior to interact with an environment. This is the
    base class for policies, sequential policies, and preprocesses.
    """

    def __init__(self, observations, actions):
        self.__observations = observations
        self.__actions = actions
        self.__training = None
        self.__timestep = 0
        self.__last_observation = None
        self.__last_action = None

    @property
    def observations(self):
        return self.__observations

    @property
    def actions(self):
        return self.__actions

    @property
    def timestep(self):
        return self.__timestep

    @property
    def training(self):
        if self.__training is None:
            raise RuntimeError('no current episide')
        return self.__training

    @property
    def testing(self):
        if self.__training is None:
            raise RuntimeError('no current episide')
        return not self.__training

    @property
    def above(self):
        """
        The spaces of forwarded observations and received actions.
        """
        self._above()

    def begin_episode(self, training):
        """
        Notify the begin of an episode.
        """
        self.__training = training
        self.__last_observation = None
        self.__last_action = None
        self._begin_episode(training)

    def end_episode(self):
        """
        Notify the end of an episode.
        """
        self.__training = None
        self._end_episode()

    def observe(self, reward, observation):
        """
        Forward an observation and previous reward to the policy.
        """
        if self.training:
            self.__timestep += 1
        if reward is not None:
            assert self.timestep == 0
            self.experience(
                self.__last_observation,
                self.__last_action,
                reward,
                observation)
        self.__last_observation = observation
        action = self._observe(reward, observation)
        self.__last_action = action
        return action

    def perform(self, action):
        """
        Return an action to the policy.
        """
        return self._perform(action)

    @property
    def _above(self):
        """
        Override if your class raises observations and thus receives actions.
        """
        return None

    def _begin_episode(self, training):
        """
        Optional hook called at the beginning of each episode.
        """
        pass

    def _end_episode(self):
        """
        Optional hook called at the end of each episode.
        """
        pass

    def _observe(self, reward, observation):
        """
        Receive a reward and an observation from the previous behavior. Return
        an action or raise a tuple of a reward and an observation to process by
        the next behavior. The first reward and last observation of an episode
        are None. Do not call this method directly.
        """
        raise (reward, observation)

    def _perform(self, action):
        """
        Receive an action from the next behavior. Return an action or raise a
        tuple of a reward and an observation to process by the next behavior.
        """
        return action

    def _experience(self, observation, action, reward, successor):
        """
        Optional hook to process the transitions. Successor is None when the
        episode ended after the reward. All other arguments are never None.
        """
        assert observation is not None
        assert action is not None
        assert reward is not None

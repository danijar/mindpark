from vizbot.core import Policy


class Experience(Policy):

    def __init__(self, interface):
        super().__init__(interface)
        self._last_observation = None
        self._last_action = None
        self._last_reward = None

    def begin_episode(self, training):
        super().begin_episode(training)
        self._last_observation = None
        self._last_action = None
        self._last_reward = None

    def observe(self, observation):
        self._apply_experience(observation)
        if self.training:
            self._last_observation = observation
        super().observe(observation)
        action = self.perform(observation)
        self._last_action = action
        assert self.actions.contains(action)
        return action

    def receive(self, reward, final):
        super().receive(reward, final)
        self._last_reward = reward
        if final:
            self._apply_experience(None)

    def perform(self, observation):
        """
        Choose an action based on the current observation.
        """
        raise NotImplementedError

    def experience(self, observation, action, reward, successor):
        """
        Optional hook to process the current transition. Successor is None when
        the episode ended after the reward. All other arguments are never None.
        When overriding, do not forget to forward the experience of terminal
        states, where `successor` is None.
        """
        raise NotImplementedError

    def _apply_experience(self, successor):
        if self._last_observation is None:
            return
        assert self._last_observation is not None
        assert self._last_action is not None
        assert self._last_reward is not None
        print(future)
        self.experience(
            self._last_observation, self._last_action,
            self._last_reward, successor)
        self._last_observation = None
        self._last_action = None
        self._last_reward = None

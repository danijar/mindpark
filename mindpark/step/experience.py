from mindpark.core import Partial


class Experience(Partial):

    def __init__(self, task):
        super().__init__(task)
        self._last_observ = None
        self._last_action = None
        self._last_reward = None

    @property
    def above_observs(self):
        return None

    @property
    def above_actions(self):
        return None

    def begin_episode(self, episode, training):
        super().begin_episode(episode, training)
        self._last_observ = None
        self._last_action = None
        self._last_reward = None

    def observe(self, observ):
        self._apply_experience(observ)
        if self.training:
            self._last_observ = observ
        super().observe(observ)
        action = self.perform(observ)
        self._last_action = action
        assert self.task.actions.contains(action)
        return action

    def receive(self, reward, final):
        super().receive(reward, final)
        self._last_reward = reward
        if final:
            self._apply_experience(None)

    def perform(self, observ):
        """
        Choose an action based on the current observ.
        """
        raise NotImplementedError

    def experience(self, observ, action, reward, successor):
        """
        Optional hook to process the current transition. Successor is None when
        the episode ended after the reward. All other arguments are never None.
        When overriding, do not forget to forward the experience of terminal
        states, where `successor` is None.
        """
        raise NotImplementedError

    def _apply_experience(self, successor):
        if self._last_observ is None:
            return
        assert self._last_observ is not None
        assert self._last_action is not None
        assert self._last_reward is not None
        self.experience(
            self._last_observ, self._last_action,
            self._last_reward, successor)
        self._last_observ = None
        self._last_action = None
        self._last_reward = None

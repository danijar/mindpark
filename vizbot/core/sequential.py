import numpy as np
from vizbot.core.policy import Policy


class Sequential(Policy):

    def __init__(self, observations, actions):
        super().__init__(observations, actions)
        self._policies = []

    @property
    def above_observations(self):
        if not self._policies:
            return self.below
        return self._policies[-1].above_observations

    @property
    def above_actions(self):
        if not self._policies:
            return self.below
        return self._policies[-1].above_actions

    def begin_episode(self, training):
        super().begin_episode(training)
        for policy in self._policies:
            policy.begin_episode(training)

    def end_episode(self):
        super().end_episode()
        for policy in self._policies:
            policy.end_episode()

    def observe(self, observation):
        super().observe(observation)
        return self._simulate(0, observation, None)

    def perform(self, action):
        super().perform(action)
        return self._simulate(len(self._policies) - 1, None, action)

    def add(self, policy, *args, **kwargs):
        if not isinstance(policy, Policy):
            policy = policy(self.above, *args, **kwargs)
        elif args or kwargs:
            raise ValueError('cannot specify args for instantiated policy')
        self._policies.append(policy)

    def _simulate(self, level, observation, action):
        while True:
            if level < 0:
                assert action is not None
                return action
            if level >= len(self._policies):
                assert observation is not None
                raise observation
            level, observation, action = self._simulate(
                level, observation, action)
        return action

    def _step(self, level, observation, action):
        assert observation is None or action is None
        policy = self._policies[level]
        try:
            if observation is not None:
                action = policy.observe(observation)
                observation = None
            elif action is not None:
                action = policy.perform(action)
            if action is None:
                raise ValueError('action cannot be None')
            level -= 1
        except np.ndarray as observation:
            action = None
            level += 1
        return level, observation, action

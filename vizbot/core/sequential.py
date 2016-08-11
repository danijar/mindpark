import numpy as np
from vizbot.core.policy import Policy


class Sequential(Policy):

    def __init__(self, observations, actions):
        super().__init__(observations, actions)
        self._steps = []

    @property
    def above_observations(self):
        if not self._steps:
            return self.below
        return self._steps[-1].policy.above_observations

    @property
    def above_actions(self):
        if not self._steps:
            return self.below
        return self._steps[-1].policy.above_actions

    def add(self, policy, *args, **kwargs):
        if not isinstance(policy, Policy):
            policy = policy(self.above, *args, **kwargs)
        elif args or kwargs:
            raise ValueError('cannot specify args for instantiated policy')
        self._steps.append(Step(policy))

    def begin_episode(self, training):
        super().begin_episode(training)
        for step in self._steps:
            step.policy.begin_episode(training)

    def end_episode(self):
        super().end_episode()
        for step in self._steps:
            step.policy.end_episode()

    def observe(self, reward, observation):
        super().observe(reward, observation)
        return self._simulate(0, (reward, observation), None)

    def perform(self, action):
        super().perform(action)
        return self._simulate(len(self._steps) - 1, None, action)

    def _simulate(self, level, input_, output):
        while True:
            if level < 0:
                assert output is not None
                return output
            if level >= len(self._steps):
                assert input_ is not None
                raise input_
            level, input_, output = self._step(level, input_, output)

    def _step(self, level, input_, output):
        assert (input_ is None) != (output is None)
        step = self._steps[level]
        try:
            if input_ is not None:
                output = step.observe(*input_)
            elif output is not None:
                output = step.perform(output)
            if output is None:
                raise ValueError('action cannot be None')
            input_ = None
            level -= 1
        except tuple as e:
            self._validate_raised_input(e)
            input_ = e
            output = None
            level += 1
        return level, input_, output

    @staticmethod
    def _validate_raised_input(e):
        if len(e) != 2:
            raise ValueError('should raise reward and observation')
        if not isinstance(e[0], (float, int)):
            raise ValueError('reward must be a number')
        if not isinstance(e[1], np.ndarray):
            raise ValueError('observation must be a Numpy array')


class Step:

    def __init__(self, policy):
        self.policy = policy
        self._last_observation = None
        self._last_action = None

    def observe(self, reward, observation):
        if not (self._last_observation is None or self._last_action is None):
            self.policy.experience(
                self._last_observation, self._last_action, reward, observation)
            self._last_observation = None
            self._last_action = None
        self._last_observation = observation
        action = self.policy.observe(reward, observation)
        self._last_action = action
        return action

    def perform(self, action):
        action = self.policy.perform(action)
        self._last_action = action
        return action

import inspect
from vizbot.core.policy import Policy, Input


class Sequential(Policy):

    def __init__(self, observations, actions):
        super().__init__(observations, actions)
        self.steps = []

    @property
    def interface(self):
        if not self.steps:
            return self.observations, self.actions
        return self.steps[-1].interface

    def add(self, policy, *args, **kwargs):
        if inspect.isclass(policy):
            policy = policy(*self.interface, *args, **kwargs)
        elif args or kwargs:
            raise ValueError('cannot specify args for instantiated policy')
        policy = ExperienceProxy(policy)
        self.steps.append(policy)

    def begin_episode(self, training):
        super().begin_episode(training)
        for policy in self.steps:
            policy.begin_episode(training)

    def end_episode(self):
        super().end_episode()
        for policy in self.steps:
            policy.end_episode()

    def observe(self, reward, observation):
        super().observe(reward, observation)
        return self._simulate(0, Input(reward, observation), None)

    def perform(self, action):
        super().perform(action)
        return self._simulate(len(self.steps) - 1, None, action)

    def _simulate(self, level, input_, action):
        while True:
            if level < 0:
                assert action is not None
                return action
            if level >= len(self.steps):
                assert input_ is not None
                raise input_
            level, input_, action = self._step(level, input_, action)

    def _step(self, level, input_, action):
        assert (input_ is None) != (action is None)
        policy = self.steps[level]
        try:
            if input_ is not None:
                assert isinstance(policy, ExperienceProxy)
                action = policy.observe(input_.reward, input_.observation)
            elif action is not None:
                action = policy.perform(action)
            input_ = None
            level -= 1
        except Input as input_:
            self._validate_input(policy.above_observations, input_)
            action = None
            level += 1
        return level, input_, action

    @staticmethod
    def _validate_input(space, input_):
        if not isinstance(input_.reward, (float, int)):
            raise ValueError('reward must be a number')
        if not space.contains(input_.observation):
            raise ValueError('invalid observation')


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
            self._policy.experience(
                self._last_observation, self._last_action, reward, observation)
        self._last_observation = observation
        action = self._policy.observe(reward, observation)
        self._last_action = action
        return action

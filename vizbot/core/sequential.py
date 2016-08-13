import inspect
from vizbot.core.policy import Policy, Input, ExperienceProxy


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
            else:
                assert False
            self._validate_action(level, action)
            input_ = None
            level -= 1
        except Input as raised:
            self._validate_input(level, raised)
            input_ = raised
            action = None
            level += 1
        return level, input_, action

    def _validate_input(self, level, input_):
        if input_.observation is None:
            return
        observations = self.steps[level].interface[0]
        if not observations.contains(input_.observation):
            message = 'invalid observation from level {}: {}'
            raise ValueError(message.format(level, input_.observation))

    def _validate_action(self, level, action):
        if level < 1:
            actions = self.actions
        else:
            actions = self.steps[level - 1].interface[1]
        if not actions.contains(action):
            message = 'invalid action from level {}: {}'
            raise ValueError(message.format(level, action))

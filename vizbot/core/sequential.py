from vizbot.core.policy import Policy


class Sequential(Policy):

    def __init__(self, observations, actions):
        super().__init__(observations, actions)
        self.__policies = []

    @property
    def above_observations(self):
        if not self.__policies:
            return self.below
        return self.__policies[-1].above_observations

    @property
    def above_actions(self):
        if not self.__policies:
            return self.below
        return self.__policies[-1].above_actions

    def add(self, policy, *args, **kwargs):
        if not isinstance(policy, Policy):
            policy = policy(self.above, *args, **kwargs)
        elif args or kwargs:
            raise ValueError('cannot specify args for instantiated policy')
        self.__policies.append(policy)

    def _begin_episode(self, training):
        for policy in self.__policies:
            policy.begin_episode(training)

    def _end_episode(self):
        for policy in self.__policies:
            policy.end_episode()

    def _observe(self, reward, observation):
        return self._simulate(0, (reward, observation), None)

    def _perform(self, action):
        return self._simulate(len(self.__policies) - 1, None, action)

    def _simulate(self, level, input_, action):
        while True:
            if level < 0:
                assert action is not None
                return action
            if level >= len(self.__policies):
                assert input_ is not None
                raise input_
            level, input_, action = self._step(level, input_, action)

    def _step(self, level, input_, action):
        assert (input_ is None) != (action is None)
        policy = self.__policies[level]
        try:
            if input_ is not None:
                action = policy.observe(*input_)
            elif action is not None:
                action = policy.perform(action)
            input_ = None
            level -= 1
        except tuple as input_:
            self._validate_raised_input(policy.above_observations, input_)
            action = None
            level += 1
        return level, input_, action

    @staticmethod
    def _validate_raised_input(space, input_):
        if len(input_) != 2:
            raise ValueError('should raise reward and observation')
        reward, observation = input_
        if not isinstance(reward, (float, int)):
            raise ValueError('reward must be a number')
        if not space.contains(observation):
            raise ValueError('invalid observation')

class Policy:

    def __init__(self, observations, actions):
        self._observations = observations
        self._actions = actions
        self._training = None
        self._timestep = 0

    @property
    def observations(self):
        return self._observations

    @property
    def actions(self):
        return self._actions

    @property
    def timestep(self):
        return self._timestep

    @property
    def training(self):
        if self._training is None:
            raise RuntimeError('currently no episode')
        return self._training

    @property
    def testing(self):
        if self._training is None:
            raise RuntimeError('currently no episode')
        return not self._training

    @property
    def above_observations(self):
        raise NotImplementedError

    @property
    def above_actions(self):
        raise NotImplementedError

    def begin_episode(self, training):
        self._training = training

    def end_episode(self):
        self._training = None

    def observe(self, reward, observation):
        self._timestep += 1

    def perform(self, action):
        return action

    def experience(self, observation, action, reward, successor):
        pass

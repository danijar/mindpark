from vizbot.core import Step


class Skip(Step):

    def __init__(self, interface, amount):
        super().__init__(interface)
        self._amount = amount
        self._action = None
        self._reward = None

    def begin_episode(self, training):
        super().begin_episode(training)
        self._reward = 0

    def step(self, observation):
        super().step(observation)
        if not self.timestep % self._amount or observation is None:
            self._action = self.above.step(observation)
        return self._action

    def experience(self, observation, action, reward, successor):
        super().experience(observation, action, reward, successor)
        self._reward += reward
        if not self.timestep % self._amount or successor is None:
            self.above.experience(observation, action, self._reward, successor)
            self._reward = 0

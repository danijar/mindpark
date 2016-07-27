import numpy as np


class Agent:

    @classmethod
    def defaults(cls):
        discount = 0.95
        return locals()

    def __init__(self, trainer, config):
        self._trainer = trainer
        self._config = config
        self._env = trainer.create_env()
        self._random = np.random.RandomState()
        self._training = None

    def start_epoch(self):
        pass

    def stop_epoch(self):
        pass

    def start_episode(self, training):
        self._training = training

    def stop_episode(self):
        pass

    def step(self, state):
        pass

    def experience(self, state, action, reward, successor):
        pass

    def close(self):
        for learner in self.learners:
            if learner is self:
                continue
            learner.close()
        self._env.close()

    @property
    def states(self):
        return self._env.states

    @property
    def actions(self):
        return self._env.actions

    @property
    def timestep(self):
        return self._trainer.timestep

    @property
    def config(self):
        return self._config

    @property
    def training(self):
        return self._training

    @property
    def learners(self):
        return (self,)

    @property
    def testee(self):
        return self

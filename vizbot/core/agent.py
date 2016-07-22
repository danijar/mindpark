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

    def __call__(self):
        while self._trainer.running:
            self._trainer.run_episode(self, self._env)

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

    def start(self):
        pass

    def step(self, state):
        pass

    def stop(self):
        pass

    def experience(self, state, action, reward, successor):
        pass

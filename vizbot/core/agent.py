import numpy as np


class Agent:

    def __init__(self, trainer):
        self._trainer = trainer
        self._random = np.random.RandomState(seed=0)
        self._env = trainer.create_env()

    def __call__(self):
        while self._trainer.running:
            self._trainer.run_episode(self, self._env)

    @property
    def states(self):
        return self._env.states

    @property
    def actions(self):
        return self._env.actions

    def start(self):
        pass

    def step(self, state):
        pass

    def stop(self):
        pass

    def experience(self, state, action, reward, successor):
        pass

    def _noop(self):
        return np.zeros(self.actions.shape)

    def _decay(self, start, end, over):
        progress = min(self._trainer.timestep, over) / over
        return (1 - progress) * start + progress * end

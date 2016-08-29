import numpy as np
from gym.spaces import Box
from mindpark.core import Partial


class ActionSample(Partial):

    def __init__(self, task, temperature=1, logprobs=False):
        super().__init__(task)
        self._temperature = temperature
        self._logprobs = logprobs

    @property
    def above_observs(self):
        return self.task.observs

    @property
    def above_actions(self):
        return Box(0, 1, self.task.actions.n)

    def observe(self, observ):
        super().observe(observ)
        action = self.above.observe(observ)
        if not self._logprobs:
            action = np.log(action / action.sum())
        action = np.exp(action / self._temperature)
        action /= action.sum()
        choice = self.random.choice(self.task.actions.n, p=action)
        return choice

    def receive(self, reward, final):
        super().receive(reward, final)
        self.above.receive(reward, final)

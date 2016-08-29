import numpy as np
from gym.spaces import Box
from mindpark.core import Partial, Metric


class ActionMax(Partial):

    def __init__(self, task):
        super().__init__(task)
        choices = self.task.actions.n
        self._values = Metric(self.task, 'action_max/values', choices)
        self._action = Metric(self.task, 'action_max/action', 1)

    @property
    def above_observs(self):
        return self.task.observs

    @property
    def above_actions(self):
        return Box(-np.inf, np.inf, self.task.actions.n)

    def observe(self, observ):
        super().observe(observ)
        values = self.above.observe(observ)
        action = values.argmax()
        self._values(*values)
        self._action(action)
        return action

    def receive(self, reward, final):
        super().receive(reward, final)
        self.above.receive(reward, final)

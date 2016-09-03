import numpy as np
from gym.spaces import Box
from mindpark.core import Partial, Metric
from mindpark.utility import Decay


class EpsilonGreedy(Partial):

    """
    Act randomly with a probability of epsilon, and use the action of the above
    policy otherwise. Note that the above policy is always evaluated, to
    collect statistics and allow to experience transitions.
    """

    def __init__(self, task,
                 from_=1, to=0.1, test=0.05, over=100, offset=0):
        super().__init__(task)
        self._offset = int(float(offset))
        self._test = test
        self._epsilon = Decay(from_, to, over)
        self._metric_epsilon = Metric(
            self.task, 'epsilon_greedy/epsilon', 1)
        self._metric_values = Metric(
            self.task, 'epsilon_greedy/values', self.task.actions.n)
        self._metric_action = Metric(
            self.task, 'epsilon_greedy/action', 1)
        self._metric_random = Metric(
            self.task, 'epsilon_greedy/random', 1)

    @property
    def above_observs(self):
        return self.task.observs

    @property
    def above_actions(self):
        return Box(-np.inf, np.inf, self.task.actions.n)

    def observe(self, observ):
        super().observe(observ)
        step = max(0, self.task.step - self._offset)
        epsilon = self._epsilon(step) if self.training else self._test
        values = self.above.observe(observ)
        if self.random.rand() <= epsilon:
            random = True
            action = self.task.actions.sample()
        else:
            random = False
            action = np.argmax(values)
        self._metric_epsilon(epsilon)
        self._metric_values(*values)
        self._metric_action(action)
        self._metric_random(random)
        return action

    def receive(self, reward, final):
        super().receive(reward, final)
        self.above.receive(reward, final)

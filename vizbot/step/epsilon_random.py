from vizbot.core import Metric
from vizbot.step.experience import Experience
from vizbot.utility import Decay


class EpsilonRandom(Experience):

    """
    Act randomly with a probability of epsilon, and use the action of the above
    policy otherwise. If the above policy inherits from `Experience`, it will
    receive transitions into `experience()` for randomly choosen actions, as
    well.
    """

    # TODO: Create a Schedule class to hold the parameters `from_`, `to`,
    # `over` and `offset`. There can also be a constant schedule.

    # TODO: Forwarding experiences will not work if the above policy is a
    # sequential policy wrapping an experience policy.

    def __init__(self, task,
                 from_=1, to=0.1, test=0.05, over=100, offset=0):
        super().__init__(task)
        self._offset = offset
        self._test = test
        self._epsilon = Decay(from_, to, over)
        self._epsilon_metric = Metric(
            self.task, 'epsilon_random/epsilon', 1)
        self._was_random = None
        self._was_random_metric = Metric(
            self.task, 'epsilon_random/was_random', 1)

    @property
    def above_observs(self):
        return self.task.observs

    @property
    def above_actions(self):
        return self.task.actions

    def perform(self, observ):
        if self.training:
            step = max(0, self.task.step - self._offset)
            epsilon = self._epsilon(step)
        else:
            epsilon = self._test
        if self.random.rand() <= epsilon:
            self._was_random = True
            action = self.task.actions.sample()
        else:
            self._was_random = False
            action = self.above.observe(observ)
        self._epsilon_metric(epsilon)
        self._was_random_metric(self._was_random)
        return action

    def receive(self, reward, final):
        super().receive(reward, final)
        if not self._was_random:
            self.above.receive(reward, final)

    def experience(self, *transition):
        # The above policy already receives transitions for which it decided
        # the action, but we want it to experience transitions with random
        # actions as well.
        if isinstance(self.above, Experience) and self._was_random:
            self.above.experience(*transition)

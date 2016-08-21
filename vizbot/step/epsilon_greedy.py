from vizbot.step.experience import Experience
from vizbot.utility import Decay


class EpsilonGreedy(Experience):

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
        self._probability = Decay(from_, to, over)
        self._was_greedy = None

    @property
    def above_observs(self):
        return self.task.observs

    @property
    def above_actions(self):
        return self.task.actions

    def perform(self, observ):
        if self.training:
            step = max(0, self.step - self._offset)
            epsilon = self._probability(step)
        else:
            epsilon = self._test
        if self.random.rand() <= epsilon:
            self._was_greedy = False
            return self.task.actions.sample()
        self._was_greedy = True
        return self.above.observe(observ)

    def receive(self, reward, final):
        super().receive(reward, final)
        if self._was_greedy:
            self.above.receive(reward, final)

    def experience(self, *transition):
        # The above policy already receives transitions for which it decided
        # the action, but we want it to experience transitions with random
        # actions as well.
        if isinstance(self.above, Experience) and not self._was_greedy:
            self.above.experience(*transition)

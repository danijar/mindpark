from mindpark.core import Partial


class Identity(Partial):

    """
    Do not inherit from this class. When modifying observations, you would
    either end up calling the above policy twice per time step, or not call the
    base policy class.
    """

    @property
    def above_observs(self):
        return self.task.observs

    @property
    def above_actions(self):
        return self.task.actions

    def observe(self, observ):
        super().observe(observ)
        action = self.above.observe(observ)
        assert self.task.actions.contains(action)
        return action

    def receive(self, reward, final):
        super().receive(reward, final)
        self.above.receive(reward, final)

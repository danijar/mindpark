from mindpark.core import Partial


class ClampReward(Partial):

    @property
    def above_observs(self):
        return self.task.observs

    @property
    def above_actions(self):
        return self.task.actions

    def observe(self, observ):
        super().observe(observ)
        return self.above.observe(observ)

    def receive(self, reward, final):
        super().receive(reward, final)
        reward = max(0, min(reward, 1))
        self.above.receive(reward, final)

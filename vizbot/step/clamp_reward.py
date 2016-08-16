from vizbot.step.identity import Identity


class ClampReward(Identity):

    def receive(self, reward, final):
        reward = max(0, min(reward, 1))
        self.above.receive(reward, final)

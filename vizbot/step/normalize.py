from vizbot.step.filter import Filter


class Normalize(Filter):

    def filter(self, observ):
        low, high = self.task.observs.low, self.task.observs.high
        observ = (observ.astype(float) - low) / (high - low)
        return observ

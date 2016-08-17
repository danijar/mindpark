from vizbot.step.filter import Filter


class Normalize(Filter):

    def filter(self, observation):
        low, high = self.observations.low, self.observations.high
        observation = (observation.astype(float) - low) / (high - low)
        return observation

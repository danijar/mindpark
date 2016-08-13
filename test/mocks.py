from gym.spaces import Box, Discrete
from vizbot import core, step


class DurationEnv(core.Env):

    def __init__(self, duration):
        self.duration = duration
        self.timestep = None

    @property
    def interface(self):
        return Box(0, 9, (8, 6, 3)), Discrete(3)

    def reset(self):
        self.timestep = 0
        return self.interface[0].sample()

    def step(self, action):
        assert action is not None
        assert self.interface[1].contains(action)
        self.timestep += 1
        if self.timestep >= self.duration:
            return 0, None
        return 0, self.interface[0].sample()


class Monitored(core.Policy):

    def step(self, observation):
        assert self.observations.contains(observation)
        self.observation = observation
        return super().step(observation)

    def experience(self, *transition):
        self.transition = transition
        super().experience(*transition)


class Sequential(Monitored, core.Sequential): pass
class Identity(Monitored, step.Identity): pass
class Skip(Monitored, step.Skip): pass
class Random(Monitored, step.Random): pass

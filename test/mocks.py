from gym.spaces import Box, Discrete
from vizbot import core, step


class DurationEnv(core.Env):

    def __init__(self, duration):
        self.duration = duration
        self.timestep = None

    @property
    def observs(self):
        # return Box(0, 4.2, (8, 6, 3))
        return Box(0, 4.2, (80, 60, 3))

    @property
    def actions(self):
        return Discrete(3)

    def reset(self):
        self.timestep = 0
        return self.observs.sample()

    def step(self, action):
        assert action is not None
        assert self.actions.contains(action)
        self.timestep += 1
        if self.timestep >= self.duration:
            return 0, None
        return 0, self.observs.sample()


class Monitored(core.Policy):

    def observe(self, observ):
        self.observ = observ
        return super().observe(observ)

    def receive(self, reward, final):
        self.reward = reward
        super().receive(reward, final)


class Sequential(Monitored, core.Sequential): pass
class Identity(Monitored, step.Identity): pass
class Skip(Monitored, step.Skip): pass
class Random(Monitored, step.Random): pass


class MockViewer:

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        pass

    def close(self):
        pass

    def pressed_keys(self):
        return []

    def delta(self):
        return (0, 0)

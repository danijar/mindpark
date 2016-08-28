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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step = None

    def begin_episode(self, episode, training):
        super().begin_episode(episode, training)
        # Store primitive types so that we capture the actual value, not a
        # reference to the possibly wrapping objects.
        self.episode = int(self.task.episode)
        self.training = bool(self.task.training)

    def observe(self, observ):
        # Task show episode of this policies, even during parallel training.
        assert self.task.episode == self.episode
        assert self.task.training == self.training
        self.observ = observ
        self.step = 0 if self.step is None else self.step + 1
        return super().observe(observ)

    def receive(self, reward, final):
        self.reward = reward
        super().receive(reward, final)


class Sequential(Monitored, core.Sequential):
    pass


class Identity(Monitored, step.Identity):
    pass


class Skip(Monitored, step.Skip):
    pass


class Random(Monitored, step.Random):
    pass


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

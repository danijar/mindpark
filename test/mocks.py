from gym.spaces import Box, Discrete
from vizbot.core import Env, Policy, Input


class DurationEnv(Env):

    def __init__(self, duration):
        self.duration = duration
        self.timestep = None

    @property
    def observations(self):
        return Box(0, 1, (80, 60))

    @property
    def actions(self):
        return Discrete(3)

    def reset(self):
        self.timestep = 0
        return self.observations.sample()

    def step(self, action):
        assert self.actions.contains(action)
        self.timestep += 1
        if self.timestep >= self.duration:
            return 0, None
        return 0, self.observations.sample()


class Monitored(Policy):

    def __init__(self, observations, actions):
        super().__init__(observations, actions)
        self.reward = None
        self.observation = None
        self.transition = None

    @property
    def interface(self):
        return self.observations, self.actions

    def observe(self, reward, observation):
        super().observe(reward, observation)
        self.reward = reward
        self.observation = observation

    def experience(self, observation, action, reward, successor):
        self.transition = observation, action, reward, successor


class Identity(Monitored):

    def observe(self, reward, observation):
        super().observe(reward, observation)
        raise Input(reward, observation)


class Skip(Monitored):

    def __init__(self, observations, actions, amount):
        super().__init__(observations, actions)
        self.amount = amount
        self.action = None

    def observe(self, reward, observation):
        super().observe(reward, observation)
        if observation is None:
            raise Input(reward, observation)
        if self.timestep % self.amount == 0:
            raise Input(reward, observation)
        return self.action

    def perform(self, action):
        super().perform(action)
        self.action = action
        return action


class Random(Monitored):

    def observe(self, reward, observation):
        super().observe(reward, observation)
        return self.actions.sample()

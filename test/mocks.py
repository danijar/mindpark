from gym.spaces import Box, Discrete
from vizbot.core import Env, Policy


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
            return None
        return self.observations.sample()


class Monitored(Policy):

    def __init__(self, observations, actions):
        super().__init__(observations, actions)
        self.reward = None
        self.observation = None

    def observe(self, reward, observation):
        self.reward = reward
        self.observation = observation

    def experience(self, observation, action, reward, successor):
        self.transition = observation, action, reward, successor


class Identity(Monitored):

    def observe(self, reward, observation):
        super().observe(reward, observation)
        raise (reward, observation)


class Skip(Monitored):

    def __init__(self, observations, actions, amount):
        super().__init__(observations, actions)
        self.amount = amount
        self.action = None

    def observe(self, reward, observation):
        super().observe(reward, observation)
        if self.timestep % self.amount > 0:
            return self.action
        raise (reward, observation)

    def perform(self, action):
        self.action = action
        return action


class Random(Monitored):
    def observe(self, reward, observation):
        super().observe(reward, observation)
        return self.observations.sample()

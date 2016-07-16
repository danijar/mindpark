import gym
from vizbot.core.env import Env


class GymEnv(Env):

    def __init__(self, name):
        super().__init__()
        self._env = gym.make(name)
        self._env.seed(0)
        self._state = None

    @property
    def states(self):
        return self._env.observation_space

    @property
    def actions(self):
        return self._env.action_space

    def begin(self):
        super().begin()
        self._state = self._env.reset()

    def step(self):
        super().step()
        action = self._agent.perform(self._state)
        self._state, reward, done, info = self._env.step(action)
        self._agent.feedback(action, reward)
        return self._state, reward, done

    def end(self):
        super().end()

    @property
    def monitor(self):
        return self._env.monitor

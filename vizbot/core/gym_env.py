import gym
from vizbot.core.env import Env, StopEpisode


class GymEnv(Env):

    def __init__(self, name, directory=None, videos=False):
        super().__init__()
        self._env = gym.make(name)
        self._env.seed(0)
        self._directory = directory
        if self._directory:
            videos = None if videos else False
            self._env.monitor.start(self._directory, videos)
        self._state = None

    @property
    def states(self):
        return self._env.observation_space

    @property
    def actions(self):
        return self._env.action_space

    def reset(self):
        return self._env.reset()

    def step(self, action):
        state, reward, done, info = self._env.step(action)
        if done:
            raise StopEpisode(self)
        return state, reward

    def close(self):
        if self._directory:
            self._env.monitor.stop()

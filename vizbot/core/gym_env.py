import gym
from threading import Lock
from vizbot.core.env import Env, StopEpisode


class GymEnv(Env):

    def __init__(self, name, directory=None, videos=False):
        super().__init__()
        self._env = gym.make(name)
        self._directory = directory
        if self._directory:
            self._env.monitor.start(self._directory, videos, resume=True)
        self._done = None
        self._lock = Lock()

    @property
    def states(self):
        return self._env.observation_space

    @property
    def actions(self):
        return self._env.action_space

    def reset(self):
        self._done = False
        return self._env.reset()

    def step(self, action):
        if self._done:
            raise StopEpisode(self)
        state, reward, done, _ = self._env.step(action)
        self._done = done
        return state, reward

    def close(self):
        if self._directory:
            self._env.monitor.stop()
        self._env.close()

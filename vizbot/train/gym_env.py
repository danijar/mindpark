import gym
from vizbot.core import Env


class GymEnv(Env):

    def __init__(self, env_name, directory=None, videos=False):
        super().__init__()
        self._env = gym.make(env_name)
        self._directory = directory
        if self._directory:
            # TODO: Try without resume.
            self._env.monitor.start(self._directory, videos, resume=True)

    @property
    def interface(self):
        return self._env.observation_space, self._env.action_space

    def reset(self):
        return self._env.reset()

    def step(self, action):
        observation, reward, done, _ = self._env.step(action)
        if done:
            assert observation is None
        return reward, observation

    def close(self):
        if self._directory:
            self._env.monitor.close()
        self._env.close()

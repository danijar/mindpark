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
        observation = self._env.reset()
        return observation

    def step(self, action):
        assert self.interface[1].contains(action)
        observation, reward, done, _ = self._env.step(action)
        assert self.interface[0].contains(observation)
        assert isinstance(reward, (int, float))
        if done:
            # May not be None if Gym aborted after too many time steps.
            observation = None
        return reward, observation

    def close(self):
        if self._directory:
            self._env.monitor.close()
        self._env.close()

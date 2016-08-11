import gym
from vizbot.core.env import Env


class GymEnv(Env):

    def __init__(self, env_name, directory=None, videos=False):
        super().__init__()
        self._env = gym.make(env_name)
        self._directory = directory
        if self._directory:
            # TODO: Try without resume.
            self._env.monitor.start(self._directory, videos, resume=True)

    @property
    def observations(self):
        return self._env.observation_space

    @property
    def actions(self):
        return self._env.action_space

    def reset(self):
        observation = self._env.reset()
        observation = self._process_observation(observation)
        raise (0, observation)

    def step(self, action):
        observation, reward, done, _ = self._env.step(action)
        observation = self._process_observation(observation)
        if done:
            return None
        raise (reward, observation)

    def close(self):
        if self._directory:
            self._env.monitor.close()
        self._env.close()

    @staticmethod
    def _process_observation(observation):
        return observation.astype(float)

from vizbot.train.gym_env import GymEnv
from vizbot.train.simulator import Simulator
from vizbot.utility import ensure_directory


class Trainer:

    def __init__(self, directory, env_name, algorithm,
                 epochs, train_steps, test_steps, videos=10):
        if directory:
            ensure_directory(directory)
        self._directory = directory
        self._algorithm = algorithm
        self._epochs = epochs
        self._videos = directory and videos
        self._video = None
        self._envs = [self._create_env(env_name, True)]
        for _ in algorithm.train_policies:
            self._envs.append(self._create_env(env_name, False))
        self._test = Simulator(
            self._envs[:1], [algorithm.test_policy], test_steps, False)
        self._train = Simulator(
            self._envs[1:], algorithm.train_policies, train_steps, True)

    def __iter__(self):
        for epoch in range(self._epochs + 1):
            self._algorithm.begin_epoch(epoch)
            if epoch:
                self._train()
            self._video = 0
            score = self._test()
            self._algorithm.end_epoch()
            yield score

    @property
    def timestep(self):
        return self._train.timestep

    def close(self):
        for policy in self._algorithm.train_policies:
            policy.close()
        self._algorithm.test_policy.close()
        self._algorithm.close()
        for env in self._envs:
            env.close()

    def _create_env(self, env_name, monitoring):
        directory = monitoring and self._directory
        env = GymEnv(env_name, directory, self._video_callback)
        return env

    def _video_callback(self, ignore):
        if not self._videos:
            return False
        every = self._test.timesteps / self._videos
        if self._test.timestep < self._video * every:
            return False
        self._video += 1
        return True

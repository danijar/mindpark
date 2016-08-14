from vizbot.core import Simulator
from vizbot.train.gym_env import GymEnv
from vizbot.utility import ensure_directory


class Trainer:

    # TODO: Move remaining logic into Benchmark class.

    def __init__(self, task, algo_cls, algo_config, videos=10):
        self._task = task
        if self._task.directory:
            ensure_directory(self._task.directory)
        self._algorithm = algo_cls(self._task, algo_config)
        self._videos = self._task.directory and videos
        self._video = None
        self._envs = [self._create_env(self._task.env_name, True)]
        for _ in self._algorithm.train_policies:
            self._envs.append(self._create_env(self._task.env_name, False))
        self._test = Simulator(
            self._envs[:1], [self._algorithm.test_policy],
            self._task.test_steps, False)
        self._train = Simulator(
            self._envs[1:], self._algorithm.train_policies,
            task.timesteps // self._epochs, True,
            lambda x: self._task.set_timestep(x))

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
        return self._task.timestep

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

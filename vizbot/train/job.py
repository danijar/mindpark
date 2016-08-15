import traceback
from vizbot.core import Simulator
from vizbot.utility import dump_yaml, print_headline
from vizbot.train.gym_env import GymEnv


class Job:

    MESSAGE_BEFORE = 'Before training average score {:.2f}'
    MESSAGE_EPOCH = 'Epoch {} timestep {} average score {:.2f}'

    def __init__(self, train_task, test_task, env_name, algo_conf, definition,
                 prefix, videos):
        self._train = train_task
        self._test = test_task
        self._env_name = env_name
        self._algo_conf = algo_conf
        self._definition = definition
        self._videos = videos
        self._video = 0
        self._envs = []

    def __call__(self, lock):
        with lock:
            print_headline(self._prefix, 'Start job')
        self._train.directory and dump_yaml(
            self._algo_conf, self._train.directory, 'algorithm.yaml')
        try:
            algorithm, test, train = self._start()
            self._execute(algorithm, test, train)
        except Exception as e:
            with lock:
                print(self._prefix, 'Failed due to exception:')
                print(e)
            if self._stacktraces:
                traceback.print_exc()
        for env in self._envs:
            env.close()

    def _start(self):
        algorithm = self._algo_conf.type(self._train, self._algo_conf)
        test = Simulator(
            self._test,
            self._create_env(self._test.directory),
            algorithm.test_policy,
            False)
        train_policies = algorithm.train_policies
        train = Simulator(
            self._train,
            [self._create_env() for _ in train_policies],
            train_policies,
            True)
        return algorithm, test, train

    def _execute(self, algorithm, test, train):
        algorithm.begin_epoch(0)
        score = test(1 / self._definition.epochs)
        algorithm.end_epoch()
        print(self._prefix, self.MESSAGE_BEFORE.format(score))
        for epoch in range(1, self.task.epochs + 1):
            algorithm.begin_epoch(epoch)
            train(1 / self._definition.epochs)
            score = test(1 / self._definition.epochs)
            algorithm.end_epoch()
            args = epoch, self._train.step, score
            print(self._prefix, self.MESSAGE_EPOCH.format(*args))

    def _create_env(self, directory=None):
        callback = directory and self._video_callback
        env = GymEnv(self._env_name, directory, callback)
        self._envs.append(env)
        return env

    def _video_callback(self, ignore):
        if not self._videos:
            return False
        every = self._test.timesteps / self._videos
        if self._test.timestep < self._video * every:
            return False
        self._video += 1
        return True

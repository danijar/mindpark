import traceback
from vizbot.core import Simulator
from vizbot.utility import dump_yaml, print_headline
from vizbot.train.gym_env import GymEnv


class Job:

    MESSAGE_BEFORE = 'Before training average score {:.2f}'
    MESSAGE_EPOCH = 'Epoch {} step {} average score {}'

    def __init__(self, train_task, test_task, env_name, algo_conf, definition,
                 prefix, videos):
        self._train = train_task
        self._test = test_task
        self._env_name = env_name
        self._algo_conf = algo_conf
        self._definition = definition
        self._prefix = prefix
        self._videos = videos
        self._video = 0
        self._envs = []

    def __call__(self, lock):
        with lock:
            print_headline(self._prefix, 'Start job')
        self._train.directory and dump_yaml(
            self._algo_conf, self._train.directory, 'algorithm.yaml')
        try:
            algorithm, train, test = self._start()
            self._execute(algorithm, train, test)
        except Exception as e:
            with lock:
                print(self._prefix, 'Failed due to exception:')
                print(e)
                traceback.print_exc()
        for env in self._envs:
            env.close()

    def _start(self):
        algorithm = self._algo_conf.type(self._train, self._algo_conf)
        train_policies = algorithm.train_policies
        train = Simulator(
            self._train,
            [self._create_env() for _ in train_policies],
            train_policies,
            True)
        test = Simulator(
            self._test,
            [self._create_env(self._test.directory)],
            [algorithm.test_policy],
            False)
        return algorithm, train, test

    def _execute(self, algorithm, train, test):
        epochs = self._definition.epochs
        test_steps = self._test.steps / (epochs + 1)
        train_steps = self._train.steps / epochs
        algorithm.begin_epoch(0)
        score = test(test_steps)
        algorithm.end_epoch()
        print(self._prefix, self.MESSAGE_BEFORE.format(score))
        for epoch in range(1, epochs + 1):
            algorithm.begin_epoch(epoch)
            train(epoch * train_steps - self._train.step)
            score = test((epoch + 1) * test_steps - self._test.step)
            algorithm.end_epoch()
            args = epoch, self._train.step, score and round(score, 2)
            print(self._prefix, self.MESSAGE_EPOCH.format(*args))

    def _create_env(self, directory=None):
        callback = directory and self._video_callback
        env = GymEnv(self._env_name, directory, callback)
        self._envs.append(env)
        return env

    def _video_callback(self, ignore):
        if not self._videos:
            return False
        every = self._test.steps / (self._definition.epochs + 1) / self._videos
        if self._test.step < self._video * every:
            return False
        self._video += 1
        return True

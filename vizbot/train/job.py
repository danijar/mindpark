import traceback
from vizbot.core import Simulator
from vizbot.utility import dump_yaml, print_headline
from vizbot.train.gym_env import GymEnv


class Job:

    # TODO: Simplify this class.

    MESSAGE_BEFORE = 'Before training average score {}'
    MESSAGE_EPOCH = 'Epoch {} step {} average score {}'

    def __init__(self, task, env_name, algo_conf, prefix, videos):
        self._task = task
        self._env_name = env_name
        self._algo_conf = algo_conf
        self._prefix = prefix
        self._videos = videos
        self._video = 0
        self._envs = []
        self._algorithm = None
        self._train = None
        self._test = None

    def __call__(self, lock):
        with lock:
            print_headline(self._prefix, 'Start job')
        self._task.directory and dump_yaml(
            self._algo_conf, self._task.directory, 'algorithm.yaml')
        try:
            self._start()
            self._execute()
        except Exception as e:
            with lock:
                print(self._prefix, 'Failed due to exception:')
                print(e)
                traceback.print_exc()
        for env in self._envs:
            env.close()

    def _start(self):
        self._algorithm = self._algo_conf.type(self._task, self._algo_conf)
        train_policies = self._algorithm.train_policies
        self._train = Simulator(
            self._task.train, train_policies,
            [self._create_env() for _ in train_policies])
        self._test = Simulator(
            self._task.test, [self._algorithm.test_policy],
            [self._create_env(self._task.directory)])

    def _execute(self):
        epochs = max(self._task.test.epochs, self._task.train.epochs)
        for epoch in range(epochs):
            self._algorithm.begin_epoch()
            self._task.set_training(False)
            if epoch < self._task.epochs:
                self._run_test()
            self._task.set_training(True)
            if epoch < self._task.epochs:
                self._train()
            self._algorithm.end_epoch()

    def _run_test(self):
        score = self._test()
        score = score and round(score, 2)
        if self._task.test.epoch:
            message = self.MESSAGE_BEFORE.format(score)
        else:
            args = self._task.epoch, self._task.train.step, score
            message = self.MESSAGE_EPOCH.format(*args)
        print(self._prefix, message)

    def _create_env(self, directory=None):
        callback = directory and self._video_callback
        env = GymEnv(self._env_name, directory, callback)
        self._envs.append(env)
        return env

    def _video_callback(self, ignore):
        # TODO: Move into GymEnv?
        if not self._videos:
            return False
        task = self._task.test
        if task.step < self._video * task.steps / task.epochs / self._videos:
            return False
        self._video += 1
        return True

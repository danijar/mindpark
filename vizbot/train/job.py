import traceback
from vizbot.core import Simulator
from vizbot.utility import Proxy, dump_yaml, print_headline
from vizbot.train.gym_env import GymEnv


class Job:

    def __init__(
            self, train_task, test_task, env_name, algo_conf, prefix, videos):
        self._train_task = train_task
        self._test_task = test_task
        self._task = Proxy(train_task)
        self._epochs = max(train_task.epochs, test_task.epochs)
        self._env_name = env_name
        self._algo_conf = algo_conf
        self._prefix = prefix
        self._videos = videos
        self._video = 0
        self._envs = []

    def __call__(self, lock):
        with lock:
            print_headline(self._prefix, 'Start job')
        self._task.directory and dump_yaml(
            self._algo_conf, self._task.directory, 'algorithm.yaml')
        try:
            algorithm = self._create_algorithm()
            training = self._create_training(algorithm)
            testing = self._create_testing(algorithm)
            for _ in range(self._epochs):
                self._epoch(algorithm, training, testing)
        except Exception as e:
            with lock:
                print(self._prefix, 'Failed due to exception:', e)
                traceback.print_exc()
        finally:
            for env in self._envs:
                env.close()

    def _epoch(self, algorithm, training, testing):
        algorithm.begin_epoch()
        self._task.change(self._test_task)
        score = testing()
        self._print_score(score)
        self._task.change(self._train_task)
        training()
        algorithm.end_epoch()

    def _create_algorithm(self):
        return self._algo_conf.type(self._task, self._algo_conf)

    def _create_training(self, algorithm):
        policies = algorithm.train_policies
        envs = [self._create_env() for _ in policies]
        return Simulator(self._train_task, policies, envs)

    def _create_testing(self, algorithm):
        policies = [algorithm.test_policy]
        envs = [self._create_env(self._task.directory)]
        return Simulator(self._test_task, policies, envs)

    def _create_env(self, directory=None):
        env = GymEnv(self._env_name, directory, self._video_callback)
        self._envs.append(env)
        return env

    def _print_score(self, score):
        score = score and round(score, 2)
        if not self._task.epoch:
            message = 'Before training average score {}'
            print(self._prefix, message.format(score))
        else:
            message = 'Epoch {} train step {} average score {}'
            args = self._task.epoch, self._train_task.step, score
            print(self._prefix, message.format(*args))

    def _video_callback(self, ignore):
        if self._task.training:
            self._video = None
            return False
        if self._video is None:
            self._video = self._videos
        if self._video:
            self._video -= 1
            return True
        return False

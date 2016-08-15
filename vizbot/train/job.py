import os
import re
import traceback
import gym
from vizbot.core import Task, Simulator
from vizbot.utility import dump_yaml, print_headline
from vizbot.train.gym_env import GymEnv


class Job:

    MESSAGE_BEFORE = 'Before training average score {:.2f}'
    MESSAGE_EPOCH = 'Epoch {} timestep {} average score {:.2f}'

    def __init__(self, experiment, env_name, algo_conf, repeat, definition,
                 videos):
        directory = self._task_directory(
            experiment, env_name, algo_conf.name, repeat, definition.repeats)
        interface = self._determine_interface(env_name)
        train_steps = definition.epochs * algo_conf.train_steps
        test_steps = definition.epochs * definition.test_steps
        self._definition = definition
        self._test = Task(interface, test_steps, directory)
        self._train = Task(interface, train_steps, directory)
        self._env_name = env_name
        self.algo_conf = algo_conf
        self._videos = videos
        self._prefix = '{} on {} ({}):'.format(
            self.algo_conf.name, self._env_name, repeat)
        self._envs = []

    def __call__(self, lock):
        with lock:
            print_headline(self._prefix, 'Start job')
        self._train.directory and dump_yaml(
            self.algo_conf, self._train.directory, 'algorithm.yaml')
        try:
            algorithm, test, train = self._start_job()
            self._run_job(algorithm, test, train)
        except Exception as e:
            with lock:
                print(self._prefix, 'Failed due to exception:')
                print(e)
            if self._stacktraces:
                traceback.print_exc()
        for env in self._envs:
            env.close()

    def _start_job(self):
        algorithm = self.algo_conf.type(self._train, self.algo_conf)
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

    def _run_job(self, algorithm, test, train):
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

    @staticmethod
    def _determine_interface(env_name):
        env = gym.make(env_name)
        interface = env.observation_space, env.action_space
        env.close()
        return interface

    @staticmethod
    def _task_directory(experiment, env_name, algo_name, repeat, repeats):
        if not experiment:
            return
        template = '{{}}-{{:0>{}}}'.format(len(str(repeats - 1)))
        name = '-'.join(re.findall(r'[a-z0-9]+', algo_name.lower()))
        directory = experiment, env_name, template.format(name, repeat)
        return os.path.join(*directory)

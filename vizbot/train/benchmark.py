import re
import itertools
import os
import time
import traceback
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
import yaml
import gym
import vizbot.env
import vizbot.algorithm
from vizbot.core import Task, Simulator
from vizbot.train import GymEnv
from vizbot.utility import use_attrdicts, print_headline, dump_yaml


class Benchmark:

    """
    Train each algorithm on each environment for multiple repeats and store
    statistics and recordings in the experiment directory.
    """

    def __init__(self, directory=None, parallel=1, videos=False,
                 stacktraces=True):
        if directory:
            directory = os.path.abspath(os.path.expanduser(directory))
        self._directory = directory
        self._parallel = parallel
        self._videos = videos
        self._stacktraces = stacktraces
        self._lock = Lock()

    def __call__(self, definition):
        start = time.time()
        definition = Definition(definition)
        experiment = self._start_experiment(definition)
        if experiment:
            name = os.path.basename(experiment)
            self._dump_yaml(self._definition, experiment, name + '.yaml')
        tasks = itertools.product(
            range(self._definition.repeats),
            self._definition.envs,
            self._definition.algorithms)
        with ThreadPoolExecutor(max_workers=self._parallel) as executor:
            for repeat, env_name, algo_conf in tasks:
                job = Job(experiment, env_name, algo_conf, repeat, definition)
                executor.submit(job.__call__)
        message = 'Congratulations, benchmark finished after {} hours'
        duration = round((time.time() - start) / 3600, 1)
        self._print_headline(message.format(duration), style='=')
        if experiment:
            print('Find results in', experiment)

    def _start_experiment(self):
        self._print_headline('Start experiment', style='=')
        if not self._directory:
            print('Dry run; no results will be stored!')
            return None
        timestamp = time.strftime('%Y-%m-%dT%H-%M-%S', time.gmtime())
        name = '{}-{}'.format(timestamp, self._definition.experiment)
        experiment = os.path.join(self._directory, name)
        print('Result will be stored in', experiment)
        return experiment


class Job:

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
        self._envs = None

    def __call__(self, lock):
        with lock:
            print_headline(self._prefix, 'Start job')
        self._train.directory and dump_yaml(
            self.algo_conf, self._train.directory, 'algorithm.yaml')
        try:
            self._perform()
        except Exception as e:
            with lock:
                print(self._prefix, 'Failed due to exception:')
                print(e)
            if self._stacktraces:
                traceback.print_exc()
        for env in self._envs:
            env.close()

    def _perform(self):
        algorithm = self.algo_conf.type(self._train, self.algo_conf)
        testee, learners = algorithm.test_policy, algorithm.train_policies
        # TODO: Provide video callback here.
        self._envs = [GymEnv(self._env_name) for _ in range(len(learners) + 1)]
        self._test = Simulator(self._test, self._envs[:1], testee, False)
        self._train = Simulator(self._train, self._envs[1:], learners, True)
        algorithm.begin_epoch(0)
        score = self._test(1 / self._definition.epochs)
        algorithm.end_epoch()
        message = 'Before training average score {:.2f}'
        print(self._prefix, message.format(score))
        for epoch in range(1, self.task.epochs + 1):
            algorithm.begin_epoch(epoch)
            self._train(1 / self._definition.epochs)
            score = self._test(1 / self._definition.epochs)
            algorithm.end_epoch()
            message = 'Epoch {} timestep {} average score {:.2f}'
            print(self._prefix, message.format(epoch, self._train.step, score))

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


class Definition:

    def __new__(cls, filepath):
        with open(os.path.expanduser(filepath)) as file_:
            definition = yaml.load(file_)
        definition = use_attrdicts(definition)
        definition.experiment = str(definition.experiment)
        definition.epochs = int(float(definition.epochs))
        definition.test_steps = int(float(definition.test_steps))
        definition.repeats = int(float(definition.repeats))
        definition.envs = list(cls._load_envs(definition.envs))
        definition.algorithms = [
            cls._load_algorithm(x) for x in definition.algorithms]
        cls._validate_definition(definition)
        return definition

    def _load_envs(cls, envs):
        available_envs = [x.id for x in gym.envs.registry.all()]
        for env in envs:
            if env not in available_envs:
                raise KeyError('unknown env name {}'.format(env))
            yield env

    def _load_algorithm(cls, config):
        if not hasattr(vizbot.algorithm, config.type):
            message = 'unknown algorithm type {}'
            raise KeyError(message.format(config.type))
        config.type = getattr(vizbot.algorithm, config.type)
        if not issubclass(config.type, vizbot.core.Algorithm):
            raise KeyError('{} is not an algorithm'.format(config.type))
        config.name = str(config.name)
        config.train_steps = int(float(config.train_steps))
        defaults = config.type.defaults()
        reserved = ('type', 'name', 'train_steps')
        for key in defaults:
            if key in reserved:
                raise KeyError("reserved key '{}' in defaults".format(key))
        for key in config:
            if key not in defaults and key not in reserved:
                raise KeyError("unknown key '{}' in config".format(key))
        config.update(defaults)
        return use_attrdicts(config)

    def _validate_definition(cls, definition):
        names = [x.name for x in definition.algorithms]
        if len(set(names)) < len(names):
            raise KeyError('each algorithm must have an unique name')
        if not all(hasattr(x, 'train_steps') for x in definition.algorithms):
            raise KeyError('each algorithm must have a training duration')

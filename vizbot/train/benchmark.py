import itertools
import os
import re
import time
from threading import Lock
import gym
from concurrent.futures import ThreadPoolExecutor
from vizbot.core import Task
from vizbot.utility import print_headline, dump_yaml
from vizbot.train.definition import Definition
from vizbot.train.job import Job


class Benchmark:

    """
    Train each algorithm on each environment for multiple repeats and store
    statistics and recordings in the experiment directory.
    """

    def __init__(self, directory=None, parallel=1, videos=0):
        if directory:
            directory = os.path.abspath(os.path.expanduser(directory))
        self._directory = directory
        self._parallel = parallel
        self._videos = videos
        self._lock = Lock()

    def __call__(self, definition):
        start = time.time()
        definition = Definition(definition)
        experiment = self._start_experiment(definition)
        self._dump_definition(experiment, definition)
        jobs = self._create_jobs(experiment, definition)
        with ThreadPoolExecutor(max_workers=self._parallel) as executor:
            for job in jobs:
                executor.submit(job, self._lock)
        duration = round((time.time() - start) / 3600, 1)
        self._log_finish(experiment, duration)

    def _dump_definition(self, experiment, definition):
        if experiment:
            name = os.path.basename(experiment)
            dump_yaml(definition, experiment, name + '.yaml')

    def _log_finish(self, experiment, duration):
        message = 'Congratulations, benchmark finished after {} hours'
        print_headline(message.format(duration), style='=')
        if experiment:
            print('Find results in', experiment)

    def _create_jobs(self, experiment, definition):
        combinations = itertools.product(
            range(definition.repeats), definition.envs, definition.algorithms)
        for repeat, env_name, algo_def in combinations:
            args = experiment, env_name, algo_def, repeat, definition
            yield self._create_job(*args)

    def _create_job(self, experiment, env_name, algo_def, repeat, definition):
        directory = self._task_directory(
            experiment, env_name, algo_def.name, repeat, definition.repeats)
        observs, actions = self._determine_interface(env_name)
        train = Task(
            observs, actions, directory,
            definition.epochs * algo_def.train_steps,
            definition.epochs, True)
        test = Task(
            observs, actions, directory,
            (definition.epochs + 1) * definition.test_steps,
            definition.epochs + 1, False)
        prefix = '{} on {} ({}):'.format(algo_def.name, env_name, repeat)
        return Job(train, test, env_name, algo_def, prefix, self._videos)

    def _start_experiment(self, definition):
        print_headline('Start experiment', style='=')
        if not self._directory:
            print('Dry run; no results will be stored!')
            return None
        timestamp = time.strftime('%Y-%m-%dT%H-%M-%S', time.gmtime())
        name = '{}-{}'.format(timestamp, definition.experiment)
        experiment = os.path.join(self._directory, name)
        print('Result will be stored in', experiment)
        return experiment

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

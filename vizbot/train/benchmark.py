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
from vizbot.train import Trainer, GymEnv
from vizbot.utility import use_attrdicts, ensure_directory


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
        definition = self._load_definition(definition)
        experiment = self._start_experiment(definition.experiment)
        if experiment:
            name = os.path.basename(experiment)
            self._dump_yaml(definition, experiment, name + '.yaml')
        tasks = itertools.product(
            range(definition.repeats), definition.envs, definition.algorithms)
        with ThreadPoolExecutor(max_workers=self._parallel) as executor:
            for repeat, env, algorithm in tasks:
                executor.submit(
                    self._start_task, repeat, env, algorithm, experiment,
                    definition)
        message = 'Congratulations, benchmark finished after {} hours'
        duration = round((time.time() - start) / 3600, 1)
        self._print_headline(message.format(duration), style='=')
        if experiment:
            print('Find results in', experiment)

    def _start_task(self, repeat, env, algorithm, experiment, definition):
        template = '{{}}-{{:0>{}}}'.format(len(str(definition.repeats - 1)))
        message = 'Train {} on {} (Repeat {})'
        if self._parallel == 1:
            self._print_headline(message.format(algorithm.name, env, repeat))
        name = '-'.join(re.findall(r'[a-z0-9]+', algorithm.name.lower()))
        agent_dir = template.format(name, repeat)
        directory = experiment and os.path.join(experiment, env, agent_dir)
        self._run_task(directory, env, algorithm, repeat, definition)

    def _run_task(self, directory, env, config, repeat, definition):
        prefix = '{} on {} ({}):'.format(config.name, env, repeat)
        self._directory and self._dump_yaml(
            config, self._directory, 'algorithm.yaml')
        try:
            algorithm = self._create_algorithm(config.type, config, env)
            trainer = Trainer(
                task, env, algorithm, definition.epochs,
                config.train_steps, definition.test_steps, self._videos)
            for epoch, score in enumerate(trainer):
                if not epoch:
                    message = 'Before training average score {:.2f}'
                    message = message.format(score)
                else:
                    message = 'Epoch {} timestep {} average score {:.2f}'
                    message = message.format(epoch, trainer.timestep, score)
                print(prefix, message)
        except Exception as e:
            with self._lock:
                print(prefix, 'Failed due to exception:')
                print(e)
            if self._stacktraces:
                traceback.print_exc()

    def _create_algorithm(self, type_, config, env_name):
        example_env = GymEnv(env_name)
        algorithm = type_(example_env.interface, config)
        example_env.close()
        return algorithm

    def _start_experiment(self, name):
        self._print_headline('Start experiment', style='=')
        if not self._directory:
            print('Dry run; no results will be stored!')
            return None
        timestamp = time.strftime('%Y-%m-%dT%H-%M-%S', time.gmtime())
        name = '{}-{}'.format(timestamp, name)
        experiment = os.path.join(self._directory, name)
        print('Result will be stored in', experiment)
        return experiment

    def _load_definition(self, definition):
        with open(os.path.expanduser(definition)) as file_:
            definition = yaml.load(file_)
        definition = use_attrdicts(definition)
        definition.experiment = str(definition.experiment)
        definition.epochs = int(float(definition.epochs))
        definition.test_steps = int(float(definition.test_steps))
        definition.repeats = int(float(definition.repeats))
        definition.envs = list(self._load_envs(definition.envs))
        definition.algorithms = [
            self._load_algorithm(x) for x in definition.algorithms]
        self._validate_definition(definition)
        return definition

    def _load_envs(self, envs):
        available_envs = [x.id for x in gym.envs.registry.all()]
        for env in envs:
            if env not in available_envs:
                raise KeyError('unknown env name {}'.format(env))
            yield env

    def _load_algorithm(self, config):
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

    def _validate_definition(self, definition):
        names = [x.name for x in definition.algorithms]
        if len(set(names)) < len(names):
            raise KeyError('each algorithm must have an unique name')
        if not all(hasattr(x, 'train_steps') for x in definition.algorithms):
            raise KeyError('each algorithm must have a training duration')

    def _dump_yaml(self, data, *path):
        def convert(obj):
            if isinstance(obj, dict):
                obj = {k: v for k, v in obj.items() if not k.startswith('_')}
                return {convert(k): convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(x) for x in obj]
            if isinstance(obj, type):
                return obj.__name__
            return obj
        filename = os.path.join(*path)
        ensure_directory(os.path.dirname(filename))
        with open(filename, 'w') as file_:
            yaml.safe_dump(convert(data), file_, default_flow_style=False)

    def _print_headline(self, *message, style='-', minwidth=40):
        with self._lock:
            message = ' '.join(message)
            width = max(minwidth, len(message))
            print('\n' + style * width)
            print(message)
            print(style * width + '\n')

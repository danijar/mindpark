import re
import itertools
import os
import time
import traceback
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
import yaml
import gym
import numpy as np
import vizbot.env
import vizbot.agent
from vizbot.core import Trainer, Agent, StopTraining
from vizbot.utility import AttrDict, use_attrdicts, ensure_directory


class Benchmark:

    """
    Train each agent on each environment for multiple repeats and store
    statistics and recordings in the experiment directory.
    """

    def __init__(self, directory=None, parallel=1, videos=False,
                 experience=False, stacktraces=False):
        if directory:
            directory = os.path.abspath(os.path.expanduser(directory))
        self._directory = directory
        self._parallel = parallel
        self._videos = videos
        self._experience = experience
        self._stacktraces = stacktraces
        self._lock = Lock()

    def __call__(self, definition):
        start = time.time()
        definition = self._load_definition(definition)
        experiment = self._start_experiment(definition.experiment)
        experiment and self._dump_yaml(definition,experiment,'experiment.yaml')
        tasks = itertools.product(
            range(definition.repeats), definition.envs, definition.agents)
        with ThreadPoolExecutor(max_workers=self._parallel) as executor:
            for repeat, env, agent in tasks:
                executor.submit(self._start_task,
                    repeat, env, agent, experiment, definition)
        message = 'Congratulations, benchmark finished after {} hours'
        duration = round((time.time() - start) / 3600, 1)
        self._print_headline(message.format(duration), style='=')
        if experiment:
            print('Find results in', experiment)

    def _start_task(self, repeat, env, agent, experiment, definition):
        template = '{{}}-{{:0>{}}}'.format(len(str(definition.repeats - 1)))
        message = 'Train {} on {} (Repeat {})'
        if self._parallel == 1:
            self._print_headline(message.format(agent.name, env, repeat))
        name = '-'.join(re.findall(r'[a-z0-9]+', agent.name.lower()))
        agent_dir = template.format(name, repeat)
        directory = experiment and os.path.join(experiment, env, agent_dir)
        self._run_task(directory, env, agent, definition)

    def _run_task(self, directory, env, agent, definition):
        prefix = '{} on {}:'.format(agent.name, env)
        trainer = Trainer(directory, env,
            timesteps=definition.timesteps,
            epoch_length=definition.epoch_length,
            prefix=prefix,
            experience=self._experience,
            videos=self._videos)
        config = agent.type.defaults()
        if 'type' in config or 'name' in config:
            print('Warning: Override reserved config keys.')
        config.update(agent)
        directory and self._dump_yaml(config, directory, 'agent.yaml')
        try:
            agent.type(trainer, AttrDict(config))()
        except Exception as e:
            with self._lock:
                print(prefix, 'Failed due to exception:')
                print(e)
            if self._stacktraces:
                traceback.print_exc()
        except StopTraining:
            pass
        scores = trainer.scores
        score = round(sum(scores) / len(scores), 3) if scores else 'none'
        print(prefix, 'Finished with average score {}'.format(score))

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
        definition.timesteps = int(float(definition.timesteps))
        definition.epoch_length = int(float(definition.epoch_length))
        definition.repeats = int(float(definition.repeats))
        definition.envs = list(self._load_envs(definition.envs))
        definition.agents = list(self._load_agents(definition.agents))
        self._validate_definition(definition)
        return definition

    def _load_envs(self, envs):
        available_envs = [x.id for x in gym.envs.registry.all()]
        for env in envs:
            if env not in available_envs:
                raise KeyError('unknown env name {}'.format(env))
            yield env

    def _load_agents(self, agents):
        for agent in agents:
            if not hasattr(vizbot.agent, agent.type):
                raise KeyError('unknown agent type {}'.format(agent.type))
            agent.type = getattr(vizbot.agent, agent.type)
            if not issubclass(agent.type, vizbot.core.Agent):
                raise KeyError('{} is not an agent'.format(agent.type))
            agent.name = str(agent.name)
            yield agent

    def _validate_definition(self, definition):
        def warn(message):
            print('Warning:', message)
            input('Press return to continue.')
        timesteps = definition.repeats * definition.timesteps
        names = [x.name for x in definition.agents]
        if len(set(names)) < len(names):
            raise KeyError('each algorithm must have an unique name')
        if self._experience and timesteps >= 10000:
            warn('Storing 10000+ timesteps consumes a lot of disk space.')
        if not self._videos and timesteps >= 10000:
            warn('Training 10000+ timesteps. Consider capturing videos.')
        if definition.epoch_length > definition.timesteps:
            warn('Less than one epoch of timesteps.')

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

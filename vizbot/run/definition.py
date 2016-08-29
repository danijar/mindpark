import os
import sys
import yaml
import vizbot.env  # Register custom envs.
import gym
import vizbot.algorithm
from vizbot.utility import use_attrdicts


class Definition:

    def __new__(cls, filepath):
        with open(os.path.expanduser(filepath)) as file_:
            definition = yaml.load(file_)
        definition = use_attrdicts(definition)
        definition.epochs = int(float(definition.epochs))
        definition.test_steps = int(float(definition.test_steps))
        definition.repeats = int(float(definition.repeats))
        definition.envs = list(cls._load_envs(definition.envs))
        definition.algorithms = [
            cls._load_algorithm(x) for x in definition.algorithms]
        cls._validate_definition(definition)
        return definition

    @classmethod
    def _load_envs(cls, envs):
        available_envs = [x.id for x in gym.envs.registry.all()]
        for env in envs:
            if env not in available_envs:
                raise KeyError('unknown env name {}'.format(env))
            yield env

    @classmethod
    def _load_algorithm(cls, algorithm):
        if not hasattr(vizbot.algorithm, algorithm.type):
            message = 'unknown algorithm type {}'
            raise KeyError(message.format(algorithm.type))
        algorithm.type = getattr(vizbot.algorithm, algorithm.type)
        algorithm.name = str(algorithm.name)
        algorithm.train_steps = int(float(algorithm.train_steps))
        if 'config' not in algorithm:
            algorithm['config'] = {}
        if not issubclass(algorithm.type, vizbot.core.Algorithm):
            raise KeyError('{} is not an algorithm'.format(algorithm.type))
        defaults = algorithm.type.defaults()
        for key in algorithm.config:
            if key not in defaults:
                message = "unknown config key '{}' for algorithm {}"
                raise KeyError(message.format(key, algorithm.type.__name__))
        return algorithm

    @classmethod
    def _validate_definition(cls, definition):
        names = [x.name for x in definition.algorithms]
        if len(set(names)) < len(names):
            raise KeyError('each algorithm must have an unique name')
        if not all(hasattr(x, 'train_steps') for x in definition.algorithms):
            raise KeyError('each algorithm must have a training duration')
        testing = hasattr(sys, '_called_from_test')
        if definition.epochs > 4 and not sys.flags.optimize and not testing:
            raise KeyError('use optimize flag when running many epochs')

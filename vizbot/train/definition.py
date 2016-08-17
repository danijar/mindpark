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
        definition.experiment = str(definition.experiment)
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
        defaults.update(config)
        return use_attrdicts(defaults)

    @classmethod
    def _validate_definition(cls, definition):
        names = [x.name for x in definition.algorithms]
        if len(set(names)) < len(names):
            raise KeyError('each algorithm must have an unique name')
        if not all(hasattr(x, 'train_steps') for x in definition.algorithms):
            raise KeyError('each algorithm must have a training duration')
        if definition.epochs > 4 and not sys.flags.optimize:
            raise KeyError('use optimize flag when running many epochs')

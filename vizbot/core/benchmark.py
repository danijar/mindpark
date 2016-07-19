import collections
import itertools
import os
import time
import numpy as np
import gym
from vizbot.core import Trainer, Agent, StopTraining
from vizbot.utility import ensure_directory


class Benchmark:

    def __init__(self, directory, repeats, timesteps,
                 videos=0, experience=False):
        """
        Train multiple agents on multiple environments and plot a comparison
        chart. Store all results in an unique experiment directory.

        Args:
            directory (path): Root directory for experiments.
            repeats (int): How often to train the agent on the same
                environment. Used to estimate the standard deviation.
            timesteps (int): Training time per repeat in frames.
            videos (int): Every how many episodes to record a video. Zero
                disables recording.
            experience (bool): Whether to store transition tuples as Numpy
                arrays. Requires a lot of disk space.
        """
        if directory:
            directory = os.path.abspath(os.path.expanduser(directory))
        self._directory = directory
        self._repeats = repeats
        self._timesteps = timesteps
        self._videos = videos
        self._experience = experience

    def __call__(self, name, envs, agents):
        """
        Train each agent on each environment for multiple repeats. Store gym
        monitorings and scores into sub directories of the experiment. Return
        the path to the experiment and the scores.
        """
        experiment = self._start_experiment(name)
        result = collections.defaultdict(dict)
        for env, agent in itertools.product(envs, agents):
            print('Benchmark', agent.__name__, 'on', env)
            directory = None
            if experiment:
                directory = os.path.join(
                    experiment, '{}-{}'.format(env, agent.__name__))
            scores = self._benchmark(directory, env, agent)
            print('Mean best return {}'.format(scores.max(1).mean()))
            result[env][agent] = scores
        if not experiment:
            return None, result
        return experiment, self.read(experiment)

    @staticmethod
    def read(experiment):
        """
        Read and return scores of an experiment from its sub directories.
        """
        result = collections.defaultdict(dict)
        for benchmark in self._get_subdirs(experiment):
            env, agent = os.path.basename(benchmark).rsplit('-', 1)
            result[env][agent] = []
            for repeat in self._get_subdirs(benchmark):
                scores = np.load(os.path.join(repeat, 'scores.npy'))
                result[env][agent].append(scores)
        return result

    def _benchmark(self, directory, env, agent):
        """
        Train an agent for several repeats and store and return scores the
        scores of each repeat and episode.
        """
        scores = []
        template = 'repeat-{:0>' + str(len(str(self._repeats - 1))) + '}'
        for repeat in range(self._repeats):
            subdirectory = None
            if directory:
                subdirectory = os.path.join(directory, template.format(repeat))
            trainer = Trainer(
                subdirectory, env, self._timesteps,
                self._videos, self._experience)
            try:
                agent(trainer)()
            except StopTraining:
                pass
            scores.append(trainer.scores)
        return scores

    def _start_experiment(self, name):
        if not self._directory:
            print('Start experiment. Dry run, no results will be saved.')
            return None
        timestamp = time.strftime('%Y-%m-%dT%H-%M-%S', time.gmtime())
        name = '{}-{}'.format(timestamp, name)
        experiment = os.path.join(self._directory, name)
        print('Start experiment', experiment)
        return experiment

    def _get_subdirs(directory):
        subdirs = os.listdir(directory)
        subdirs = [os.path.join(directory, x) for x in subdirs]
        subdirs = [x for x in subdirs if os.path.isdir(x)]
        return sorted(subdirs)

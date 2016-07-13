import collections
import itertools
import os
import time
import numpy as np
import gym
from vizbot.core import Agent
from vizbot.utility import ensure_directory


class Simulator:

    def __init__(self, root, repeats, episodes, recording=True):
        self._root = os.path.abspath(os.path.expanduser(root))
        self._repeats = repeats
        self._episodes = episodes
        self._recording = recording

    def __call__(self, name, envs, agents):
        """
        Train each agent on each environment. Store gym monitorings, rewards,
        and durations into sub directories of the experiment. Return the path
        to the experiment and the results.
        """
        assert all(issubclass(x, Agent) for x in agents)
        timestamp = time.strftime('%Y-%m-%dT%H-%M-%S', time.gmtime())
        experiment = os.path.join(self._root, '{}-{}'.format(timestamp, name))
        ensure_directory(experiment)
        print('Start experiment', experiment)
        message = 'Min duration {} Mean best reward {}'
        for env_name, agent_cls in itertools.product(envs, agents):
            directory = os.path.join(
                experiment, '{}-{}'.format(env_name, agent_cls.__name__))
            env = gym.make(env_name)
            print('Benchmark', agent_cls.__name__, 'on', env_name)
            reward, duration = self._benchmark(directory, env, agent_cls(env))
            print(message.format(reward.max(axis=1).mean(), duration.min()))
        result = self.read(experiment)
        return experiment, result

    @staticmethod
    def read(experiment):
        """
        Read and return results of an experiment from its sub directories.
        """
        benchmarks = os.listdir(experiment)
        benchmarks = [os.path.join(experiment, x) for x in benchmarks]
        benchmarks = [x for x in benchmarks if os.path.isdir(x)]
        result = collections.defaultdict(dict)
        for benchmark in benchmarks:
            env, agent = os.path.basename(benchmark).rsplit('-', 1)
            reward = np.load(os.path.join(benchmark, 'rewards.npy'))
            durations = np.load(os.path.join(benchmark, 'durations.npy'))
            result[env][agent] = reward
        return result

    def _benchmark(self, directory, env, agent):
        """
        Train an agent for several repeats and store statistics. Return the
        average and standard deviation of rewards along the episodes, and the
        minimum training duration.
        """
        ensure_directory(directory)
        rewards, durations = [], []
        template = 'repeat-{:0>' + str(len(str(self._repeats - 1))) + '}'
        for repeat in range(self._repeats):
            subdirectory = os.path.join(directory, template.format(repeat))
            reward, duration = self._simulate(subdirectory, env, agent)
            rewards.append(reward)
            durations.append(duration)
            print(' ' + '.' * (repeat + 1), end='\r', flush=True)
        print('')
        rewards, durations = np.array(rewards), np.array(durations)
        np.save(os.path.join(directory, 'rewards.npy'), rewards)
        np.save(os.path.join(directory, 'durations.npy'), durations)
        return rewards, durations

    def _simulate(self, directory, env, agent):
        """
        Train an agent in an environment and store its gym monitoring. Return
        rewards and training duration.
        """
        env.monitor.start(directory, None if self._recording else False)
        start, rewards = time.time(), []
        state, reward, done = env.reset(), 0, False
        for episode in range(self._episodes):
            rewards.append(0)
            while not done:
                action = agent(state, reward)
                state, reward, done, _ = env.step(action)
                rewards[-1] += reward
        duration = time.time() - start
        return rewards, duration

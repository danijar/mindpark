import collections
import itertools
import os
import time
import numpy as np
import gym
from vizbot.core import Agent
from vizbot.utility import ensure_directory


class Simulator:

    def __init__(self, root, repeats, episodes,
                 videos=True, experience=False):
        self._root = os.path.abspath(os.path.expanduser(root))
        self._repeats = repeats
        self._episodes = episodes
        self._videos = videos
        self._experience = experience

    def __call__(self, name, envs, agents):
        """
        Train each agent on each environment. Store gym monitorings, returns,
        and durations into sub directories of the experiment. Return the path
        to the experiment and the results.
        """
        assert all(issubclass(x, Agent) for x in agents)
        timestamp = time.strftime('%Y-%m-%dT%H-%M-%S', time.gmtime())
        experiment = os.path.join(self._root, '{}-{}'.format(timestamp, name))
        ensure_directory(experiment)
        print('Start experiment', experiment)
        message = 'Min duration {} Mean best return {}'
        for env, agent in itertools.product(envs, agents):
            print('Benchmark', agent.__name__, 'on', env)
            directory = os.path.join(
                experiment, '{}-{}'.format(env, agent.__name__))
            returns, durations = self._benchmark(directory, env, agent)
            print(message.format(returns.max(axis=1).mean(), durations.min()))
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
            returns = np.load(os.path.join(benchmark, 'returns.npy'))
            durations = np.load(os.path.join(benchmark, 'durations.npy'))
            result[env][agent] = returns
        return result

    def _benchmark(self, directory, env_name, agent_cls):
        """
        Train an agent for several repeats and store statistics. Return the
        returns and durations of each eposide.
        """
        ensure_directory(directory)
        returns, durations = [], []
        template = 'repeat-{:0>' + str(len(str(self._repeats - 1))) + '}'
        for repeat in range(self._repeats):
            subdirectory = os.path.join(directory, template.format(repeat))
            env = gym.make(env_name)
            agent = agent_cls(env)
            return_, duration = self._train(subdirectory, env, agent)
            returns.append(return_)
            durations.append(duration)
            print(' ' + '.' * (repeat + 1), end='\r', flush=True)
        print('')
        returns, durations = np.array(returns), np.array(durations)
        np.save(os.path.join(directory, 'returns.npy'), np.array(returns))
        np.save(os.path.join(directory, 'durations.npy'), np.array(durations))
        return returns, durations

    def _train(self, directory, env, agent):
        """
        Train an agent in an environment and store its gym monitoring. Return
        returns of each episode and the overall duration.
        """
        env.monitor.start(directory, None if self._videos else False)
        returns, states, rewards, start = [], [], [], time.time()
        for episode in range(self._episodes):
            return_, state, reward = self._episode(env, agent)
            returns.append(return_)
            states += state
            rewards += reward
        if self._experience:
            states, rewards = np.array(states), np.array(rewards)
            filepath = os.path.join(directory, 'experience.npz')
            np.savez_compressed(filepath, states=states, rewards=rewards)
        return np.array(returns), time.time() - start

    def _episode(self, env, agent):
        """
        Reset the environment and simulate the agent for one episode. Return
        the return and, if recorded, the experience.
        """
        return_, states, rewards = 0, [], []
        state, reward, done = env.reset(), 0, False
        agent.begin()
        while not done:
            action = agent.step(state)
            state, reward, done, _ = env.step(action)
            agent.feedback(reward)
            return_ += reward
            if self._experience:
                states.append(state)
                rewards.append(reward)
        agent.end()
        return return_, states, rewards

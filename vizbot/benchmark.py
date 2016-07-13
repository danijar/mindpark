import argparse
import collections
import itertools
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import gym
import vizbot.agent
from vizbot.utility import ensure_directory, AttrDict


class ComparisonPlot:

    COLORS = ('green', 'blue', 'red', 'yellow')

    LABEL_ARGS = AttrDict(
        loc='best', frameon=False, fontsize='medium', labelspacing=0)

    def __init__(self, ncols, title):
        self._ncols = ncols
        self._fig = plt.figure(figsize=(12, 4))
        self._fig.suptitle(title, fontsize=16)
        self._index = 1

    def add(self, title, xlabel, ylabel, **lines):
        lines = sorted(lines.items(), key=lambda x: -x[1].sum())
        ax = self._next_plot()
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        for index, (label, line) in enumerate(lines):
            color = self.COLORS[index]
            self._plot(ax, label, line, color)
        ax.legend(**self.LABEL_ARGS)

    def save(self, filepath):
        self._fig.tight_layout(rect=[0, 0, 1, .93])
        self._fig.savefig(filepath, dpi=300)

    def _next_plot(self):
        if self._index > self._ncols:
            raise RuntimeError
        ax = self._fig.add_subplot(1, self._ncols, self._index)
        self._index += 1
        return ax

    def _plot(self, ax, label, line, color):
        means = np.cumsum(line.mean(axis=0))
        stds = line.std(axis=0)
        area = np.arange(len(means)), means - stds, means + stds
        ax.fill_between(*area, color=color, alpha=0.15)
        ax.plot(means, label=label, color=color)


def read_comparison(directory):
    benchmarks = os.listdir(directory)
    benchmarks = [os.path.join(directory, x) for x in benchmarks]
    benchmarks = [x for x in benchmarks if os.path.isdir(x)]
    rewards = collections.defaultdict(dict)
    for benchmark in benchmarks:
        env, agent = os.path.basename(benchmark).rsplit('-', 1)
        reward = np.load(os.path.join(benchmark, 'rewards.npy'))
        durations = np.load(os.path.join(benchmark, 'durations.npy'))
        rewards[env][agent] = reward
    return rewards


def train(directory, env, agent, episodes):
    """
    Train an agent in an environment and store its gym monitoring. Return
    rewards and training duration.
    """
    ensure_directory(directory)
    env.monitor.start(directory)
    start, rewards = time.time(), []
    state, reward, done = env.reset(), 0, False
    for episode in range(episodes):
        rewards.append(0)
        while not done:
            action = agent(state, reward)
            state, reward, done, _ = env.step(action)
            rewards[-1] += reward
    duration = time.time() - start
    return rewards, duration


def benchmark(root, env, agent, repeats, episodes):
    """
    Train an agent for several repeats and store statistics. Return the average
    and standard deviation of rewards along the episodes, and the minimum
    training duration.
    """
    rewards, durations = [], []
    template = 'repeat-{:0>' + str(len(str(episodes - 1))) + '}'
    for repeat in range(repeats):
        directory = os.path.join(root, template.format(repeat))
        reward, duration = train(directory, env, agent, episodes)
        rewards.append(reward)
        durations.append(duration)
    rewards, durations = np.array(rewards), np.array(durations)
    np.save(os.path.join(root, 'rewards.npy'), rewards)
    np.save(os.path.join(root, 'durations.npy'), durations)
    return rewards, durations


def compare(root, envs, agents, repeats, episodes):
    """
    Train each agent on each environment. Store gym monitorings and rewards in
    sub directories. Generate a plot comparing the rewards.
    """
    root = os.path.expanduser(root)
    message = 'Min duration {} Mean best reward {}'
    for env_name, agent_name in itertools.product(envs, agents):
        directory = os.path.join(root, '{}-{}'.format(env_name, agent_name))
        env = gym.make(env_name)
        agent = getattr(vizbot.agent, agent_name)(env)
        print('Benchmark', agent_name, 'on', env_name)
        rewards, durations = benchmark(
            directory, env, agent, repeats, episodes)
        print(message.format(rewards.max(axis=1).mean(), durations.min()))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--envs', nargs='+',
        default=['DoomDeathmatch-v0'])
    parser.add_argument(
        '-a', '--agents', nargs='+',
        default=['Noop'])
    parser.add_argument(
        '-r', '--repeats', type=int,
        default='10')
    parser.add_argument(
        '-n', '--episodes', type=int,
        default='10')
    parser.add_argument(
        '-o', '--directory',
        default='~/experiment/gym')
    parser.add_argument(
        '-e', '--experiment',
        default='experiment')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    timestamp = time.strftime('%Y-%m-%dT%H-%M-%S', time.gmtime())
    directory = os.path.abspath(os.path.expanduser(args.directory))
    directory = os.path.join(
        directory, '{}-{}'.format(timestamp, args.experiment))
    print('Start experiment', directory)
    compare(directory, args.envs, args.agents, args.repeats, args.episodes)
    rewards = read_comparison(directory)
    plot = ComparisonPlot(len(rewards), os.path.basename(directory))
    for env, agents in rewards.items():
        plot.add(env, 'Training Episode', 'Cumulative Reward', **agents)
    plot.save(os.path.join(directory, 'comparison.pdf'))


if __name__ == '__main__':
    main()

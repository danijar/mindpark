import time
import os
import argparse
import collections
import numpy as np
import matplotlib.pyplot as plt
import vizbot.agent
from vizbot.utility import ensure_directory


def read_comparison(self, directory):
    benchmarks = [x for x in os.listdir(directory) if os.path.isdir(x)]
    rewards = collections.defaultdict(dict)
    for benchmark in benchmarks:
        env, agent = os.path.basename(benchmark).rsplit('-', 1)
        reward = np.load(os.path.join(benchmark, 'rewards.npy'))
        durations = np.load(os.path.join(benchmark, 'durations.npy'))
        rewards[env][agent] = reward
    return rewards


def plot_comparison(self, directory, rewards):
    fig, ax = plt.subplots(ncols=len(rewards), figsize=(12, 4))
    for index, env in enumerate(sorted(rewards.keys())):
        for agent in rewards[env]:
            means = rewards[env][agent].mean(axis=0)
            ax[index].plot(means, label=agent)
    fig.savefig(os.path.join(directory, 'comparison.pdf'), dpi=300)


def train(directory, env, agent, episodes):
    """
    Train an agent in an environment and store its gym monitoring. Return
    rewards and training duration.
    """
    state, reward, done = env.reset(), 0, False
    ensure_directory(directory)
    env.monitor.start(directory)
    start, rewards = time.time(), []
    for episode in episodes:
        rewards.append(0)
        while not done:
            action = agent(state, reward)
            state, reward, done = env.step(action)
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
    np.save(os.path.join(directory, 'rewards.npy'), rewards)
    np.save(os.path.join(directory, 'durations.npy'), durations)
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
        '-d', '--directory',
        default='~/experiment/gym')
    parser.add_argument(
        '-e', '--experiment',
        default='experiment')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    timestamp = time.strftime('%Y-%m-%dT%H-%M-%', time.gmtime())
    directory = os.path.abspath(os.path.expanduser(args.directory))
    directory = os.path.join(directory, '{}-{}'.format(timestamp, args.name))
    print('Start experiment', directory)
    compare(directory, args.envs, args.agents, args.repeats, args.episodes)
    rewards = read_comparison(directory)
    plot_comparison(directory, rewards)


if __name__ == '__main__':
    main()

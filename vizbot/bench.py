import argparse
import logging
import os
import vizbot.agent
from vizbot.utility import DeviationFigure
from vizbot.core import Simulator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--envs', nargs='+',
        default=['DoomDeathmatch-v0'])
    parser.add_argument(
        '-a', '--agents', nargs='+',
        default=['Noop', 'Random'])
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
    parser.add_argument(
        '-c', '--recording', action='store_true',
        default=False)
    args = parser.parse_args()
    return args


def plot_result(directory, rewards):
    plot = DeviationFigure(len(rewards), os.path.basename(directory))
    for env, agents in rewards.items():
        plot.add(env, 'Training Episode', 'Cumulative Reward', **agents)
    plot.save(os.path.join(directory, 'comparison.pdf'))


def main():
    args = parse_args()
    simulator = Simulator(
        args.directory, args.repeats, args.episodes, args.recording)
    agents = [getattr(vizbot.agent, x) for x in args.agents]
    logging.getLogger('gym').setLevel(logging.WARNING)
    experiment, result = simulator(args.experiment, args.envs, agents)
    plot_result(experiment, result)


if __name__ == '__main__':
    main()

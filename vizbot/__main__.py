import sys
import argparse
import logging
import os
import vizbot.agent
import vizbot.env
from vizbot.utility import DeviationFigure, color_stack_trace
from vizbot.core import Simulator


def parse_args():
    nearest_int = lambda x: int(float(x))
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-d', '--envs', nargs='+',
        help='list of gym environments',
        default=['SimpleDoom-v0'])
    parser.add_argument(
        '-a', '--agents', nargs='+',
        help='list of agents from vizbot.agent module',
        default=['KeyboardDoom', 'Random', 'Noop'])
    parser.add_argument(
        '-r', '--repeats', type=nearest_int,
        help='repeat training to estimate deviation',
        default=5)
    parser.add_argument(
        '-e', '--epochs', type=nearest_int,
        help='how long to train an agent on an environment',
        default=100)
    parser.add_argument(
        '-n', '--timesteps', type=nearest_int,
        help='the number of timesteps per epoch',
        default=50000)
    parser.add_argument(
        '-o', '--directory',
        help='root folder for all experiments',
        default='~/experiment/gym')
    parser.add_argument(
        '-l', '--experiment',
        help='name of this experiment, gets timestamp prefix',
        default='experiment')
    parser.add_argument(
        '-x', '--dry-run', action='store_true',
        help='no not monitor or store results',
        default=False)
    parser.add_argument(
        '-v', '--videos', action='store_true',
        help='store videos in mp4 format every few episodes',
        default=False)
    parser.add_argument(
        '-c', '--experience', action='store_true',
        help='store all transition tuples in numpy format',
        default=False)
    args = parser.parse_args()
    return args


def validate_args(args):
    timesteps = args.repeats * args.epochs * args.timesteps
    if args.experience and timesteps >= 10000:
        print('Warning:', 'Storing 10000+ timesteps consumes a lot of disk space.')
        input('Press return to continue.')
    if not args.videos and timesteps >= 10000:
        print('Warning:', 'Training 10000+ timesteps. Consider capturing videos.')
        input('Press return to continue.')


def plot_result(directory, rewards):
    plot = DeviationFigure(len(rewards), os.path.basename(directory))
    for env, agents in rewards.items():
        plot.add(env, 'Training Epochs', 'Average Reward', **agents)
    plot.save(os.path.join(directory, 'comparison.pdf'))


def main():
    sys.excepthook = color_stack_trace
    args = parse_args()
    validate_args(args)
    simulator = Simulator(
        args.directory, args.repeats, args.epochs, args.timesteps,
        args.dry_run, args.videos, args.experience)
    agents = [getattr(vizbot.agent, x) for x in args.agents]
    logging.getLogger('gym').setLevel(logging.WARNING)
    experiment, result = simulator(args.experiment, args.envs, agents)
    if not args.dry_run:
        plot_result(experiment, result)


if __name__ == '__main__':
    main()

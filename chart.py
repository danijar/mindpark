import sys
import argparse
import glob
import os
import numpy as np
from vizbot.core import Benchmark
from vizbot.utility import EpochFigure, color_stack_trace


def parse_args():
    nearest_int = lambda x: int(float(x))
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'experiment',
        help='name of the experiment, can be a glob expresstion',
        default='experiment')
    parser.add_argument(
        '-e', '--epoch-size', type=nearest_int,
        help='how to group average scores in chart and prints',
        default=5e4)
    args = parser.parse_args()
    return args


def timesteps(durations):
    return max(max(max(
        np.sum(repeat[:-1])
        for repeat in agent)
        for agent in env.values())
        for env in durations.values())


def plot_result(args, directory, scores, durations):
    title = os.path.basename(directory)
    epochs = (timesteps(durations) + args.epoch_size - 1) // args.epoch_size
    if epochs < 1:
        raise ValueError('epoch size must be smaller than the total duration')
    plot = EpochFigure(len(scores), title, epochs, args.epoch_size)
    for env in scores:
        score, duration = scores[env], durations[env]
        plot.add(env, 'Training Epochs', 'Average Reward', score, duration)
    plot.save(os.path.join(directory, 'comparison.pdf'))


def main():
    sys.excepthook = color_stack_trace
    args = parse_args()
    args.experiment = os.path.expanduser(args.experiment)
    if '*' in args.experiment:
        paths = glob.glob(args.experiment)
        if not paths:
            raise ValueError('glob does not match any path')
        if len(paths) > 1:
            print('Matches:', '\n'.join(paths))
            raise ValueError('glob must match exactly one path')
        args.experiment = paths[0]
    scores, durations = Benchmark.read(args.experiment)
    plot_result(args, args.experiment, scores, durations)


if __name__ == '__main__':
    main()

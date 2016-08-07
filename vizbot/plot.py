import sys
import argparse
import glob
import os
import yaml
import json
import collections
import numpy as np
from vizbot.utility import (
    use_attrdicts, get_subdirs, color_stack_trace, natural_sorted)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'experiments', nargs='?', default='~/experiment/gym/*',
        help='glob expresstion of one or more experiment directories')
    parser.add_argument(
        '-e', '--extension', default='pdf',
        help='filename extension of the plot; selects output format')
    parser.add_argument(
        '-f', '--force', action='store_true', default=False,
        help='replace existing plots')
    parser.add_argument(
        '-r', '--resolution', type=int, default=1,
        help='amount of plotted points per epoch')
    args = parser.parse_args()
    return args


def collect_stats(directory):
    """
    Read scores and durations of a monitoring directory. Return them as nested
    lists over monitorings and episodes.
    """
    scores, durations = [], []
    statistics = glob.glob(os.path.join(directory, '*.stats.json'))
    for stats in natural_sorted(statistics):
        with open(stats) as file_:
            epoch = use_attrdicts(json.load(file_))
        assert epoch.timestamps == sorted(epoch.timestamps)
        scores.append(epoch.episode_rewards)
        durations.append(epoch.episode_lengths)
    return scores, durations


def read_result(experiment):
    """
    Read scores and durations of an experiment. Return them as dictionaries
    indexed by environment and agent, containing nested lists over repeats,
    epochs and episodes.
    """
    scores, durations = {}, {}
    for env_dir in get_subdirs(experiment):
        env = os.path.basename(env_dir)
        scores[env] = collections.defaultdict(list)
        durations[env] = collections.defaultdict(list)
        for directory in get_subdirs(env_dir):
            _, repeat = os.path.basename(directory).rsplit('-', 1)
            agent = read_yaml(directory, 'agent.yaml').name
            score, duration = collect_stats(directory)
            scores[env][agent].append(score)
            durations[env][agent].append(duration)
    return scores, durations


def plot_experiment(experiment, filename, resolution):
    from vizbot.utility import EpochFigure
    if not os.path.isfile(os.path.join(experiment, 'experiment.yaml')):
        raise ValueError(experiment + ' does not contain a definition')
    definition = read_yaml(experiment, 'experiment.yaml')
    scores, durations = read_result(experiment)
    plot = EpochFigure(
        len(scores), definition.experiment, definition.epochs, resolution)
    for env in sorted(scores.keys()):
        if not len(scores[env]):
            continue
        plot.add(env, 'Training Epoch', 'Average Return', scores[env])
    plot.save(os.path.join(experiment, filename))


def read_yaml(*path):
    path = os.path.join(*path)
    with open(path) as file_:
        return use_attrdicts(yaml.load(file_))


def main():
    sys.excepthook = color_stack_trace
    args = parse_args()
    args.experiments = os.path.expanduser(args.experiments)
    paths = glob.glob(args.experiments)
    if not paths:
        print('The glob expression does not match any path.')
    for path in paths:
        if not os.path.isfile(os.path.join(path, 'experiment.yaml')):
            continue
        filename = os.path.basename(path) + '.' + args.extension
        if os.path.isfile(os.path.join(path, filename)) and not args.force:
            print('Skip existing plot', filename)
            continue
        print('Generate plot', filename)
        plot_experiment(path, filename, args.resolution)


if __name__ == '__main__':
    main()

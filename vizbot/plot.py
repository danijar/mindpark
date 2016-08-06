import sys
import argparse
import glob
import os
import yaml
import json
import collections
import numpy as np
from vizbot.utility import use_attrdicts, get_subdirs, color_stack_trace


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
    args = parser.parse_args()
    return args


def collect_stats(directory):
    timestamps, scores, durations = [], [], []
    for stats in glob.glob(os.path.join(directory, '*.stats.json')):
        with open(stats) as file_:
            stats = use_attrdicts(json.load(file_))
        timestamps += stats.timestamps
        scores += stats.episode_rewards
        durations += stats.episode_lengths
    if not len(timestamps) == len(scores) == len(durations):
        raise ValueError('inconsistent monitoring files detected')
    order = np.argsort(timestamps)
    scores, durations = np.array(scores), np.array(durations)
    scores, durations = scores[order], durations[order]
    return scores, durations


def read_result(experiment):
    scores, durations = {}, {}
    for env_dir in get_subdirs(experiment):
        env = os.path.basename(env_dir)
        scores[env] = collections.defaultdict(list)
        durations[env] = collections.defaultdict(list)
        for directory in get_subdirs(env_dir):
            agent, repeat = os.path.basename(directory).rsplit('-', 1)
            score, duration = collect_stats(directory)
            scores[env][agent].append(score)
            durations[env][agent].append(duration)
    return scores, durations


def plot_experiment(experiment, filename):
    from vizbot.utility import EpochFigure
    if not os.path.isfile(os.path.join(experiment, 'experiment.yaml')):
        raise ValueError(experiment + ' does not contain a definition')
    definition = use_attrdicts(read_yaml(experiment, 'experiment.yaml'))
    scores, durations = read_result(experiment)
    resolution = 1
    plot = EpochFigure(
        len(scores), definition.experiment, resolution,
        definition.epochs, definition.test_steps)
    for env in sorted(scores.keys()):
        score, duration = scores[env], durations[env]
        if not len(score):
            continue
        plot.add(env, 'Training Epoch', 'Average Return', score, duration)
    plot.save(os.path.join(experiment, filename))


def max_timesteps(durations):
    return max(max(max(
        np.sum(repeat[:-1])
        for repeat in agent)
        for agent in env.values())
        for env in durations.values())


def read_yaml(*path):
    path = os.path.join(*path)
    with open(path) as file_:
        return yaml.load(file_)


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
        plot_experiment(path, filename)


if __name__ == '__main__':
    main()

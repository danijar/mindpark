import os
import sys
import argparse
from mindpark.stats.stats import Stats
from mindpark.utility import get_subdirs


def parse_args(args):
    parser = argparse.ArgumentParser(
        'mindpark stats',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'experiment',
        help='part of one or more experiment names')
    parser.add_argument(
        '-d', '--directory', default='~/experiment/mindpark/',
        help='root directory for all experiments')
    parser.add_argument(
        '-t', '--type', default='pdf',
        help='file extension of the plots, determines the output format')
    parser.add_argument(
        '-m', '--metrics', nargs='*', default=None,
        help='names of the metrics to display; defaults to all metrics')
    parser.add_argument(
        '-f', '--force', action='store_true', default=False,
        help='overwrite existing plots')
    parser.add_argument(
        '-r', '--resolution', type=int, default=1,
        help='amount of plotted points per epoch')
    args = parser.parse_args(args)
    return args


def main(args):
    args = parse_args(args)
    args.directory = os.path.expanduser(args.directory)
    stats = Stats(args.type, args.metrics)
    for experiment in find_experiments(args):
        stats(experiment)


def find_experiments(args):
    patterns = args.experiment.split()
    for experiment in get_subdirs(args.directory):
        basename = os.path.basename(experiment)
        if all(x in basename for x in patterns):
            yield experiment


if __name__ == '__main__':
    main(sys.argv)

import sys
import argparse
import logging
from vizbot.utility import color_stack_trace
from vizbot.train.benchmark import Benchmark


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    nearest_int = lambda x: int(float(x))
    parser.add_argument(
        'definition',
        help='YAML file describing the experiment',
        default='definition/full.yaml')
    parser.add_argument(
        '-o', '--directory',
        help='root folder for all experiments',
        default='~/experiment/gym')
    parser.add_argument(
        '-p', '--parallel', type=int,
        help='how many agents to train at the same time',
        default=1)
    parser.add_argument(
        '-x', '--dry-run', action='store_true',
        help='no not monitor or store results',
        default=False)
    parser.add_argument(
        '-v', '--videos', type=nearest_int,
        help='if and how many videos to capture per testing phase',
        default=1)
    parser.add_argument(
        '-q', '--quiet', action='store_true',
        help='do not print stack traces for exceptions raised in agents',
        default=False)
    args = parser.parse_args()
    return args


def main():
    sys.excepthook = color_stack_trace
    args = parse_args()
    benchmark = Benchmark(
        args.directory if not args.dry_run else None,
        args.parallel, args.videos, not args.quiet)
    logging.getLogger('gym').setLevel(logging.WARNING)
    benchmark(args.definition)


if __name__ == '__main__':
    main()

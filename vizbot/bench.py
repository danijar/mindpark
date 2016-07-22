import sys
import argparse
import logging
from vizbot.utility import color_stack_trace


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    nearest_int = lambda x: int(float(x))
    parser.add_argument(
        '-o', '--directory',
        help='root folder for all experiments',
        default='~/experiment/gym')
    parser.add_argument(
        '-d', '--definition',
        help='YAML file describing the experiment',
        default='definition/full.yaml')
    parser.add_argument(
        '-p', '--parallel', type=int,
        help='how many agents to train at the same time',
        default=1)
    parser.add_argument(
        '-x', '--dry-run', action='store_true',
        help='no not monitor or store results',
        default=False)
    parser.add_argument(
        '-c', '--videos', type=nearest_int,
        help='if and every how many episodes to capture videos',
        default=1000)
    parser.add_argument(
        '-e', '--experience', action='store_true',
        help='store all transition tuples in numpy format',
        default=False)
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='print stack traces for exceptions raised in agents',
        default=False)
    parser.add_argument(
        '-i', '--ignored',
        help='ignored argument can be used to label the process',
        default='')
    args = parser.parse_args()
    return args


def main():
    sys.excepthook = color_stack_trace
    args = parse_args()
    from vizbot.core import Benchmark
    benchmark = Benchmark(
        args.directory if not args.dry_run else None,
        args.parallel, args.videos, args.experience, args.verbose)
    logging.getLogger('gym').setLevel(logging.WARNING)
    benchmark(args.definition)


if __name__ == '__main__':
    main()

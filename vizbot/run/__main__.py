import sys
import argparse
import logging
from vizbot.run.benchmark import Benchmark


def parse_args(args):
    parser = argparse.ArgumentParser(
        'vizbot run',
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
        help='how many algorithms to train in parallel',
        default=1)
    parser.add_argument(
        '-x', '--dry-run', action='store_true',
        help='do not store any results',
        default=False)
    parser.add_argument(
        '-v', '--videos', type=nearest_int,
        help='how many videos to capture per epoch',
        default=1)
    args = parser.parse_args(args)
    return args


def main(args):
    args = parse_args(args)
    benchmark = Benchmark(
        args.directory if not args.dry_run else None,
        args.parallel, args.videos)
    logging.getLogger('gym').setLevel(logging.WARNING)
    benchmark(args.definition)


if __name__ == '__main__':
    main(sys.argv)

import sys
import argparse
import logging
from mindpark.run.benchmark import Benchmark
from mindpark.utility import color_stack_trace


def parse_args(args):
    parser = argparse.ArgumentParser(
        'mindpark run',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'definition',
        help='YAML file describing the experiment')
    parser.add_argument(
        '-d', '--directory', default='~/experiment/mindpark',
        help='root directory for all experiments')
    parser.add_argument(
        '-p', '--parallel', type=int, default=1,
        help='how many algorithms to train in parallel')
    parser.add_argument(
        '-v', '--videos', type=int, default=1,
        help='how many videos to capture per epoch')
    parser.add_argument(
        '-x', '--dry-run', action='store_true', default=False,
        help='do not store any results')
    args = parser.parse_args(args)
    return args


def main(args):
    color_stack_trace()
    args = parse_args(args)
    directory = (not args.dry_run) and args.directory
    benchmark = Benchmark(directory, args.parallel, args.videos)
    logging.getLogger('gym').setLevel(logging.WARNING)
    benchmark(args.definition)


if __name__ == '__main__':
    main(sys.argv)

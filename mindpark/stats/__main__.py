import sys
import argparse


def parse_args(args):
    parser = argparse.ArgumentParser(
        'mindpark stats',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # ...
    args = parser.parse_args(args)
    return args


def main(args):
    args = parse_args(args)
    raise NotImplementedError


if __name__ == '__main__':
    main(sys.argv)

import importlib
import argparse
from mindpark.utility import color_stack_trace


def main():
    color_stack_trace()
    commands = ['run', 'stats', 'score']
    parser = argparse.ArgumentParser('mindpark')
    parser.add_argument('command', choices=commands)
    args, remaining = parser.parse_known_args()
    main = importlib.import_module('mindpark.{}.__main__'.format(args.command))
    main.main(remaining)


if __name__ == '__main__':
    main()

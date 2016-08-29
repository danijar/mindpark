import importlib
import argparse
from vizbot.utility import color_stack_trace


def main():
    color_stack_trace()
    commands = ['run', 'stats', 'score']
    parser = argparse.ArgumentParser('vizbot')
    parser.add_argument('command', choices=commands)
    args, remaining = parser.parse_known_args()
    main = importlib.import_module('vizbot.{}.__main__'.format(args.command))
    main.main(remaining)


if __name__ == '__main__':
    main()

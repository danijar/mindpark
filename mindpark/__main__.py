import importlib
import argparse
import mindpark as mp


def main():
    mp.utility.color_stack_trace()
    commands = ['run', 'stats']
    parser = argparse.ArgumentParser('mindpark', add_help=False)
    parser.add_argument('command', choices=commands)
    args, remaining = parser.parse_known_args()
    main = importlib.import_module('mindpark.{}.__main__'.format(args.command))
    main.main(remaining)


if __name__ == '__main__':
    main()

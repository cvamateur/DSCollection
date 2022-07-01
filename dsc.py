import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import cli_parser as cli


def main():
    parser = cli.get_main_parser()

    args = parser.parse_args()
    if args.tasks is None:
        print(parser.print_help())

    print(args)



if __name__ == '__main__':
    sys.exit(main())
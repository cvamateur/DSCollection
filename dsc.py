import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.tasks import cli as cli


def main():
    args = cli.get_main_parser().parse_args()

    if args.task == cli.TASK.VISUALIZE:
        from src.tasks.visualize import main
        args.input = args.input[0]
        return main(args)

    if args.task == cli.TASK.EXTRACT:
        return

    if args.task == cli.TASK.GENERATE:
        return

    if args.task == cli.TASK.COMBINE:
        from src.tasks.combine import main
        return main(args)

    if args.task == cli.TASK.SPLIT:
        from src.tasks.split import main
        return main(args)


if __name__ == '__main__':
    # visualize -i ~/Datasets/VOC2007 --ext .jpg -o ~/output --backend=1 --show-color-map

    sys.exit(main())

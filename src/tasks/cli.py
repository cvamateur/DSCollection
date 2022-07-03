import argparse
import sys

from .task import TASK


class ArgumentParser(argparse.ArgumentParser):

    def error(self, message: str):
        sys.stderr.write("error: %s\n" % message)
        self.print_help()
        sys.exit(2)


def get_main_parser() -> ArgumentParser:
    mainParser = build_common_arguments()
    tasks = mainParser.add_subparsers(title="tasks commands", dest="task", required=True)

    # Add generate task parser
    generateParser = tasks.add_parser(TASK.GENERATE)
    build_common_arguments(generateParser)
    add_output_dtypes_arguments(generateParser)
    add_generate_task_arguments(generateParser)

    # Add extract task parser
    extractParser = tasks.add_parser(TASK.EXTRACT)
    build_common_arguments(extractParser)
    add_output_dtypes_arguments(extractParser)
    add_extract_task_arguments(extractParser)

    # Add visualize task parser
    visualizeParser = tasks.add_parser(TASK.VISUALIZE)
    build_common_arguments(visualizeParser)
    add_visualize_task_arguments(visualizeParser)

    # Add split task parser
    splitParser = tasks.add_parser(TASK.SPLIT)
    build_common_arguments(splitParser)
    add_split_task_argument(splitParser)

    # Add combine task parser
    combineParser = tasks.add_parser(TASK.COMBINE)
    build_common_arguments(combineParser)
    add_combine_task_argument(combineParser)

    return mainParser


def build_common_arguments(parser: ArgumentParser = None, *, usage: str = None):
    if parser is None:
        parser = ArgumentParser(add_help=False, usage=usage)
    parser.add_argument("-i", "--input", nargs="+", help="Input video or directory.")
    parser.add_argument("-o", "--output", help="Output directory.")
    parser.add_argument("-c", "--contiguous", action="store_true", help="Rename files sequentially.")
    return parser


def add_process_arguments(parser: ArgumentParser):
    parser.add_argument("--crop-size", type=int, default=1296, help="Center crop size (default: 1296).")
    parser.add_argument("--roi-offset", type=int, default=0, help="Offset of roi cropping (default: 0).")
    parser.add_argument("--camera-shifts", default="(0,0,0,0)", help="Offsets of camera cropping (default (0,0,0,0)).")
    parser.add_argument("--grayscale", action="store_true", help="Convert to grayscale image")
    return parser


def add_output_dtypes_arguments(parser: ArgumentParser):
    parser.add_argument("--ext", help="Image file extension.")
    gp = parser.add_mutually_exclusive_group(required=False)
    gp.add_argument('--voc', action="store_true", help="Save as PascalVOC dataset.")
    gp.add_argument('--kitti', action="store_true", default=True, help="Save as KITTI dataset")
    gp.add_argument('--coco', action="store_true", help="Save as COCO dataset.")
    return parser


def add_generate_task_arguments(parser: ArgumentParser):
    parser.add_argument("-s", "--show", action="store_true", help="Show pipeline.")
    parser.add_argument("-m", "--model", action="append", default=[], help="Model config file.")


def add_extract_task_arguments(parser: ArgumentParser):
    ...


def add_visualize_task_arguments(parser: ArgumentParser):
    parser.add_argument("-l", "--label", action="store_true", help="Show label names.")
    parser.add_argument("-n", "--num-images", type=int, default=10, help="Number images to show.")
    parser.add_argument("--class-names", help="A string of class names seperated by ';'.")
    parser.add_argument("--backend", type=int, default=1, help="0: OpenCV; 1: Matplotlib (default)")
    parser.add_argument("--color-map", default="hsv", help="Colormap Name. Use --show-color-map show all names.")
    parser.add_argument("--show-color-map", action="store_true", help="Show all color map names.")
    parser.add_argument("--rows", type=int, help="Number rows in an image (Matplotlib only).")
    parser.add_argument("--cols", type=int, help="Number cols in an image (Matplotlib only).")
    parser.add_argument("--win-name", help="Window name (OpenCV only).")


def add_split_task_argument(parser: ArgumentParser):
    parser.add_argument("--keep", action="store_true", help="Keep original dataset.")
    parser.add_argument("--names", help="Filename suffix of each partition; separated by ';'.")
    gp = parser.add_mutually_exclusive_group(required=True)
    gp.add_argument("-ns", "--num-splits", type=int, help="Number partitions to equally split.")
    gp.add_argument("-np", "--num-per-each", type=int, help="Maximum number of samples in each part.")
    gp.add_argument("-nf", "--num-fractions", help="Fractions per each part; separated by ';', sums up to 1.0.")


def add_combine_task_argument(parser: ArgumentParser):
    parser.add_argument("--keep", action="store_true", help="Keep original dataset.")

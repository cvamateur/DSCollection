import argparse
from argparse import ArgumentParser

TASKS = ["extract", "generate", "visualize", "merge", "split"]


def get_main_parser() -> ArgumentParser:
    parser = ArgumentParser("Launch for DSCollection - a collection of dataset tools.")

    tasks = parser.add_subparsers(title="tasks commands", dest="tasks")

    # Add generate task parser
    generateParser = tasks.add_parser("generate", help="Generate new dataset from videos/images.")
    build_common_arguments(generateParser)
    add_generate_task_arguments(generateParser)

    # Add extract task parser
    extractParser = tasks.add_parser("extract", help="Extract specific classes from dataset.")
    build_common_arguments(extractParser)
    add_extract_task_arguments(extractParser)

    # Add visualize task parser
    visualizeParser = tasks.add_parser("visualize", help="Visualize dataset.")
    build_common_arguments(visualizeParser)
    add_visualize_task_arguments(visualizeParser)

    return parser


def build_common_arguments(parser: ArgumentParser = None):
    if parser is None:
        parser = ArgumentParser(add_help=False)
    parser.add_argument("-i", "--input", nargs="+", help="Input video or directory.")
    parser.add_argument("-o", "--output", default="./output", help="Output directory.")
    parser.add_argument("-c", "--contiguous", action="store_true", help="Rename files sequentially.")
    parser.add_argument("--ext", type=str, default=".png", help="Image file extension (default .png).")
    return parser


def build_process_arguments(parser: ArgumentParser = None):
    if parser is None:
        parser = ArgumentParser(add_help=False)
    parser.add_argument("--crop-size", type=int, default=1296, help="Center crop size (default: 1296).")
    parser.add_argument("--roi-offset", type=int, default=0, help="Offset of roi cropping (default: 0).")
    parser.add_argument("--camera-shifts", default="(0,0,0,0)", help="Offsets of camera cropping (default (0,0,0,0)).")
    parser.add_argument("--grayscale", action="store_true", help="Convert to grayscale image")
    return parser


def add_output_dtypes(parser: ArgumentParser):
    gp = parser.add_mutually_exclusive_group(required=False)
    gp.add_argument('--voc', action="store_true")
    gp.add_argument('--coco', action="store_true")
    gp.add_argument('--kitti', action="store_true", default=True)
    return parser


def add_generate_task_arguments(parser: ArgumentParser):
    add_output_dtypes(parser, True)
    parser.add_argument("-s", "--show", action="store_true", help="Show pipeline.")
    parser.add_argument("-m", "--model", action="append", default=[], help="Model config file.")


def add_extract_task_arguments(parser: ArgumentParser):
    add_output_dtypes(parser, True)


def add_visualize_task_arguments(parser: ArgumentParser):
    parser.add_argument("--clsNames", help="A string of class names seperated by ','.")

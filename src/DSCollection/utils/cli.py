import argparse
import sys
import ast

from .tasks import TASK
from DSCollection.core.dataset import DatasetType
from DSCollection._DSCollection import __version__ as v
from DSCollection._DSCollection.visualize import get_colormap_info

OPTSEP = ','


class ArgumentParser(argparse.ArgumentParser):

    def error(self, message: str):
        sys.stderr.write("error: %s\n" % message)
        self.print_help()
        sys.exit(2)

    def clear_actions(self):
        self._actions.clear()


class _EarlyStopAction(argparse.Action):
    """
    Custom action that if option presents, just prints stop-info
    and exit the program.
    """
    def __init__(self, option_strings,
                 stop=None,
                 dest=argparse.SUPPRESS,
                 default=argparse.SUPPRESS,
                 **kwargs):
        self.stop = stop
        kwargs.pop("nargs", None)
        super(_EarlyStopAction, self).__init__(
            option_strings, dest=dest, default=default, nargs=0, **kwargs)

    def __call__(self, parser: ArgumentParser, *args, **kwargs):
        parser.clear_actions()
        sys.stdout.write(self.stop)
        parser.exit(0)


class _OptSeqAction(argparse._StoreAction):
    """
    Custom action that split value by OPT_SEP and set new value as python list
    to namespace.
    """
    def __init__(self, option_strings, dest, sep=OPTSEP, **kwargs):
        self.sep = sep
        super(_OptSeqAction, self).__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, value, option_string=None):
        value = value.split(self.sep)
        super(_OptSeqAction, self).__call__(parser, namespace, value, option_string)


class _EvalValueAction(argparse._StoreAction):
    """
    Custom action that call ast.literal_eval() on value and stores.
    """
    def __call__(self, parser, namespace, value, option_string=None):
        try:
            ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass
        super(_EvalValueAction, self).__call__(parser, namespace, value, option_string)


class _KwargsAction(argparse._StoreAction):
    """
    Custom action that store remaining arguments as kwargs.
    """
    def __call__(self, parser, namespace, value, option_string=None):
        kwargs = {}
        it = iter(value)
        while True:
            try:
                k = next(it)
            except StopIteration:
                break

            k = k.split('=')
            if len(k) == 2:
                k, v = k
            else:
                k = k[0]
                try:
                    v = next(it)
                except StopIteration:
                    msg = f"{k} has no value"
                    raise RuntimeError(msg)
                if '=' in v:
                    msg = f"{k} has no value"
                    raise RuntimeError(msg)
            kwargs[k] = v
        super(_KwargsAction, self).__call__(parser, namespace, kwargs, option_string=None)


def get_cli_parser() -> ArgumentParser:
    mainParser = ArgumentParser(add_help=False, epilog=TASK.__doc__)
    mainParser.add_argument("-v", "--version", action="version", version=f"DSCollection {v}")

    # Parent parsers
    commParser = build_common_arguments()
    outTypeParser = build_output_dtypes_arguments()

    # Subparsers for each task
    tasks = mainParser.add_subparsers(title="DSCollection commands", dest="task", required=True)

    # Add generate task parser
    generateParser = tasks.add_parser(TASK.GENERATE, parents=[commParser, outTypeParser])
    add_generate_task_arguments(generateParser)

    # Add extract task parser
    extractParser = tasks.add_parser(TASK.EXTRACT, parents=[commParser, outTypeParser])
    add_extract_task_arguments(extractParser)

    # Add visualize task parser
    visualizeParser = tasks.add_parser(TASK.VISUALIZE, parents=[commParser])
    add_visualize_task_arguments(visualizeParser)

    # Add split task parser
    splitParser = tasks.add_parser(TASK.SPLIT,  parents=[commParser])
    add_split_task_argument(splitParser)

    # Add combine task parser
    combineParser = tasks.add_parser(TASK.COMBINE,  parents=[commParser])
    add_combine_task_argument(combineParser)

    return mainParser


def build_common_arguments(parser: ArgumentParser = None):
    if parser is None:
        parser = ArgumentParser(add_help=False)
    parser.add_argument("-i", "--input", nargs="+", required=True, help="Input video or directory.")
    parser.add_argument("-o", "--output", required="visualize" not in sys.argv,  help="Output directory.")
    parser.add_argument("-c", "--contiguous", action="store_true", help="Rename files sequentially.")
    return parser


def add_process_arguments(parser: ArgumentParser):
    parser.add_argument("--crop-size", type=int, default=1296, help="Center crop size (default: 1296).")
    parser.add_argument("--roi-offset", type=int, default=0, help="Offset of roi cropping (default: 0).")
    parser.add_argument("--camera-shifts", default="(0,0,0,0)", action=_EvalValueAction,
                        help="Offsets of camera cropping (default (0,0,0,0)).")
    parser.add_argument("--grayscale", action="store_true", help="Convert to grayscale image")
    return parser


def build_output_dtypes_arguments(parser: ArgumentParser = None):
    if parser is None:
        parser = ArgumentParser(add_help=False)
    gp = parser.add_mutually_exclusive_group()
    gp.add_argument('--voc', dest="dtype", action="store_const", const=DatasetType.VOC, help="Save as VOC.")
    gp.add_argument('--kitti', dest="dtype", action="store_const", const=DatasetType.KITTI, help="Save as KITTI (default).")
    gp.add_argument('--coco', dest="dtype", action="store_const", const=DatasetType.COCO, help="Save as COCO.")
    parser.set_defaults(dtype=DatasetType.KITTI)
    return parser


def add_generate_task_arguments(parser: ArgumentParser):
    parser.add_argument("-s", "--show", action="store_true", help="Show pipeline.")
    parser.add_argument("-m", "--model", action="append", default=[], help="Model config file.")
    parser.add_argument("--ext", default=".png", help="Image file extension.")


def add_extract_task_arguments(parser: ArgumentParser):
    parser.add_argument("-cls", "--classes", action=_OptSeqAction, help=f"Classes to extract, separate by '{OPTSEP}'.")
    gp = parser.add_mutually_exclusive_group()
    gp.add_argument("-n", "--num-images", dest="n_imgs", type=int, metavar="N",
                    help="Randomly extract N images from each input dataset.")
    gp.add_argument("-f", "--fraction", dest="n_imgs", type=float,
                    help="Extract percentage of samples, must be a float between (0, 1.0).")

    # Optional argument for some datasets
    parser.add_argument("--split", help="Optional keyword argument of `Dataset`, eg. split=train.")
    parser.add_argument("--year", help="Optional keyword argument of `Dataset`, eg. year=2017.")
    parser.add_argument("--kwargs", nargs='*', action=_KwargsAction,
                        help="Other keyword arguments used by Dataset constructor.")


def add_visualize_task_arguments(parser: ArgumentParser):
    parser.add_argument("-l", "--labels", action="store_true", help="Show label names.")
    parser.add_argument("-n", "--num-images", type=int, default=10, help="Number images to show.")
    parser.add_argument("--class-names", action=_OptSeqAction, help=f"A set of class names seperated by '{OPTSEP}'.")
    parser.add_argument("--backend", type=int, default=0, help="0: OpenCV (default); 1: Matplotlib.")
    parser.add_argument("--cmap", default="hsv", help="Colormap Name. Use --show-color-map show all names.")
    parser.add_argument("--show-all-cmaps", action=_EarlyStopAction, stop=get_colormap_info(),
                        help="Show all color-map names then exit.")
    parser.add_argument("--rows", type=int, help="Number rows in an image (Matplotlib only).")
    parser.add_argument("--cols", type=int, help="Number cols in an image (Matplotlib only).")
    parser.add_argument("--win-name", help="Window name (OpenCV only).")


def add_split_task_argument(parser: ArgumentParser):
    parser.add_argument("--drop", action="store_true", help="Keep original dataset.")
    parser.add_argument("--names", action=_OptSeqAction,
                        help=f"A set of filename suffix, use to rename each partition; separated by '{OPTSEP}'.")
    gp = parser.add_mutually_exclusive_group(required=True)
    gp.add_argument("-ns", "--num-splits", type=int, help="Number partitions to equally split.")
    gp.add_argument("-np", "--num-per-each", type=int, help="Maximum number of samples in each part.")
    gp.add_argument("-nf", "--num-fractions", action=_OptSeqAction,
                    help=f"Percentage (float number) per each part; separated by '{OPTSEP}', sums up to 1.0.")


def add_combine_task_argument(parser: ArgumentParser):
    parser.add_argument("--drop", action="store_true", help="Keep original dataset.")

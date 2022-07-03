import sys

from ..core.reorganize import DatasetSplitter


def main(args):
    s = DatasetSplitter(args.num_splits, args.num_per_each, args.num_fractions, args.names)
    for input_dir in args.input:
        try:
            s.run(input_dir, args.output, contiguous=args.contiguous, keep=args.keep)
        except Exception as e:
            sys.stderr.write("error: %s\n" % e)
            raise e
import sys

from .._DSCollection.reorganize import DatasetCombiner


def main(args):
    c = DatasetCombiner(args.input)
    try:
        if args.output is None:
            msg = "Output directory is None"
            raise ValueError(msg)
        c.run(args.output, args.contiguous, args.keep)
    except Exception as e:
        sys.stderr.write("error: %s\n" % e)
        raise e

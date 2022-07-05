from DSCollection.utils.tasks import TaskDispatcher, TASK
from DSCollection.core import DatasetSplitter


@TaskDispatcher(TASK.SPLIT)
def main(args):
    s = DatasetSplitter(args.num_splits, args.num_per_each, args.num_fractions, args.names)
    for input_dir in args.input:
        s.run(input_dir, args.output, contiguous=args.contiguous, keep=args.drop)

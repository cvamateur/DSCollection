from ..utils.tasks import TaskDispatcher, TASK
from ..core.reorganize import DatasetSplitter


@TaskDispatcher(TASK.SPLIT)
def main(args):
    s = DatasetSplitter(args.num_splits, args.num_per_each, args.num_fractions, args.names)
    for input_dir in args.input:
        s.run(input_dir, args.output, contiguous=args.contiguous, drop=args.drop)

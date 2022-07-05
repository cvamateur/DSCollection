from DSCollection.utils.tasks import TaskDispatcher, TASK
from DSCollection.core import DatasetCombiner


@TaskDispatcher(TASK.COMBINE)
def main(args):
    if args.output is None:
        msg = "Output directory is None"
        raise ValueError(msg)
    c = DatasetCombiner(args.input)
    c.run(args.output, args.contiguous, args.drop)

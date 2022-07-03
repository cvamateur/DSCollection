from ..utils.tasks import TaskDispatcher, TASK
from ..core.reorganize import DatasetCombiner


@TaskDispatcher(TASK.COMBINE)
def main(args):
    if args.output is None:
        msg = "Output directory is None"
        raise ValueError(msg)
    c = DatasetCombiner(args.input)
    c.run(args.output, args.contiguous, args.keep)

from ..utils.tasks import TaskDispatcher, TASK
from ..core.extractor import DataExtractor


@TaskDispatcher(TASK.EXTRACT)
def main(args):

    if not args.input:
        msg = "Empty input.  use -i/--input <path>"
        raise RuntimeError(msg)
    if not args.output:
        msg = "Output "

    print(args)


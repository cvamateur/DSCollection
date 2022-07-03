from ..utils.tasks import TaskDispatcher, TASK
from ..core.extractor import DataExtractor


@TaskDispatcher(TASK.EXTRACT)
def main_extract(args):
    print(args)


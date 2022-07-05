from ..utils.tasks import TaskDispatcher, TASK


@TaskDispatcher(TASK.GENERATE)
def main(args):
    print(args)
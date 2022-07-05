from DSCollection.utils.tasks import TaskDispatcher, TASK


@TaskDispatcher(TASK.GENERATE)
def main(args):
    print(args)
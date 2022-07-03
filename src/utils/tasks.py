from typing import Callable


class TASK:
    # Extract classes from dataset
    EXTRACT = "extract"

    # Generate new dataset from raw fisheye video
    GENERATE = "generate"

    # Visualize and annotating images from dataset
    VISUALIZE = "visualize"

    # Merge multiple dataset into one
    COMBINE = "combine"

    # Split one dataset into multiple parts
    SPLIT = "split"


class TaskDispatcher:

    _REG_TASK_FN = {}

    def __init__(self, task: str):
        self.task = task

    def __call__(self, fn: Callable):
        self._REG_TASK_FN[self.task] = fn
        return fn

    @classmethod
    def get_task_handler(cls, task: str):
        if task not in cls._REG_TASK_FN:
            msg = f"error: unrecognized task: `{task}`"
            raise RuntimeError(msg)
        return cls._REG_TASK_FN[task]
from typing import Callable, List


class TASK:
    """
    1. extract:
        Extract some or all classes from original datasets, save as VOC | KITTI.

    2. generate:
        Generate new dataset from raw fisheye videos, automated labeling by given model
        and save as VOC | KITTI.

    3. visualize:
        Visualize dataset using OpenCV or Matplotlib.

    4. combine:
        Combine multiple dataset into a whole new dataset, all input datasets must be
        of same type. Only images and labels are moved to new dataset.

    5. split:
        Split a dataset into multiple partitions, only images and labels are moved to
        partitions.

    6. process:

    """
    EXTRACT = "extract"
    GENERATE = "generate"
    VISUALIZE = "visualize"
    COMBINE = "combine"
    SPLIT = "split"
    PROCESS = "process"

    @classmethod
    def list_all(cls) -> List[str]:
        return [cls.EXTRACT, cls.GENERATE, cls.VISUALIZE, cls.COMBINE, cls.SPLIT, cls.PROCESS]


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

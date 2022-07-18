# @Time    : 2022/7/14 下午5:48
# @Author  : Boyang
# @Site    : 
# @File    : augmentation.py
# @Software: PyCharm
from ..core.augmentation import Augmentation, Invoker, Rotate, Save
from ..core.dataset import Dataset, KITTI
from ..utils.common import check_path
from ..utils.tasks import TaskDispatcher, TASK


@TaskDispatcher(task=TASK.AUGMENTATION)
def main(args):
    aug = Augmentation(args.output, args.name, args.dtype, args.contiguous)
    invoker = Invoker()

    if args.rotate != 0:
        cmd = Rotate(aug, args.rotate, args.divide_number)
        invoker.command(cmd)

    if args.output:
        cmd = Save(aug)
        invoker.command(cmd)

    input_datasets = []
    for path in args.input:
        path = check_path(path)
        data_class = Dataset.find_dataset(path)
        if data_class is None:
            dataset = KITTI(path, imgDir="default/image_2", lblDir="default/label_2")
        else:
            dataset = data_class(path)
        input_datasets.append(dataset)

    for ds in input_datasets:
        invoker.execute(ds)

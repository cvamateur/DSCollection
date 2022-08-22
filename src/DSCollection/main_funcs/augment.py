# @Time    : 2022/7/14 下午5:48
# @Author  : Boyang
# @Site    : 
# @File    : augmentation.py
# @Software: PyCharm
import glob
import shutil
from os.path import join as path_join
from typing import List

from ..core.augment import (Augmentation, Invoker, Rotate, Save, CenterCrop, Blur, CoarseDropout, DownScale)
from ..core.dataset import Dataset, KITTI
from ..utils.common import check_path
from ..utils.tasks import TaskDispatcher, TASK


def _special_augment(cmd, part: str, dtype: str, contiguous: bool, input_datasets: List[Dataset]) -> Dataset:
    invoker = Invoker()
    invoker.add_command(cmd)
    aug = Augmentation(part, dtype, contiguous)
    cmd = Save(aug)
    invoker.add_command(cmd)
    for ds in input_datasets:
        invoker.execute(ds)

    data_class = Dataset.find_dataset(part)
    nds = data_class(part)
    if len(input_datasets) > 1:
        input_datasets[-1] = nds
    else:
        input_datasets.append(nds)
    return nds


def _merge_and_clean(aug: Augmentation, ds_list: List[Dataset]):
    invoker = Invoker()
    cmd = Save(aug)
    invoker.add_command(cmd)
    for ds in ds_list:
        invoker.execute(ds)
    for dirname in glob.glob(path_join(aug.output, "part*")):
        part = path_join(aug.output, dirname)
        shutil.rmtree(part)


@TaskDispatcher(task=TASK.AUGMENT)
def main(args):
    input_datasets = []
    for path in args.input:
        path = check_path(path)
        data_class = Dataset.find_dataset(path)
        if data_class is None:
            dataset = KITTI(path, imgDir="default/image_2", lblDir="default/label_2")
        else:
            dataset = data_class(path)
        input_datasets.append(dataset)

    aug = Augmentation(args.output, args.dtype, args.contiguous)
    invoker = Invoker()

    i = 0
    all_ds = input_datasets.copy()
    if args.rotate:
        cmd = Rotate(aug, args.degree, args.divide_number)
        if args.special_mode:
            part = path_join(aug.output, "part{}".format(i))
            nds = _special_augment(cmd, part, args.dtype, args.contiguous, input_datasets)
            all_ds.append(nds)
            i += 1
        else:
            invoker.add_command(cmd)

    if args.blur:
        cmd = Blur(
            blur_limit=args.blur_limit,
            p=args.probability,
        )
        if args.special_mode:
            part = path_join(aug.output, "part{}".format(i))
            nds = _special_augment(cmd, part, args.dtype, args.contiguous, input_datasets)
            all_ds.append(nds)
            i += 1
        else:
            invoker.add_command(cmd)

    if args.center_crop:
        cmd = CenterCrop(
            width=args.center_crop_width,
            height=args.center_crop_height,
            min_area=args.min_area,
            min_visibility=args.min_visibility,
            p=args.probability
        )
        if args.special_mode:
            part = path_join(aug.output, "part{}".format(i))
            nds = _special_augment(cmd, part, args.dtype, args.contiguous, input_datasets)
            all_ds.append(nds)
            i += 1
        else:
            invoker.add_command(cmd)

    if args.coarse_dropout:
        cmd = CoarseDropout(
            min_area=args.min_area,
            min_visibility=args.min_visibility,
            max_holes=args.max_holes,
            max_width=args.max_width,
            max_height=args.max_height,
            min_holes=args.min_holes,
            min_width=args.min_width,
            min_height=args.min_height,
            fill_value=args.fill_value,
            mask_fill_value=args.mask_fill_value,
            p=args.probability
        )
        if args.special_mode:
            part = path_join(aug.output, "part{}".format(i))
            nds = _special_augment(cmd, part, args.dtype, args.contiguous, input_datasets)
            all_ds.append(nds)
            i += 1
        else:
            invoker.add_command(cmd)
    if args.down_scale:
        cmd = DownScale(
            min_area=args.min_area,
            min_visibility=args.min_visibility,
            scale_min=args.scale_min,
            scale_max=args.scale_max,
            interpolation=args.interpolation,
            p=args.probability
        )
        if args.special_mode:
            part = path_join(aug.output, "part{}".format(i))
            nds = _special_augment(cmd, part, args.dtype, args.contiguous, input_datasets)
            all_ds.append(nds)
            i += 1
        else:
            invoker.add_command(cmd)

    if args.special_mode:
        _merge_and_clean(aug, all_ds)
    else:
        cmd = Save(aug)
        invoker.add_command(cmd)
        for ds in input_datasets:
            invoker.execute(ds)

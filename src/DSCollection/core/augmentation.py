# @Time    : 2022/7/14 下午5:48
# @Author  : Boyang
# @Site    : 
# @File    : augmentation.py
# @Software: PyCharm
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from concurrent.futures.process import ProcessPoolExecutor
from os.path import join as path_join
from typing import List, Callable

import numpy as np

from .common import chunk_dataset
from ..core.convertor import Convertor, LabelInfo
from ..core.dataset import Dataset, KITTI, ImageLabel
from ..utils.imgutil import ImageUtil

MAX_WORKER = 5


class Invoker:
    def __init__(self):
        self.commands: List[Command] = []

    def command(self, c: Command):
        self.commands.append(c)

    def execute(self, ds: Dataset):
        for img_data, labels in chunk_dataset(ds):
            for c in self.commands:
                c.execute(img_data, labels)


class Command(ABC):
    """Abstract Command"""

    @abstractmethod
    def execute(self, *args, **kwargs):
        raise NotImplementedError


class Rotate(Command):
    """Concrete Command"""

    def __init__(self, receiver: Augmentation):
        self.receiver = receiver

    def execute(self, img_data: List[np.ndarray], labels: List[ImageLabel]):
        self.receiver.rotate(img_data, img_data)


class Save(Command):
    def __init__(self, receiver: Augmentation):
        self.receiver = receiver

    def execute(self, img_data: List[np.ndarray], labels: List[ImageLabel]):
        self.receiver.save(img_data, labels)


class Augmentation:
    """Receiver"""

    def __init__(self, output: str, name: str, dst_dtype: str, contiguous: bool):
        self.output = path_join(output, name)
        assert not os.path.exists(self.output), "Output dataset already exist."

        self.output_dst = Dataset.from_type(dst_dtype)(output)
        self.output_dst.create_structure(output, name)
        self.convertor = Convertor.from_type(dst_dtype)
        self.contiguous = contiguous

        self._count = 0
        self.pro_executor = ProcessPoolExecutor(max_workers=MAX_WORKER)

    def rotate(self, img_data: List[np.ndarray], labels: List[ImageLabel]):
        ...

    @staticmethod
    def _save(cvt: Convertor, img_path: str, lbl_path: str, img: np.ndarray, label: ImageLabel):
        label_info = cvt.convert_label(1, label)
        bimg = ImageUtil.encode(img)
        with open(img_path, 'wb') as f:
            f.write(bimg)
        with open(lbl_path, 'wb') as f:
            f.write(label_info.data)

    def save(self, img_data: List[np.ndarray], labels: List[ImageLabel]):
        futures = []
        for img, lbl in zip(img_data, labels):
            if self.contiguous:
                img_name = "{:06}{}".format(self._count, ".png")
                lab_name = "{:06}{}".format(self._count, ".txt")
            else:
                img_name = "{:06}{}".format(self._count, ".png")
                lab_name = "{:06}{}".format(self._count, ".txt")

            img_path = path_join(self.output, self.output_dst.imgDirName, img_name)
            lbl_path = path_join(self.output, self.output_dst.lblDirName, lab_name)

            futures.append(self.pro_executor.submit(self._save, self.convertor, img_path, lbl_path, img, lbl))
            self._count += 1

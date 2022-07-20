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
from typing import List, Tuple

import numpy as np

from .common import chunk_dataset
from ..core.convertor import Convertor
from ..core.dataset import Dataset, ImageLabel, Box
from ..core.rotate import rotated_calculate_divide_point_bbox, rotate_image
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
                img_data, labels = c.execute(img_data, labels)


class Command(ABC):
    """Abstract Command"""

    @abstractmethod
    def execute(self, *args, **kwargs) -> Tuple[List[np.ndarray], List[ImageLabel]]:
        raise NotImplementedError


class Rotate(Command):
    """Concrete Command"""

    def __init__(self, receiver: Augmentation, degree: int, divide_num: int):
        self.receiver = receiver
        self.degree = degree
        self.ndivd = divide_num

    def execute(self, img_data: List[np.ndarray], labels: List[ImageLabel]):
        img_data, labels = self.receiver.rotate(self.degree, self.ndivd, img_data, labels)
        return img_data, labels


class Save(Command):
    def __init__(self, receiver: Augmentation):
        self.receiver = receiver

    def execute(self, img_data: List[np.ndarray], labels: List[ImageLabel]):
        self.receiver.save(img_data, labels)
        return img_data, labels


class Augmentation:
    """Receiver"""

    def __init__(self, output: str, dst_dtype: str, contiguous: bool):
        self.output = output
        assert not os.path.exists(self.output), "Output dataset already exist."

        self.output_dst = Dataset.from_type(dst_dtype)(output)
        self.output_dst.create_structure(*os.path.split(self.output))
        self.convertor = Convertor.from_type(dst_dtype)
        self.contiguous = contiguous

        self._count = 0
        self.pro_executor = ProcessPoolExecutor(max_workers=MAX_WORKER)

    @classmethod
    def rotate(cls, degree: int, number_divide: int, img_data: List[np.ndarray], labels: List[ImageLabel]):
        res_img = []
        for image, lab in zip(img_data, labels):
            h, w = image.shape[:2]
            rta_img = rotate_image(image, degree=degree)
            boxes = []
            for obj in lab.boxes:
                minx, miny, maxx, maxy = rotated_calculate_divide_point_bbox((obj.left, obj.top, obj.right, obj.bottom),
                                                                             degree, h, w, number_divide)
                boxes.append(Box(minx, miny, maxx, maxy, obj.name))
            lab.boxes = boxes
            res_img.append(rta_img)
        return res_img, labels

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

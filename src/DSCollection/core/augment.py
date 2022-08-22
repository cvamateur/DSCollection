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
from typing import List, Optional, Callable

import albumentations as A
import cv2
import numpy as np
from albumentations.core.bbox_utils import BboxProcessor

from .common import chunk_dataset
from ..core.convertor import Convertor
from ..core.dataset import Dataset, ImageLabel, Box
from ..core.rotate import rotated_calculate_divide_point_bbox, rotate_image
from ..utils.imgutil import ImageUtil

MAX_WORKER = 5


class Invoker:
    def __init__(self):
        self.commands: List[Command] = []

    def add_command(self, c: Command):
        self.commands.append(c)

    def execute(self, ds: Dataset):
        for img_data, labels in chunk_dataset(ds):
            for c in self.commands:
                img_data, labels = c.execute(img_data, labels)


class Command(ABC):
    """Abstract Command"""

    @abstractmethod
    def execute(self, img_data: List[np.ndarray], labels: List[ImageLabel]):
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


class PixelAugmentCommand(Command):

    def __init__(self, aug_func: Callable, min_area: float, min_visibility: float):
        self.aug_func = aug_func
        self.min_area = min_area
        self.min_visibility = min_visibility

    def execute(self, img_data: List[np.ndarray], labels: List[ImageLabel]):
        res_img = []
        for img in img_data:
            data = self.aug_func(image=img)
            res_img.append(data["image"])

        return res_img, labels


class SpatialAugmentCommand(Command):

    def __init__(self, aug_func: Callable, min_area: float, min_visibility: float):
        self.aug_func = aug_func
        self.min_area = min_area
        self.min_visibility = min_visibility

    def execute(self, img_data: List[np.ndarray], labels: List[ImageLabel]):
        process = BboxProcessor(
            A.BboxParams('pascal_voc', min_area=self.min_area, min_visibility=self.min_visibility))
        res_lbl, res_img = [], []
        for img, lbl in zip(img_data, labels):
            data = {"image": img, "bboxes": [[*b.coord, b.name] for b in lbl.boxes]}
            process.preprocess(data)
            data = self.aug_func(**data)
            process.postprocess(data)

            res_img.append(data["image"])
            boxes = [Box(*b) for b in data["bboxes"]]
            lbl.boxes = boxes
            res_lbl.append(lbl)

        return res_img, res_lbl


class Blur(PixelAugmentCommand):
    def __init__(self, blur_limit: int = 7, min_area: float = 0.0, min_visibility: float = 0.0, p: float = 1.0):
        aug_func = A.Blur(blur_limit, p=p)
        super(Blur, self).__init__(aug_func, min_area=min_area, min_visibility=min_visibility)


class CenterCrop(SpatialAugmentCommand):
    def __init__(self, width: int, height: int, min_area=0.0, min_visibility=0.0, p: float = 1.0):
        center_crop = A.CenterCrop(height, width, p=p)
        super(CenterCrop, self).__init__(center_crop, min_area, min_visibility)


class CoarseDropout(PixelAugmentCommand):
    def __init__(self,
                 min_area: float = 0.0,
                 min_visibility: float = 0.0,
                 max_holes: int = 8,
                 max_height: int = 8,
                 max_width: int = 8,
                 min_holes: Optional[int] = None,
                 min_height: Optional[int] = None,
                 min_width: Optional[int] = None,
                 fill_value: int = 0,
                 mask_fill_value: Optional[int] = None,
                 always_apply: bool = False,
                 p: float = 0.5
                 ):
        aug_func = A.CoarseDropout(max_holes, max_height, max_width, min_holes, min_height, min_width,
                                   fill_value, mask_fill_value, always_apply, p=p)

        super(CoarseDropout, self).__init__(aug_func, min_area, min_visibility)


class DownScale(PixelAugmentCommand):
    def __init__(self,
                 min_area: float = 0.0,
                 min_visibility: float = 0.0,
                 scale_min=0.25,
                 scale_max=0.25,
                 interpolation=cv2.INTER_NEAREST,
                 p: float = 0.5
                 ):
        aug_func = A.Downscale(scale_min=scale_min, scale_max=scale_max, interpolation=interpolation, p=p)
        super(DownScale, self).__init__(aug_func, min_area, min_visibility)


# class TestCommand(Command):
#     def __init__(self):
#         center_crop = A.CenterCrop(400, 400)
#         coarse_dropout = A.CoarseDropout()
#
#         self.compose = A.Compose([center_crop, coarse_dropout],
#                                  bbox_params=A.BboxParams('pascal_voc'))
#
#     def execute(self, img_data: List[np.ndarray], labels: List[ImageLabel]):
#         for img, lbl in zip(img_data, labels):
#             bboxes = [[*b.coord, b.name] for b in lbl.boxes]
#             self.compose(image=img, bboxes=bboxes)


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
        assert not os.path.exists(self.output), f"Directory `{self.output}` exist."

        self.convertor = Convertor.from_type(dst_dtype)
        self.convertor.create_dataset(*os.path.split(output))

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
                if maxx > minx and maxy > miny:
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

            img_path = path_join(self.output, self.convertor.imgDirName, img_name)
            lbl_path = path_join(self.output, self.convertor.lblDirName, lab_name)

            futures.append(self.pro_executor.submit(self._save, self.convertor, img_path, lbl_path, img, lbl))
            self._count += 1

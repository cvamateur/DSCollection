# @Time    : 2022/7/6 下午1:49
# @Author  : Boyang
# @Site    : 
# @File    : process.py
# @Software: PyCharm
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED, Future, ProcessPoolExecutor
from os.path import join as path_join
from typing import List, Callable, Tuple, Dict
import shutil

import cv2
import numpy as np
from tqdm import tqdm

from .convertor import Convertor
from ..core.dataset import Dataset, ImageLabel, KITTI, Box
from ..utils.common import check_path
from ..utils.imgutil import ImageUtil

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

CHUNK_SIZE = 10
MAX_WORKER = 5


class Process:

    def __init__(self, inputs: List[str], dst_dtype: str, output: str = None, name: str = "Merged",
                 force: bool = False):
        if output:
            self.output = path_join(output, name)
            if os.path.exists(self.output):
                if force:
                    shutil.rmtree(self.output)
                else:
                    raise FileExistsError("Output dataset already exist.")

            self.output_dst = Dataset.from_type(dst_dtype)(output)
            self.output_dst.create_structure(output, name)
            self.convertor = Convertor.from_type(dst_dtype)
        else:
            self.output_dst = None

        self.datasets = []
        for path in inputs:
            path = check_path(path)
            data_class = Dataset.find_dataset(path)
            if data_class is None:
                dataset = KITTI(path, imgDir="default/image_2", lblDir="default/label_2")
            else:
                dataset = data_class(path)

            self.datasets.append(dataset)

        self.executor = ThreadPoolExecutor(max_workers=MAX_WORKER)
        self.pro_executor = ProcessPoolExecutor(max_workers=MAX_WORKER)
        self.current_dataset_index = 0
        self.total_input = len(inputs)
        self._count = 0

    @classmethod
    def _read_image(cls, img_path: str):
        with open(img_path, "rb") as f:
            b = f.read()
            return ImageUtil.decode(b)

    def read_image(self, labels: List[ImageLabel], root: str, img_dir_name: str) -> Dict[Future, ImageLabel]:
        future = {}
        for lbl in labels:
            img_path = path_join(root, img_dir_name, lbl.fileName)
            future.update({self.executor.submit(self._read_image, img_path): lbl})

        return future

    @staticmethod
    def _chunkify(size: int, iter_data: List[ImageLabel]):
        res = []
        for lab in iter_data:
            res.append(lab)
            if len(res) == size:
                yield res
                res.clear()

        yield res

    @classmethod
    def _wait_completed(cls, fs):
        res = wait(fs, return_when=ALL_COMPLETED)
        if res.not_done:
            logger.warning("Exist failed task")

    def load(self) -> Tuple[Dict[ImageLabel, np.ndarray]]:
        for dataset in self.datasets:
            self.current_dataset_index += 1
            dataset.load(skip_empty=False)
            iter_labels = self._chunkify(CHUNK_SIZE, dataset.labels)
            labels = next(iter_labels)
            pre_data = self.read_image(labels, dataset.root, dataset.imgDirName)
            if len(pre_data) == 0:
                continue

            while True:
                self._wait_completed(pre_data)
                labels = next(iter_labels)
                next_data = self.read_image(labels, dataset.root, dataset.imgDirName)
                yield [f.result() for f in pre_data.keys()], [l for l in pre_data.values()]
                pre_data, next_data = next_data, None

                if len(pre_data) < CHUNK_SIZE:
                    self._wait_completed(pre_data)
                    yield [f.result() for f in pre_data.keys()], [l for l in pre_data.values()]
                    break

    @staticmethod
    def center_crop(w: int, h: int, crop_size: int):
        """
        Crop image in the center.

        :param w: Width of original image.
        :param h: Height of original image.
        :param crop_size: Width and height of final cropped image.
        :return: (x1, y1, x2, y2) Coordinate of the cropped image.
        """
        dy = max(0, (h - crop_size)) // 2
        dx = max(0, (w - crop_size)) // 2
        return dx, dy

    @staticmethod
    def roi_crop(crop_size: int, tw: int, th: int, offset: int):
        """
        Return coordinates of the largest ROI in `crop_size` region,
        whose aspect ratio is same as tw/th.

        NOTE:
            The coordinates are relative to center cropped region.

        :param crop_size: Size of center cropped region.
        :param tw: Width of a region whose aspect is same as the final ROI's
        :param th: Height of a region whose aspect is same as the final ROI's
        :param offset: A small offset
        :return:
        """
        if tw >= th:
            nh = int(th / tw * crop_size)
            delta = (crop_size - nh) // 2
            offset = max(min(delta, offset), -delta)
            x1 = 0
            y1 = delta + offset
            x2 = crop_size
            y2 = y1 + nh
        else:
            nw = int(tw / th * crop_size)
            delta = (crop_size - nw) // 2
            offset = max(min(delta, offset), -delta)
            x1 = delta + offset
            y1 = 0
            x2 = x1 + nw
            y2 = crop_size

        return x1, y1, x2, y2

    @staticmethod
    def show(img_array: np.ndarray):
        cv2.imshow("ImageShow", img_array)
        k = cv2.waitKey(0)
        if k == ord('q') or k == 27:
            sys.exit(-1)

    @staticmethod
    def annotate_image(img: np.ndarray, boxes: List[Box], color=(0, 255, 0), line_type=cv2.FILLED):
        for obj in boxes:
            pt1 = int(obj.left), int(obj.top)
            pt2 = int(obj.right), int(obj.bottom)

            img = cv2.rectangle(img, pt1, pt2, color, 2, lineType=line_type)
            img = cv2.putText(img, obj.name, (int(obj.left), int(obj.top - 2)), cv2.FONT_HERSHEY_PLAIN, 0.9,
                              (0, 255, 0), 1)

        return img

    @staticmethod
    def dotted_line(image, pt1, pt2, color, thickness, *_):
        def interpolate(pt1, pt2, gap=8):
            in_xs = list(range(pt1[0], pt2[0] + 1, gap))
            in_ys = list(range(pt1[1], pt2[1] + 1, gap))

            while len(in_xs) < len(in_ys):
                in_xs.append(in_xs[-1])

            while len(in_ys) < len(in_xs):
                in_ys.append(in_ys[-1])

            return [(x, y) for x, y in zip(in_xs, in_ys)]

        x0, x1 = min(pt1[0], pt2[0]), max(pt1[0], pt2[0])
        y0, y1 = min(pt1[1], pt2[1]), max(pt1[1], pt2[1])

        # left
        for pt in interpolate((x0, y0), (x0, y1)):
            cv2.circle(image, pt, 1, color, -thickness)

        # top
        for pt in interpolate((x0, y0), (x1, y0)):
            cv2.circle(image, pt, 1, color, -thickness)

        # right
        for pt in interpolate((x1, y0), (x1, y1)):
            cv2.circle(image, pt, 1, color, -thickness)

        # bottom
        for pt in interpolate((x0, y1), (x1, y1)):
            cv2.circle(image, pt, 1, color, -thickness)

    @staticmethod
    def process_label(lbl: ImageLabel,
                      roi: Tuple[int, int, int, int],
                      min_area: int,
                      min_ratio: float,
                      max_ratio: float,
                      verbose: bool = False,
                      scale: float = 1.0) -> Tuple[List[Box], List[Box]]:
        """
        Rectify bounding boxes in a label file given ROI and filters rules.
        """
        MINX, MAXX = 0, roi[2] - roi[0]
        MINY, MAXY = 0, roi[3] - roi[1]

        keeped_objs = []
        skipped_objs = []
        boxes = []
        skipped_boxes = []
        total = 0

        for box in lbl.boxes:
            # relative coord to roi
            xmin = min(max(MINX, box.left - roi[0]), MAXX) * scale
            ymin = min(max(MINY, box.top - roi[1]), MAXY) * scale
            xmax = min(max(MINX, box.right - roi[0]), MAXX) * scale
            ymax = min(max(MINY, box.bottom - roi[1]), MAXY) * scale
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)

            total += 1
            w = xmax - xmin
            h = ymax - ymin
            a = w * h
            r = w / (h + 1e-6)

            # filter objects
            if a >= min_area and min_ratio <= r <= max_ratio:
                keeped_objs.append((xmin, ymin, xmax, ymax, w, h, a, r, box.name))
                boxes.append(Box(xmin, ymin, xmax, ymax, box.name))
            else:
                skipped_objs.append((xmin, ymin, xmax, ymax, w, h, a, r, box.name))
                skipped_boxes.append(Box(xmin, ymin, xmax, ymax, box.name))

        if verbose:
            print(f"Number objects: {len(keeped_objs)} / {total}")
            for obj in keeped_objs:
                box = [f"{c:<4d}" for c in obj[:4]]
                print(f"  {' '.join(box)} | {obj[4]:>3d}x{obj[5]:<3d}={obj[6]:<5d} | {obj[7]:.2f}")

            if len(skipped_objs):
                print("Skipped objects:")
                for obj in skipped_objs:
                    box = [f"{c:<4d}" for c in obj[:4]]
                    print(f"  {' '.join(box)} | {obj[4]:>3d}x{obj[5]:<3d}={obj[6]:<5d} | {obj[7]:.2f}")
            print()
        return boxes, skipped_boxes

    @staticmethod
    def _save(cvt: Convertor, img_path: str, lbl_path: str, img: np.ndarray, label: ImageLabel, ow: int, oh: int):

        label_info = cvt.convert_label(1, label)
        img = cv2.resize(img, (ow, oh), interpolation=cv2.INTER_CUBIC)
        bimg = ImageUtil.encode(img)
        with open(img_path, 'wb') as f:
            f.write(bimg)
        with open(lbl_path, 'wb') as f:
            f.write(label_info.data)

    def save(self, img_data: List[np.ndarray], labels: List[ImageLabel], contiguous: bool, ow: int, oh: int):
        futures = []
        for img, lbl in zip(img_data, labels):
            if contiguous:
                img_name = "{:06}{}".format(self._count, ".png")
                lab_name = "{:06}{}".format(self._count, ".txt")
            else:
                img_name = "{:06}{}".format(self._count, ".png")
                lab_name = "{:06}{}".format(self._count, ".txt")

            img_path = path_join(self.output, self.output_dst.imgDirName, img_name)
            lbl_path = path_join(self.output, self.output_dst.lblDirName, lab_name)

            futures.append(self.pro_executor.submit(self._save, self.convertor, img_path, lbl_path, img, lbl, ow, oh))
            self._count += 1

    def run(self, args):
        width, height = map(int, args.output_size)
        crop_size = args.crop_size
        offset = args.roi_offset

        process_bar = tqdm(desc="Process")
        process_bar.total = self.total_input
        crt_process = 0

        for img_data, labels in self.load():
            batch_data = np.asarray(img_data)
            h, w = batch_data.shape[1:-1]
            dx, dy = self.center_crop(w, h, crop_size)
            x1, y1, x2, y2 = self.roi_crop(crop_size, width, height, offset)
            batch_data = batch_data[:, y1 + dy:y2 + dy, x1 + dx:x2 + dx, :]

            roi = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)
            processed_labels = []
            for i, lbl in enumerate(labels):
                keeps, skips = self.process_label(lbl, roi,
                                                  min_area=args.min_area,
                                                  max_ratio=args.max_ratio,
                                                  min_ratio=args.min_ratio,
                                                  scale=height / (roi[3] - roi[1]))
                lbl.boxes = keeps
                if args.show:
                    img = batch_data[i, ...].copy()
                    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
                    img = self.annotate_image(img, keeps)

                    for sk_lbl in skips:
                        pt1 = int(sk_lbl.left), int(sk_lbl.top)
                        pt2 = int(sk_lbl.right), int(sk_lbl.bottom)
                        self.dotted_line(img, pt1, pt2, (0, 200, 0), 2)
                    self.show(img)

                processed_labels.append(lbl)

            if self.output_dst is not None:
                img_data = list(batch_data[:, ])
                self.save(img_data, processed_labels, args.contiguous, width, height)

            if self.current_dataset_index - crt_process == 1:
                process_bar.update()


class Compose:
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, img: np.ndarray):
        for t in self.transforms:
            img = t(img)

        return img
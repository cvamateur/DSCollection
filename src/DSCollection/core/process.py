# @Time    : 2022/7/6 下午1:49
# @Author  : Boyang
# @Site    : 
# @File    : process.py
# @Software: PyCharm
import logging
import math
import os
import shutil
import sys
from math import ceil
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED, Future
from os.path import join as path_join
from typing import List, Tuple, Dict

import cv2
import numpy as np
from tqdm import tqdm

from .convertor import Convertor, LabelInfo
from ..core.dataset import Dataset, ImageLabel, KITTI, Box
from ..utils.common import check_path
from ..utils.imgutil import ImageUtil

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

CHUNK_SIZE = 1
MAX_WORKER = os.cpu_count() - 1


class Process:

    def __init__(self, inputs: List[str], dst_dtype: str, output: str, force: bool = False, img_ext: str = ".jpg",
                 class_map: Dict[str, str] = None):
        self.cls_map = class_map if class_map else {}
        self.img_ext = img_ext if img_ext.startswith(".") else f".{img_ext}"
        self.output = output
        if os.path.exists(self.output):
            if force:
                shutil.rmtree(self.output)
            else:
                raise FileExistsError("Output dataset already exist.")

        self.convertor = Convertor.from_type(dst_dtype)
        self.convertor.create_dataset(*os.path.split(output))

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
        if res:
            yield res

    @classmethod
    def _wait_completed(cls, fs):
        res = wait(fs, return_when=ALL_COMPLETED)
        if res.not_done:
            logger.warning("Exist failed task")

    def load(self) -> Tuple[List[np.ndarray], List[ImageLabel]]:
        process_bar = tqdm(total=0, desc="Process")
        for dataset in self.datasets:
            dataset.load(skip_empty=False)
            process_bar.total += len(dataset)
            # convert_result = self.convertor.convert(dataset)
            iter_labels = self._chunkify(CHUNK_SIZE, dataset.labels)
            try:
                labels = next(iter_labels)
                process_bar.update(len(labels))
            except StopIteration:
                continue
            pre_data = self.read_image(labels, dataset.root, dataset.imgDirName)

            while True:
                self._wait_completed(pre_data)
                try:
                    labels = next(iter_labels)
                    process_bar.update(len(labels))
                except StopIteration:
                    yield [f.result() for f in pre_data.keys()], [l for l in pre_data.values()]
                    break
                next_data = self.read_image(labels, dataset.root, dataset.imgDirName)
                yield [f.result() for f in pre_data.keys()], [l for l in pre_data.values()]
                pre_data, next_data = next_data, None

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
    def _center_crop(img_size, tw: int, th: int):
        """
        Return coordinates of the largest ROI in `crop_size` region,
        whose aspect ratio is same as tw/th.

        NOTE:
            The coordinates are relative to center cropped region.

        :param img_size: Size of image size.
        :param tw: Width of a region whose aspect is same as the final ROI's
        :param th: Height of a region whose aspect is same as the final ROI's
        :return:
        """
        ih, iw = img_size
        scale_h, scale_w = th / ih, tw / iw
        if scale_h > scale_w:
            nw = int(tw / scale_h)
            delta = (iw - nw) // 2
            x1 = delta
            y1 = 0
            x2 = delta + nw
            y2 = ih
        else:
            nh = int(th / scale_w)
            delta = (ih - nh) // 2
            x1 = 0
            y1 = delta
            x2 = iw
            y2 = delta + nh

        return x1, y1, x2, y2

    @staticmethod
    def _percent_crop2(img_size, tw: int, th: int, crop_ratio: float):
        ih, iw = img_size
        a = tw / th
        crop_w = math.sqrt(ih * iw * crop_ratio * a)
        crop_h = crop_w / a
        cx, cy = iw / 2, ih / 2
        dw, dh = crop_w // 2, crop_h // 2
        x1, y1, x2, y2 = int(cx - dw), int(cy - dh), int(cx + dw), int(cy + dh)
        return x1, y1, x2, y2

    @staticmethod
    def _tri_cut(cx1, cy1, cx2, cy2, iw: int, ih: int):
        tw, th = cx2 - cx1, cy2 - cy1
        s = set()
        s.add((cx1, cy1, cx2, cy2))
        if tw < iw:
            s.add((0, 0, tw, th))
            s.add((iw - tw, 0, iw, ih))
        if th < ih:
            s.add((0, 0, tw, th))
            s.add((0, ih - th, iw, ih))

        return s

    @staticmethod
    def _five_cut(cx1, cy1, cx2, cy2, iw: int, ih: int):
        tw, th = cx2 - cx1, cy2 - cy1
        s = set()
        s.add((0, 0, tw, th))
        s.add((iw - tw, 0, iw, th))
        s.add((cx1, cy1, cx2, cy2))
        s.add((0, ih - th, tw, ih))
        s.add((iw - tw, ih - th, iw, ih))

        return s

    @staticmethod
    def _cut_many(img_size: Tuple[int, int], tw: int, th: int):
        ih, iw = img_size
        nh, nw = ceil(ih / th), ceil(iw / tw)
        gh = 0 if nh == 1 else (ih - th) // (nh - 1)
        gw = 0 if nw == 1 else (iw - tw) // (nw - 1)
        s = set()
        for i in range(nh):
            for j in range(nw):
                sy, sx = i * gh, j * gw
                s.add((sx, sy, sx + tw, sy + th))
        return s

    @classmethod
    def cal_roi(cls, img_size: Tuple[int, int], tw: int, th: int, crop_mode: int, crop_ratio: float):
        ih, iw = img_size
        cx1, cy1, cx2, cy2 = cls._center_crop(img_size, tw, th)
        if crop_mode == 1:
            return [(cx1, cy1, cx2, cy2)]
        elif crop_mode == 3:
            return cls._tri_cut(cx1, cy1, cx2, cy2, iw, ih)
        elif crop_mode == 5:
            cx1, cy1, cx2, cy2 = cls._percent_crop2(img_size, tw, th, crop_ratio)
            return cls._five_cut(cx1, cy1, cx2, cy2, iw, ih)
        elif crop_mode == 0:
            return cls._cut_many(img_size, tw, th)
        else:
            raise NotImplementedError("crop-mode: %d", crop_mode)

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

    def process_label(self, lbl: ImageLabel,
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
            box_name = self.cls_map.get(box.name, box.name)
            if a >= min_area and min_ratio <= r <= max_ratio:
                keeped_objs.append((xmin, ymin, xmax, ymax, w, h, a, r, box_name))
                boxes.append(Box(xmin, ymin, xmax, ymax, box_name))
            else:
                skipped_objs.append((xmin, ymin, xmax, ymax, w, h, a, r, box_name))
                skipped_boxes.append(Box(xmin, ymin, xmax, ymax, box_name))

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
    def _save(img_path: str, lbl_path: str, img: np.ndarray, label_info: LabelInfo, width: int, height: int, ext: str):
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
        bimg = ImageUtil.encode(img, ext)
        with open(img_path, 'wb') as f:
            f.write(bimg)
        with open(lbl_path, 'wb') as f:
            f.write(label_info.data)

    def save(self, img_data: List[np.ndarray], labels: List[ImageLabel], local_idx: List[int],
             contiguous: bool, width: int, height: int, keep_emp_lbl: bool):
        futures = []
        d = 8
        if self._count > 10 ** d:
            d += 1
        filename_format = "{:0%d}{}" % d
        for img, lbl, roiIdx in zip(img_data, labels, local_idx):
            lbl_info = self.convertor.convert_label(self._count, lbl)
            if contiguous:
                img_name = filename_format.format(self._count, self.img_ext)
                lab_name = filename_format.format(self._count, self.convertor.lblExt)
            else:
                filename, ext = os.path.splitext(lbl_info.imgName.replace('/', '-'))
                img_name = f"{filename}-{roiIdx}-{self.img_ext}"
                lab_name = f"{filename}-{roiIdx}-{self.convertor.lblExt}"
            if len(lbl.boxes) == 0 and not keep_emp_lbl:
                continue

            img_path = path_join(self.output, self.convertor.imgDirName, img_name)
            lbl_path = path_join(self.output, self.convertor.lblDirName, lab_name)
            future = self.executor.submit(self._save, img_path, lbl_path, img, lbl_info, width, height, self.img_ext)
            futures.append(future)
            self._count += 1

        self._wait_completed(futures)

    def run(self, args):
        width, height = map(int, args.output_size)
        crop_size = args.crop_size
        crop_mode = args.crop_mode
        crop_ratio = args.crop_ratio
        crop_ratio = min(max(0.1, crop_ratio), 1.0)
        keep_emp_lbl = args.keep_empty_label

        img_data: List[np.ndarray]
        labels: List[ImageLabel]
        for img_data, labels in self.load():
            for image, label in zip(img_data, labels):
                h, w = image.shape[:-1]
                dx = dy = 0
                crop_w, crop_h = w, h
                if crop_size:
                    dx, dy = self.center_crop(w, h, crop_size)
                    crop_w = crop_h = crop_size
                rois = self.cal_roi((crop_h, crop_w), width, height, crop_mode, crop_ratio)
                processed_labels = []
                processed_images = []
                processed_roi_index = []
                for i, (x1, y1, x2, y2) in enumerate(rois):
                    _img = image[dy + y1:dy + y2, dx + x1:dx + x2, :]
                    roi = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)
                    keeps, skips = self.process_label(label, roi,
                                                      min_area=args.min_area,
                                                      max_ratio=args.max_ratio,
                                                      min_ratio=args.min_ratio,
                                                      scale=height / (roi[3] - roi[1]),
                                                      verbose=args.verbose)

                    if not keeps and not keep_emp_lbl:
                        if args.verbose:
                            sys.stderr.write(f"skip: got empty label while `c{args.crop_mode}` "
                                             f"processing the {i}-th roi from image: {label.fileName}\n")
                        if not args.show:
                            continue

                    _lbl = ImageLabel(label.fileName, keeps)
                    if args.show:
                        img = _img.copy()
                        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
                        img = self.annotate_image(img, keeps)

                        for sk_lbl in skips:
                            pt1 = int(sk_lbl.left), int(sk_lbl.top)
                            pt2 = int(sk_lbl.right), int(sk_lbl.bottom)
                            self.dotted_line(img, pt1, pt2, (0, 200, 0), 2)
                        self.show(img)

                    processed_images.append(_img)
                    processed_labels.append(_lbl)
                    processed_roi_index.append(i)
                if self.output is not None:
                    self.save(processed_images, processed_labels, processed_roi_index,
                              args.contiguous, width, height, keep_emp_lbl)

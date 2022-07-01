import os
import argparse
import time

import cv2
import numpy as np

from tqdm import tqdm
from typing import Tuple, List

root_dir = os.path.dirname(os.path.abspath(__file__))
fmt = "%06d"
image_ext = ".png"
label_ext = ".txt"
kitti_fmt = "{} 0 0 0 {} {} {} {} 0 0 0 0 0 0 0\n"
label_names = ["face"]
num_workers = 1


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", type=str, default="Merged", help="Name of merged dataset directory.")
    parser.add_argument("-o", "--output", type=str, default=root_dir, help="Output directory.")
    parser.add_argument("-i", "--inputs", nargs='*', help="Split directories to merge.")
    parser.add_argument("-c", "--contiguous", action="store_true", help=f"Rename image and label file as sequence.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Report dropped images.")
    parser.add_argument("-s", "--show", action="store_true", help="Whether to show annotaed images.")
    parser.add_argument("-j", "--num-workers", type=int, default=num_workers, help="Number of workers (NOT USE YET).")
    parser.add_argument("--subdir", type=str, default="default", help="Subdirectory that images and labels reside.")
    parser.add_argument("--image-dir-name", type=str, default="image_2", help="Directory name of images.")
    parser.add_argument("--label-dir-name", type=str, default="label_2", help="Directory name of labels.")
    parser.add_argument("--ext", type=str, default=image_ext, help="File extension of image.")

    parser.add_argument("--input-size", type=str, default="736x416", help="WxH dimension of model input tensor.")
    parser.add_argument("--crop-size", type=int, default=1100, help="Crop size.")
    parser.add_argument("--offset", type=int, default=0, help="Offset of the shorter size for roi cropping.")
    parser.add_argument("--min-area", type=int, default=25 * 25, help="Minimum area that objects to be ignored.")
    parser.add_argument("--min-ratio", type=float, default=0.36, help="Minimum w/h ratio that objects will be ignored.")
    parser.add_argument("--grayscale", action="store_true", help="Generate grayscale images.")
    parser.add_argument("--keep-empty-label", action="store_true", help="Whether to keep label if it's empty.")
    return parser.parse_args()


def create_kitti_directory(args):
    assert os.path.exists(args.output), f"error: path not exists: {args.output}"

    merged_dir = os.path.join(args.output, args.name)
    if os.path.exists(merged_dir):
        print(f"error: merged dataset: `{args.name}` has already exists, manually delete it first")

    os.makedirs(merged_dir, exist_ok=True)
    image_dir = os.path.join(args.output, args.name, "images")
    label_dir = os.path.join(args.output, args.name, "labels")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)
    return image_dir, label_dir


def center_crop(w: int, h: int, crop_size: int):
    """
    Crop image in the center.

    :param w: Width of original image.
    :param h: Height of original image.
    :param crop_size: Width and height of final cropped image.
    :return: (x1, y1, x2, y2) Coordinate of the cropped image.
    """
    dy = (h - crop_size) // 2
    dx = (w - crop_size) // 2
    return dx, dy


def calc_roi(crop_size: int, tw: int, th: int, offset: int):
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
        offset = max(min(offset, delta), -delta)
        x1 = 0
        y1 = delta + offset
        x2 = crop_size
        y2 = y1 + nh
    else:
        nw = int(tw / th * crop_size)
        delta = (crop_size - nw) // 2
        offset = max(min(offset, delta), -delta)
        x1 = delta + offset
        y1 = 0
        x2 = x1 + nw
        y2 = crop_size
    return x1, y1, x2, y2


def process_image(image: np.ndarray, crop_size: int, tw: int, th: int, offset: int):
    """
    Calculate coordinates of the largest ROI for model input.
    """
    h, w = image.shape[:2]

    # center crop
    dx, dy = center_crop(w, h, crop_size)

    # roi relative to center crop
    x1, y1, x2, y2 = calc_roi(crop_size, tw, th, offset)

    # roi relative to original image
    return x1 + dx, y1 + dy, x2 + dx, y2 + dy


def process_label(lbl_path: str,
                  roi: Tuple[int, int, int, int],
                  min_area: int,
                  min_ratio: float,
                  verbose: bool = False):
    """
    Rectify bounding boxes in a label file given ROI and filters rules.
    """
    MINX, MAXX = 0, roi[2] - roi[0]
    MINY, MAXY = 0, roi[3] - roi[1]

    keeped_objs = []
    skipped_objs = []
    total = 0
    with open(lbl_path, 'r') as f:
        for line in f:
            line_info = line.strip().split()
            label_name = line_info[0]
            label_idx = label_names.index(label_name)

            # absolute coord
            xmin = round(float(line_info[4]))
            ymin = round(float(line_info[5]))
            xmax = round(float(line_info[6]))
            ymax = round(float(line_info[7]))

            # relative coord to roi
            xmin = min(max(MINX, xmin - roi[0]), MAXX)
            ymin = min(max(MINY, ymin - roi[1]), MAXY)
            xmax = min(max(MINX, xmax - roi[0]), MAXX)
            ymax = min(max(MINY, ymax - roi[1]), MAXY)

            total += 1
            w = xmax - xmin
            h = ymax - ymin
            a = w * h
            r = w / (h + 1e-6)

            # filter objects
            if a >= min_area and min_ratio <= r <= 1.0 / min_ratio:
                keeped_objs.append((xmin, ymin, xmax, ymax, w, h, a, r, label_idx))
            else:
                skipped_objs.append((xmin, ymin, xmax, ymax, w, h, a, r, label_idx))

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
    return keeped_objs, skipped_objs


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


def annotate_image(image: np.ndarray, keeps: List[tuple], skips: List[tuple]):
    for obj in keeps:
        xmin = obj[0]
        ymin = obj[1]
        xmax = obj[2]
        ymax = obj[3]
        lbl_name = label_names[obj[-1]]

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(image, lbl_name, (xmin, ymin - 2), cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 255, 0), 1)

    for obj in skips:
        xmin = obj[0]
        ymin = obj[1]
        xmax = obj[2]
        ymax = obj[3]

        dotted_line(image, (xmin, ymin), (xmax, ymax), (0, 200, 0), 2)

    return image


def write_image(path: str, image: np.ndarray, tw: int, th: int):
    image = cv2.resize(image, (tw, th), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(path, image)


def write_label(path: str, objs: List[tuple], scale: float):
    with open(path, 'w') as f:
        for obj in objs:
            label_name = label_names[obj[-1]]
            label_line = kitti_fmt.format(label_name, *[int(x * scale) for x in obj[0:4]])
            f.write(label_line)


def main(args):
    global fmt
    global image_ext
    global num_workers

    image_ext = args.ext
    num_workers = args.num_workers
    assert args.inputs, "error: no input splits to merge"
    for inp in args.inputs:
        assert os.path.exists(inp), f"error: path not exists: {inp}"

    dst_img_dir, dst_lbl_dir = create_kitti_directory(args)
    width, height = args.input_size.split('x')
    width, height = int(width), int(height)
    progress_bar = tqdm(desc=f"{args.name}")

    count = 0
    cv2.namedWindow("ImageShow")
    for inp in args.inputs:
        images_dir = os.path.join(inp, args.subdir, args.image_dir_name)
        labels_dir = os.path.join(inp, args.subdir, args.label_dir_name)
        labels = set(os.listdir(labels_dir))
        progress_bar.total = len(labels)
        for image_name in os.listdir(images_dir):

            if not image_name.endswith(args.ext):
                if args.verbose:
                    print(f"skip an image with wrong extension: {image_name}")
                continue

            label_name = os.path.splitext(image_name)[0] + label_ext
            if label_name not in labels:
                if args.verbose:
                    print(f"skip an image without label: {image_name}")
                continue

            if not args.verbose:
                progress_bar.update()

            src_img = os.path.join(images_dir, image_name)
            src_lbl = os.path.join(labels_dir, label_name)
            if args.consecutive:
                dst_img = os.path.join(dst_img_dir, fmt % count + image_ext)
                dst_lbl = os.path.join(dst_lbl_dir, fmt % count + label_ext)
            else:
                dst_img = os.path.join(dst_img_dir, image_name)
                dst_lbl = os.path.join(dst_lbl_dir, label_name)

            if args.verbose:
                print(f"\n{image_name}")

            # Read original image
            t0 = time.perf_counter()
            image = cv2.imread(src_img, cv2.IMREAD_COLOR)

            t1 = time.perf_counter()
            # Calculate final cropped roi coordinates
            roi = process_image(image, args.crop_size, width, height, args.offset)

            t2 = time.perf_counter()
            # Filter out objects
            keeps, skips = process_label(src_lbl, roi, args.min_area, args.min_ratio, args.verbose)

            if not args.keep_empty_label and not keeps:
                continue
            count += 1

            # Annotate ROI with ground truth boxes
            image = image[roi[1]:roi[3], roi[0]:roi[2]]
            if args.grayscale:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Show annotated ROI
            if args.show:
                image_copy = image.copy()
                if args.grayscale:
                    image_copy = cv2.merge([image_copy] * 3)

                annotate_image(image_copy, keeps, skips)
                cv2.imshow("ImageShow", image_copy)
                k = cv2.waitKey(0)
                if k == ord('q') or k == 27:
                    break

            t3 = time.perf_counter()

            write_image(dst_img, image, width, height)
            write_label(dst_lbl, keeps, height / (roi[3] - roi[1]))

            t4 = time.perf_counter()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(get_args())

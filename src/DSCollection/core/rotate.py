# @Time    : 2022/6/27 下午3:57
# @Author  : Boyang
# @Site    : 
# @File    : rotate.py
# @Software: PyCharm
import math
from functools import lru_cache
from typing import Tuple

import cv2
import numpy as np


@lru_cache()
def _get_rotation_matrix_2d(center: Tuple[int, int], degree: int, scale: float = 1.0):
    return cv2.getRotationMatrix2D(center, degree, scale)


def rotate_image(image: np.ndarray, degree: int):
    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2
    M = _get_rotation_matrix_2d((cx, cy), degree)
    return cv2.warpAffine(image, M, (w, h))


def sin(degree):
    return math.sin(math.radians(degree))


def cos(degree):
    return math.cos(math.radians(degree))


def rotate_point_by_point(point1: Tuple[float, float], center_point: Tuple[float, float], degree: int, height: int):
    """Rotate point by one point"""
    x1, y1 = point1
    cx, cy = center_point
    y1, cy = height - y1, height - cy  # 翻转y轴
    x1, y1 = x1 - cx, y1 - cy  # 换原点
    rx1 = x1 * cos(degree) - y1 * sin(degree)
    ry1 = x1 * sin(degree) + y1 * cos(degree)

    x1, y1 = rx1 + cx, height - (ry1 + cy)

    return x1, y1


PointType = Tuple[int, int]


def calculate_four_point(label: Tuple[float, float, float, float]) -> Tuple[PointType, PointType, PointType, PointType]:
    ltx, lty = label[:2]
    rbx, rby = label[2:]

    rtx, rty = rbx, lty
    lbx, lby = ltx, rby

    return (ltx, lty), (rtx, rty), (lbx, lby), (rbx, rby)


def calculate_max_bbox(*points):
    x_list = [p[0] for p in points]
    y_list = [p[1] for p in points]
    minx = min(x_list)
    miny = min(y_list)
    maxx = max(x_list)
    maxy = max(y_list)

    return minx, miny, maxx, maxy


def calculate_mid_bbox(lt: Tuple[float, float], rt: Tuple[float, float], rb: Tuple[float, float],
                       lb: Tuple[float, float], div: int):
    mid_p1 = (lt[0] + rt[0]) / div, (lt[1] + lt[1]) / div
    mid_p2 = (rt[0] + rb[0]) / div, (rt[1] + rb[1]) / div
    mid_p3 = (rb[0] + lb[0]) / div, (rb[1] + lb[1]) / div
    mid_p4 = (lb[0] + lt[0]) / div, (lb[1] + lt[1]) / div

    xmin = min([mid_p1[0], mid_p2[0], mid_p3[0], mid_p4[0]])
    xmax = max([mid_p1[0], mid_p2[0], mid_p3[0], mid_p4[0]])
    ymin = min([mid_p1[1], mid_p2[1], mid_p3[1], mid_p4[1]])
    ymax = max([mid_p1[1], mid_p2[1], mid_p3[1], mid_p4[1]])

    return xmin, ymin, xmax, ymax


def calculate_midpoint_bbox(*points):
    point_arr = np.array(points)
    b = point_arr.shape[0]
    mid_point = (point_arr[..., np.newaxis, :] + point_arr[np.newaxis, ...]) / 2

    mid_point_mask = np.identity(b) == 0
    mid_point = mid_point[mid_point_mask]
    mid_point = mid_point.reshape(-1, 2)

    xmin = mid_point[:, 0].min()
    ymin = mid_point[:, 1].min()
    xmax = mid_point[:, 0].max()
    ymax = mid_point[:, 1].max()

    return xmin, ymin, xmax, ymax


def _rotate_label(label: Tuple[float, float, float, float], degree: int, h: int, w: int):
    """Rotate four point"""
    lt, rt, lb, rb = calculate_four_point(label)
    cx, cy = w // 2, h // 2
    rlt = rotate_point_by_point(lt, (cx, cy), degree, h)
    rrt = rotate_point_by_point(rt, (cx, cy), degree, h)
    rlb = rotate_point_by_point(lb, (cx, cy), degree, h)
    rrb = rotate_point_by_point(rb, (cx, cy), degree, h)

    return rlt, rrt, rlb, rrb


def _calculate_divide_points(point1: Tuple[float, float], point2: Tuple[float, float], div: int, h: int, w: int):
    pt1 = np.array(point1)
    pt2 = np.array(point2)
    points = []
    i = 1
    while i < div:
        factor = i / (div - i)
        p = (pt1 + factor * pt2) / (factor + 1)
        x, y = p.tolist()
        x = min(max(0, x), w)
        y = min(max(0, y), h)
        points.append((x, y))
        i += 1

    return points


def rotated_calculate_max_bbox(label: Tuple[float, float, float, float], degree: int, h: int, w: int):
    rlf, rlr, rlb, rrb = _rotate_label(label, degree, h, w)
    return calculate_max_bbox(rlf, rlr, rlb, rrb)


def rotated_calculate_midpoint_bbox(label: Tuple[float, float, float, float], degree: int, h: int, w: int):
    rlf, rlr, rlb, rrb = _rotate_label(label, degree, h, w)

    return calculate_midpoint_bbox(rlf, rlr, rlb, rrb)


def rotated_calculate_divide_point_bbox(label: Tuple[float, float, float, float], degree: int, h: int, w: int,
                                        div: int):
    rlt, rrt, rlb, rrb = _rotate_label(label, degree, h, w)
    points = _calculate_divide_points(rlt, rrt, div, h, w)
    points.extend(_calculate_divide_points(rlt, rlb, div, h, w))
    points.extend(_calculate_divide_points(rrt, rrb, div, h, w))
    points.extend(_calculate_divide_points(rlb, rrb, div, h, w))
    points = np.array(points)
    return calculate_max_bbox(*points.tolist())

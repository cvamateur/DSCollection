import sys
import os.path
import threading

import cv2
import numpy as np
import matplotlib.pyplot as plt

from queue import Queue
from enum import IntEnum
from collections import defaultdict
from abc import ABC, abstractmethod
from typing import List, Union, Tuple,DefaultDict

from .dataset import Dataset
from .img_util import ImageUtil

_T_COLOR = Tuple[int, int, int]


class VBackend(IntEnum):
    OPENCV = 0
    PYPLOT = 1


MAX_VIS_CLASSES: int = 10


class Visualizer:

    def __init__(self, backend: Union[int, VBackend] = VBackend.OPENCV, cMap: str = "hsv"):
        if backend == VBackend.OPENCV:
            self._bk = _OpenCV_Backend()
        elif backend == VBackend.PYPLOT:
            self._bk = _Pyplot_Backend()
        else:
            msg = f"error: unknown vbackend: {backend}"
            raise RuntimeError(msg)
        self._buffer = Queue(maxsize=2)
        self._cMap = get_cmap(MAX_VIS_CLASSES, cMap)
        self._clsColorMap: DefaultDict[str, _T_COLOR] = defaultdict(self.new_color)

    def new_color(self):
        idx = len(self._clsColorMap)
        while idx >= MAX_VIS_CLASSES:
            idx -= MAX_VIS_CLASSES
        c = self._cMap(len(self._clsColorMap))
        return int(256*c[2]), int(256*c[1]), int(256*c[0])

    def _read_img(self, imgPaths: List[str]):
        for i, path in enumerate(imgPaths):
            if not os.path.exists(path):
                sys.stderr.write(f"warn: skip image not exist {path}")
            else:
                imgRaw = open(path, 'rb').read()
                self._buffer.put((i, imgRaw), block=True)
        else:
            self._buffer.put((None, None), block=True)

    def show(self, ds: Dataset, clsNames: List[str] = None, nImgs: Union[int, float] = 10, showName: bool = True):
        ds.load(clsNames, nImgs)
        imgPaths = [os.path.join(ds.root, ds.imgDirName, l.fileName) for l in ds.labels]

        readThread = threading.Thread(target=self._read_img, args=(imgPaths,))
        readThread.daemon = True
        readThread.start()

        try:
            while True:
                index, img = self._buffer.get(timeout=5)
                if img is None: break
                img = ImageUtil.decode(img)
                label = ds.labels[index]
                for box in label.boxes:
                    clsName = box.name
                    pt1 = int(box.left), int(box.top)
                    pt2 = int(box.right), int(box.bottom)
                    color = self._clsColorMap[clsName]
                    cv2.rectangle(img, pt1, pt2, color, 1, cv2.LINE_AA)
                    if showName:
                        cv2.putText(img, clsName, (pt1[0], pt1[1] - 2), cv2.FONT_HERSHEY_PLAIN, 1.0, color, 1)
                self._bk.visualize(img)
        except Exception as e:
            sys.stderr.write(str(e))
            return


class _VisBackend(ABC):

    @abstractmethod
    def visualize(self, image: np.ndarray):
        raise NotImplementedError


class _OpenCV_Backend(_VisBackend):

    def __init__(self, winName: str = "Image"):
        self.winName = winName
        self.window = cv2.namedWindow(winName)

    def visualize(self, image: np.ndarray):

        cv2.imshow(self.winName, image)
        k = cv2.waitKey(0) & 0xFF
        if k in [ord('q'), 27]:
            raise SystemExit

    def __del__(self):
        cv2.destroyWindow(self.winName)


class _Pyplot_Backend(_VisBackend):

    def __init__(self, rows: int = 1, cols: int = 1):
        self.rows = rows
        self.cols = cols
        self._i = 1

    def visualize(self, image: np.ndarray):
        plt.subplot(self.rows, self.rows, self._i)
        plt.imshow(image[..., ::-1])
        if self._i == self.rows * self.cols:
            plt.show()
            self._i = 1
        else:
            self._i += 1


def get_cmap(n, name='prism'):
    """
    Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGBA color; the keyword argument name must be a standard mpl colormap name.
    [ColorMap](https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.Colormap.html#matplotlib.colors.Colormap)
    [ColorMap-Names](https://matplotlib.org/stable/gallery/color/colormap_reference.html)
    @Params
    -------
    n (int):
        Maps colors into a range of distinct int values [0, n-1]
    name (string):
        colorMap names, refer to the link `[ColorMap-Names]` for more details.
    @Returns
    -------
        A colorMap instance, which is a callable function that takes an int number and returns a tuple
        of RGBA color.
    @Usage
    ------
    cmap = get_cmap(len(data))
    for i, (X, Y) in enumerate(data):
        scatter(X, Y, c=cmap(i))
    """
    return plt.cm.get_cmap(name, n)

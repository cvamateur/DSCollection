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
from tqdm import tqdm

from src.utils.common import check_path
from .dataset import Dataset
from src.utils.imgutil import ImageUtil

_T_COLOR = Tuple[int, int, int]


class VBackend(IntEnum):
    OPENCV = 0
    PYPLOT = 1


MAX_VIS_CLASSES: int = 10


class Visualizer:

    def __init__(self, backend: Union[int, VBackend] = VBackend.OPENCV, cMap: str = "hsv", **kwargs):
        if backend == VBackend.OPENCV:
            winName = kwargs.pop("winName", "Image")
            self._bk = _OpenCV_Backend(winName)
        elif backend == VBackend.PYPLOT:
            rows = kwargs.pop("rows", 1)
            cols = kwargs.pop("cols", 1)
            self._bk = _Pyplot_Backend(rows, cols)
        else:
            msg = f"error: unknown vbackend: {backend}"
            raise RuntimeError(msg)
        self._buffer = Queue(maxsize=2)
        self._cMap = plt.cm.get_cmap(cMap, MAX_VIS_CLASSES)
        self._clsColorMap: DefaultDict[str, _T_COLOR] = defaultdict(self.new_color)

    def new_color(self) -> _T_COLOR:
        idx = len(self._clsColorMap)
        while idx >= MAX_VIS_CLASSES:
            idx -= MAX_VIS_CLASSES
        c = self._cMap(len(self._clsColorMap))
        return int(256*c[0]), int(256*c[1]), int(256*c[2])

    def _read_img(self, imgPaths: List[str]):
        for i, path in enumerate(imgPaths):
            if not os.path.exists(path):
                sys.stderr.write(f"warn: skip image not exist {path}")
            else:
                imgRaw = open(path, 'rb').read()
                self._buffer.put((i, path, imgRaw), block=True)
        else:
            self._buffer.put((None, None, None), block=True)

    def show(self, ds: Dataset,
             clsNames: List[str] = None,
             nImgs: Union[int, float] = 10,
             showName: bool = True,
             outputDir: str = None):

        if outputDir is not None:
            outputDir = check_path(outputDir, existence=True)
            outputDir = os.path.join(outputDir, "AnnotatedImages")
            os.makedirs(outputDir, exist_ok=True)
        ds.load(clsNames, nImgs)
        imgPaths = [os.path.join(ds.root, ds.imgDirName, l.fileName) for l in ds.labels]

        readThread = threading.Thread(target=self._read_img, args=(imgPaths,))
        readThread.daemon = True
        readThread.start()

        progress = tqdm(total=len(imgPaths), desc="Visualize Dataset")
        try:
            while True:
                index, path, img = self._buffer.get(timeout=5)
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
                progress.update()
                if outputDir is not None:
                    outPath = os.path.join(outputDir, os.path.basename(path))
                    cv2.imwrite(outPath, img)

        except Exception as e:
            progress.close()
            sys.stderr.write("error: %s\n" % e)
            sys.exit(-1)
        finally:
            self._bk.release()


class _VisBackend(ABC):

    @abstractmethod
    def visualize(self, image: np.ndarray):
        raise NotImplementedError

    def release(self):
        pass


class _OpenCV_Backend(_VisBackend):

    def __init__(self, winName: str = None):
        if winName is None: winName = "Image"
        self.winName = winName
        self.window = cv2.namedWindow(winName)

    def visualize(self, image: np.ndarray):

        cv2.imshow(self.winName, image)
        k = cv2.waitKey(0) & 0xFF
        if k in [ord('q'), 27]:
            raise SystemExit

    def release(self):
        cv2.destroyWindow(self.winName)


class _Pyplot_Backend(_VisBackend):

    def __init__(self, rows: int = None, cols: int = None):
        if rows is None: rows = 1
        if cols is None: cols = 1
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

    def release(self):
        plt.show()
# @Time    : 2022/7/15 下午6:24
# @Author  : Boyang
# @Site    : 
# @File    : common.py
# @Software: PyCharm
import logging
from concurrent.futures.thread import ThreadPoolExecutor
from concurrent.futures import wait, ALL_COMPLETED
from typing import List, Dict
from os.path import join as path_join
from concurrent.futures import Future

from ..core.dataset import Dataset, ImageLabel
from ..utils.imgutil import ImageUtil

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MAX_WORKER = 5
CHUNK_SIZE = 10

_executor = ThreadPoolExecutor(max_workers=MAX_WORKER)


# TODO :recycle:

def _chunkify(size: int, iter_data: List[ImageLabel]):
    res = []
    for lab in iter_data:
        res.append(lab)
        if len(res) == size:
            yield res
            res.clear()

    yield res


def _read_image(img_path: str):
    with open(img_path, "rb") as f:
        b = f.read()
        return ImageUtil.decode(b)


def read_image(labels: List[ImageLabel], root: str, img_dir_name: str) -> Dict[Future, ImageLabel]:
    future = {}
    for lbl in labels:
        img_path = path_join(root, img_dir_name, lbl.fileName)
        future.update({_executor.submit(_read_image, img_path): lbl})

    return future


def _wait_completed(fs):
    res = wait(fs, return_when=ALL_COMPLETED)
    if res.not_done:
        logger.warning("Exist failed task")


def chunk_dataset(dataset: Dataset):
    dataset.load(skip_empty=False)
    iter_labels = _chunkify(CHUNK_SIZE, dataset.labels)
    labels = next(iter_labels)
    pre_data = read_image(labels, dataset.root, dataset.imgDirName)
    # if len(pre_data) == 0:
    #     continue

    while True:
        _wait_completed(pre_data)
        labels = next(iter_labels)
        next_data = read_image(labels, dataset.root, dataset.imgDirName)
        yield [f.result() for f in pre_data.keys()], [l for l in pre_data.values()]
        pre_data, next_data = next_data, None

        if len(pre_data) < CHUNK_SIZE:
            _wait_completed(pre_data)
            yield [f.result() for f in pre_data.keys()], [l for l in pre_data.values()]
            break

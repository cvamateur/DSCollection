import os.path
import dataclasses
from abc import ABC, abstractmethod
from concurrent.futures import as_completed, ProcessPoolExecutor
from typing import Dict, Any, List, Type, Union

from tqdm import tqdm

from ..utils.common import check_path
from ..utils.voc import dict_to_xml, pretty_xml
from .dataset import Dataset, DatasetType, ImageLabel


@dataclasses.dataclass(frozen=True, repr=False, eq=False)
class LabelInfo:
    """
    Structure holding information about a hold annotation file.

    A convertor usually iterate every `ImageLabel` loaded by `Dataset`,
    and convert to a list of `LabelInfo`.

    @Attributes:
    ------------
    index (int):
        An unsigned integer of the index in the converted list.

    data (bytes):
        Serialized bytes that can be directly writen by `Writer` which
        implement Writer.write(data) method to write label file.
    imgName (str):
        Filename of the original image.
    """
    __slots__ = "index", "data", "imgName"
    index: int
    data: bytes
    imgName: str

    def __getstate__(self):
        return dict(
            (slot, getattr(self, slot))
            for slot in self.__slots__
            if hasattr(self, slot))

    def __setstate__(self, state):
        # BUG FIX: Unpickable frozen dataclasses
        # https://stackoverflow.com/questions/55307017/pickle-a-frozen-dataclass-that-has-slots
        for slot, value in state.items():
            object.__setattr__(self, slot, value)


path_join = os.path.join


class Convertor(ABC):
    _known_cvt = {}
    lblExt: str
    imgDirName = "images"
    lblDirName = "labels"

    @classmethod
    def from_type(cls, dtype: str, *args, **kwargs) -> "Convertor":
        if dtype not in cls._known_cvt:
            msg = f"Unknown dataset type: {dtype}"
            raise ValueError(msg)
        return cls._known_cvt[dtype](*args, **kwargs)

    @classmethod
    def create_dataset(cls, root: str, name: str, imgDirName: str = None, lblDirName: str = None):
        dstDir = check_path(root, True)
        root = os.path.join(dstDir, name)
        imgDir = os.path.join(root, imgDirName if lblDirName else cls.imgDirName)
        lblDir = os.path.join(root, lblDirName if lblDirName else cls.lblDirName)
        os.makedirs(imgDir, exist_ok=True)
        os.makedirs(lblDir, exist_ok=True)
        return root

    @abstractmethod
    def convert(self, ds: Dataset) -> List[LabelInfo]:
        raise NotImplementedError

    @abstractmethod
    def convert_label(self, index: int, label: ImageLabel) -> LabelInfo:
        raise NotImplementedError

    def __init_subclass__(cls, dtype: str = None, **kwargs):
        super().__init_subclass__(**kwargs)
        if dtype is not None:
            cls._known_cvt[dtype] = cls


class VOCConvertor(Convertor, dtype=DatasetType.VOC):
    lblExt = ".xml"
    imgDirName = "JPEGImages"
    lblDirName = "Annotations"

    def __init__(self):
        super(VOCConvertor, self).__init__()
        self.n_workers = os.cpu_count() - 1
        self._last_n = 0

    def convert(self, ds: Dataset) -> List[LabelInfo]:
        results: List[Union[LabelInfo, None]] = [None] * len(ds.labels)

        futures = []
        with ProcessPoolExecutor(self.n_workers) as executor:
            for i, label in enumerate(ds.labels):
                future = executor.submit(self.convert_label, self._last_n + i, label)
                futures.append(future)

            for future in tqdm(as_completed(futures), total=len(futures), desc="Convert dataset"):
                lblInfo: LabelInfo = future.result()
                results[lblInfo.index] = lblInfo

        self._last_n += len(ds.labels)
        return results

    @staticmethod
    def _label_dict(label: ImageLabel) -> Dict[str, Any]:
        d = {
            "filename": label.fileName,
            "size": {
                "width": label.width,
                "height": label.height,
                "depth": label.depth,
            },
            "object": []
        }
        for box in label.boxes:
            box_dict = {
                "name": box.name,
                "truncated": 0,
                "difficult": 0,
                "bndbox": {
                    "xmin": box.left,
                    "ymin": box.top,
                    "xmax": box.right,
                    "ymax": box.bottom,
                }
            }
            d.setdefault("object", []).append(box_dict)
        return d

    def cvt_label_bytes(self, label: ImageLabel) -> bytes:
        label_dict = self._label_dict(label)
        xml_node = dict_to_xml(label_dict)
        return pretty_xml(xml_node).encode("utf-8")

    def convert_label(self, index: int, label: ImageLabel) -> LabelInfo:
        xml_str = self.cvt_label_bytes(label)
        return LabelInfo(index, xml_str, label.fileName)


class KITTIConvertor(VOCConvertor, dtype=DatasetType.KITTI):
    lblExt = ".txt"
    imgDirName = "images"
    lblDirName = "labels"

    def convert_label(self, index: int, label: ImageLabel) -> LabelInfo:
        kitti_str = self.cvt_label_bytes(label)
        return LabelInfo(index, kitti_str, label.fileName)

    def cvt_label_bytes(self, label: ImageLabel) -> bytes:
        label_fmt = "{} 0 0 0 {} {} {} {} 0 0 0 0 0 0 0"
        kitti_lines = [label_fmt.format(box.name, *box.coord) for box in label.boxes]
        return '\n'.join(kitti_lines).encode("utf-8")
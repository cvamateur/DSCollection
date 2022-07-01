import os.path
import dataclasses
from abc import ABC, abstractmethod
from concurrent.futures import as_completed, ProcessPoolExecutor
from typing import Dict, Any, List, Type, Union

from tqdm import tqdm

from .voc import dict_to_xml, pretty_xml
from .dataset import Dataset, DatasetFormat, ImageLabel


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


path_join = os.path.join


class Convertor(ABC):
    _known_cvt = {}
    lblExt: str
    imgDirName = "images"
    lblDirName = "labels"

    @classmethod
    def from_type(cls, dtype: DatasetFormat) -> Type["Convertor"]:
        if dtype not in cls._known_cvt:
            msg = f"Unknown dataset type: {dtype}"
            raise TypeError(msg)
        return cls._known_cvt[dtype]

    @abstractmethod
    def convert(self, ds: Dataset) -> List[LabelInfo]:
        raise NotImplementedError

    def __init_subclass__(cls, dtype=None, **kwargs):
        super().__init_subclass__(**kwargs)
        if dtype is not None:
            cls._known_cvt[dtype] = cls


class VOCConvertor(Convertor, dtype=DatasetFormat.VOC):
    lblExt = ".xml"
    imgDirName = "JPEGImages"
    lblDirName = "Annotations"

    def __init__(self):
        super(VOCConvertor, self).__init__()
        self.n_workers = os.cpu_count() - 1

    def convert(self, ds: Dataset) -> List[LabelInfo]:
        results: List[Union[LabelInfo, None]] = [None] * len(ds.labels)

        futures = []
        with ProcessPoolExecutor(self.n_workers) as executor:
            for i, label in enumerate(ds.labels):
                future = executor.submit(self.cvt_label, i, label)
                futures.append(future)

            for future in tqdm(as_completed(futures), total=len(futures), desc="Convert dataset"):
                lblInfo: LabelInfo = future.result()
                results[lblInfo.index] = lblInfo

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

    def cvt_label(self, index: int, label: ImageLabel) -> LabelInfo:
        label_dict = self._label_dict(label)
        xml_node = dict_to_xml(label_dict)
        xml_str = pretty_xml(xml_node).encode("utf-8")
        return LabelInfo(index, xml_str, label.fileName)


class KITTIConvertor(VOCConvertor, dtype=DatasetFormat.KITTI):
    lblExt = ".txt"
    imgDirName = "images"
    lblDirName = "labels"

    def cvt_label(self, index: int, label: ImageLabel) -> LabelInfo:
        label_fmt = "{} 0 0 0 {} {} {} {} 0 0 0 0 0 0 0"
        kitti_lines = [label_fmt.format(box.name, *box.coord) for box in label.boxes]
        kitti_str = '\n'.join(kitti_lines).encode("utf-8")
        return LabelInfo(index, kitti_str, label.fileName)

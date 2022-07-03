import os.path
import sys
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Union, Type

from tqdm import tqdm
from pycocotools.coco import COCO as coco

from src.utils.common import check_path, silent, is_image
from src.utils.voc import xml_to_dict

_T_coord = Union[int, float]


@dataclass(frozen=True, repr=False, eq=False)
class Box:
    __slots__ = "left", "top", "right", "bottom", "name"
    left: _T_coord
    top: _T_coord
    right: _T_coord
    bottom: _T_coord
    name: str

    def __getstate__(self):
        return dict(
            (slot, getattr(self, slot))
            for slot in self.__slots__
            if hasattr(self, slot)
        )

    def __setstate__(self, state):
        # BUG FIX: Unpickable frozen dataclasses
        # https://stackoverflow.com/questions/55307017/pickle-a-frozen-dataclass-that-has-slots
        for slot, value in state.items():
            object.__setattr__(self, slot, value)

    @property
    def coord(self):
        return self.left, self.top, self.right, self.bottom


del _T_coord


@dataclass
class ImageLabel:
    fileName: str
    boxes: List[Box]
    width: int = field(default=-1)
    height: int = field(default=-1)
    depth: int = field(default=-1)


class DatasetType:
    """
    DsCollection legacy output one the three types dataset formats:
        1. VOC:
            Images are saved in subdir `JPEGImages`;
            Labels are saved as xml files under subdir `Annotations`;

        2. KITTI
            Images are saved in subdir `images`;
            Labels are saved as txt files under subdir `labels`

        3. COCO
            Currently not supported.
    """
    VOC     = "VOC"
    KITTI   = "KITTI"
    COCO    = "COCO"


class Dataset(ABC):
    _known_ds = {}
    imgDirName: str
    lblDirName: str
    lblExt: str

    @abstractmethod
    def __init__(self, root: str):
        self.root = check_path(root, True)
        self.labels: List[ImageLabel] = []

    def __len__(self):
        return self.labels.__len__()

    def __iter__(self):
        for label in self.labels:
            yield label

    def __init_subclass__(cls, dtype: str = None, **kwargs):
        super().__init_subclass__(**kwargs)
        if dtype is not None:
            cls._known_ds[dtype] = cls

    @classmethod
    def from_type(cls, dtype: str) -> Type["Dataset"]:
        if dtype not in cls._known_ds:
            msg = f"Unknown dataset name {dtype}"
            raise ValueError(msg)
        return cls._known_ds[dtype]

    @classmethod
    def create_structure(cls, dstDir: str, dsName: str) -> str:
        dstDir = check_path(dstDir, True)
        root = check_path(os.path.join(dstDir, dsName), False)
        os.makedirs(os.path.join(root, cls.imgDirName))
        os.makedirs(os.path.join(root, cls.lblDirName))
        return root

    @classmethod
    def check_is_correct_path(cls, root: str) -> bool:
        try:
            check_path(root, existence=True)
            check_path(os.path.join(root, cls.imgDirName), existence=True)
            check_path(os.path.join(root, cls.lblDirName), existence=True)
        except FileNotFoundError:
            return False
        else:
            return True

    @classmethod
    def find_dataset(cls, root: str) -> Union[Type["Dataset"], None]:
        check_path(root, existence=True)
        for klass in cls._known_ds.values():
            if klass.check_is_correct_path(root):
                return klass
        return None

    @classmethod
    def get_known_datasets(cls):
        return list(cls._known_ds.keys())

    @abstractmethod
    def load(self, clsNames: List[str] = None, nImgs: Union[int, float] = None):
        """
        Load all labels into a list of ImageLabel.
        If clsNames is given, then only load labels of those classes.

        @Arguments:
        ----------
        clsName (List[str]):
            A list of class names to load.

        nImgs (int | float):
            An integer for number of labels to load, or a float number
            between (0.0, 1.0] for portion of labels to load.
        """
        raise NotImplementedError


class VOC(Dataset, dtype=DatasetType.VOC):
    imgDirName = "JPEGImages"
    lblDirName = "Annotations"
    lblExt = ".xml"

    def __init__(self, root: str):
        super(VOC, self).__init__(root)

    def load(self, clsNames: List[str] = None, nImgs: Union[int, float] = None):
        if isinstance(nImgs, float):
            if nImgs <= 0.0 or nImgs > 1.0:
                msg = "nImgs must be float between (0.0, 1.0]"
                raise RuntimeError(msg)

        imageRoot = os.path.join(self.root, self.imgDirName)
        labelRoot = os.path.join(self.root, self.lblDirName)
        imageFnames = [p for p in os.listdir(imageRoot) if is_image(p)]
        labelFnames = [os.path.splitext(p)[0] + self.lblExt for p in imageFnames]
        imagePaths = [os.path.join(imageRoot, name) for name in imageFnames]
        labelPaths = [os.path.join(labelRoot, name) for name in labelFnames]
        if len(imagePaths) != len(labelPaths):
            msg = "Images not match with labels, dataset corrupted"
            raise RuntimeError(msg)
        if not all(map(os.path.exists, imagePaths)):
            msg = "Not all images exists, dataset corrupted"
            raise RuntimeError(msg)
        if not all(map(os.path.exists, labelPaths)):
            msg = "Not all labels exists, dataset corrupted"
            raise RuntimeError(msg)

        if nImgs is not None:
            if isinstance(nImgs, float):
                nImgs = int(nImgs * len(imagePaths))
            if isinstance(nImgs, int):
                indices = random.sample(list(range(len(imagePaths))), k=min(nImgs, len(imagePaths)))
                labelPaths = [labelPaths[i] for i in indices]
                imageFnames = [imageFnames[i] for i in indices]
            else:
                msg = "nImgs must be an integer or float number"
                raise RuntimeError(msg)

        labels = []
        progress = tqdm(zip(imageFnames, labelPaths), total=len(imageFnames), desc="Load dataset")
        for i, (imgName, lblFile) in enumerate(progress):
            if nImgs is not None and i >= nImgs:
                break
            label = self.load_label(imgName, lblFile, clsNames)
            if label is not None:
                labels.append(label)
            else:
                sys.stderr.write(f"warn: skip image with empty label: {imgName}\n")
        self.labels = labels

    @staticmethod
    def load_label(imgName: str, lblFile: str, clsNames: List[str]) -> Union[ImageLabel, None]:
        ann_dict = xml_to_dict(lblFile)["annotation"]
        try:
            width = ann_dict["size"]["width"]
            height = ann_dict["size"]["height"]
            depth = ann_dict["size"].get("depth", 3)
            objects = ann_dict["object"]
        except KeyError:
            return

        boxes = []
        for obj in objects:
            try:
                clsName = obj["name"]
                bndbox = obj["bndbox"]
                xmin = round(float(bndbox["xmin"]))
                ymin = round(float(bndbox["ymin"]))
                xmax = round(float(bndbox["xmax"]))
                ymax = round(float(bndbox["ymax"]))
            except KeyError:
                continue
            if clsNames is None or clsName in clsNames:
                boxes.append(Box(xmin, ymin, xmax, ymax, clsName))

        if boxes:
            return ImageLabel(imgName, boxes, width, height, depth)


class KITTI(VOC, dtype=DatasetType.KITTI):
    imgDirName = "images"
    lblDirName = "labels"
    lblExt = ".txt"

    def __init__(self, root: str, imgDir: str = "images", lblDir: str = "labels"):
        super(KITTI, self).__init__(root)
        self.imgDirName = imgDir
        self.lblDirName = lblDir

    @staticmethod
    def load_label(imgName: str, lblFile: str, clsNames: List[str]) -> Union[ImageLabel, None]:
        with open(lblFile, 'r') as f:
            kitti_lines = f.readlines()

        boxes = []
        for line in kitti_lines:
            line_info = line.strip().split()
            try:
                clsName = line_info[0]
                xmin = round(float(line_info[4]))
                ymin = round(float(line_info[5]))
                xmax = round(float(line_info[6]))
                ymax = round(float(line_info[7]))
            except (IndexError, ValueError):
                continue
            if clsNames is None or clsName in clsNames:
                boxes.append(Box(xmin, ymin, xmax, ymax, clsName))

        if boxes:
            return ImageLabel(imgName, boxes)


class COCO(Dataset, dtype=DatasetType.COCO):
    lblDirName = "annotations"
    imgDirName = "{split}{year}"

    SPLITS = ["train", "val", "test"]
    YEARS = [2017]

    def __init__(self, root: str, split: str = "train", year: int = 2017):
        super(COCO, self).__init__(root)
        assert split in self.SPLITS, f"split must be one of {self.SPLITS}"
        assert year in self.YEARS, f"year must be one of {self.YEARS}"
        self.imgDirName.format(split=split, year=year)

    def load(self, clsNames: List[str] = None, nImgs: Union[int, float] = None):

        if isinstance(nImgs, float):
            if nImgs <= 0.0 or nImgs > 1.0:
                msg = "nImgs must be float between (0.0, 1.0]"
                raise RuntimeError(msg)

        ann_file = os.path.join(self.root, self.lblDirName, f"instances_{self.imgDirName}.json")
        sys.stdout.write("Load annotations to memory...")
        with silent():
            _coco = coco(ann_file)

        if clsNames is None:
            clsNames = []
        catIds = _coco.getCatIds(catNms=clsNames)
        mapId2Cls = dict((d["id"], d["name"]) for d in _coco.loadCats(catIds))
        imgIds = set()
        for catId in catIds:
            imgId_list = _coco.getImgIds(catIds=[catId])
            imgIds.update(imgId_list)
        imgIds = list(imgIds)

        if isinstance(nImgs, float):
            nImgs = int(nImgs * len(imgIds))
        if isinstance(nImgs, int):
            indices = random.sample(list(range(len(imgIds))), k=min(nImgs, len(imgIds)))
            imgIds = [imgIds[i] for i in indices]
        else:
            msg = "nImgs must be an integer or float number"
            raise RuntimeError(msg)

        labels = []
        for i, imgId in tqdm(enumerate(imgIds), desc="Load dataset", total=len(imgIds)):
            if nImgs is not None and i >= nImgs:
                break
            label, imgInfo = self.load_label(_coco, imgId, catIds, mapId2Cls)
            if label is not None:
                labels.append(label)
            else:
                imgName = imgInfo.get("file_name", f"imgId={imgId}")
                sys.stderr.write(f"warn: skip image with empty label: {imgName}\n")

        self.labels = labels

    def load_label(self, c: coco, imgId: int, catIds: List[int],
                   mapId2cls: Dict[int, str]) -> Tuple[Union[ImageLabel, None], Dict[str, Any]]:

        imgInfo = c.loadImgs(imgId)[0]
        annIds = c.getAnnIds(imgIds=imgId, catIds=catIds)
        anns = c.loadAnns(annIds)

        try:
            imgName = imgInfo["file_name"]
            width = imgInfo["width"]
            height = imgInfo["height"]
            depth = imgInfo.get("depth", 3)
        except KeyError:
            return None, imgInfo

        try:
            srcImg = os.path.join(self.root, self.imgDirName, imgName)
            check_path(srcImg, existence=True)
        except FileNotFoundError:
            return None, imgInfo

        boxes = []
        for ann in anns:
            try:
                clsName = mapId2cls[ann["category_id"]]
                xmin = round(float(ann["bbox"][0]))
                ymin = round(float(ann["bbox"][1]))
                xmax = round(float(ann["bbox"][2])) + xmin
                ymax = round(float(ann["bbox"][3])) + ymin
            except (KeyError, ValueError):
                continue
            if not ann.get("iscrowd", 0):
                boxes.append(Box(xmin, ymin, xmax, ymax, clsName))

        if boxes:
            return ImageLabel(imgName, boxes, width, height, depth), imgInfo
        else:
            return None, imgInfo


class WiderFace(Dataset):
    ...


class WiderPersion(Dataset):
    ...
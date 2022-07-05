import glob
import os.path
import sys
import ast
import random
import configparser
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    """
    Base class for parsing a dataset. To extend and register more dataset types
    that recognized by this base class, one must implement:

    - self.load():
        Load all or part of labels into a list of `ImageLabel` objects.

    If subclass with additional keyword argument `dtype=<name>`, the subclass
    will be registered to Dataset, thus users are able to call `Dataset.find_dataset()`
    from base class to find correct subclass, like VOC or KITTI, if the subclass
    implements `self.check_is_correct_path()` correctly.

    NOTE:
        __init__() in subclasses should only accept `root (str)` as positional argument,
        other arguments should pass by keyword arguments. Be aware that keyword argument
        could be None if using cli command parser, thus check argument is None is necessary.
    """
    _known_ds = {}
    imgDirName: str
    lblDirName: str
    lblExt: str

    @abstractmethod
    def __init__(self, root: str, *_, **__):
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
        """
        Implement this method to tell whether the input root directory
        is this dataset, which can then allows others to instantiate
        correct Dataset instance from method: `Dataset.find_dataset(root)`.
        """
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

    def __init__(self, root: str, *_, **__):
        super(VOC, self).__init__(root, *_, **__)

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
            return None

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
        return None


class KITTI(VOC, dtype=DatasetType.KITTI):
    imgDirName = "images"
    lblDirName = "labels"
    lblExt = ".txt"

    def __init__(self, root: str, *_, imgDir: str = "images", lblDir: str = "labels", **__):
        super(KITTI, self).__init__(root, *_, **__)
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
        else:
            return None


class COCO(Dataset, dtype=DatasetType.COCO):
    lblDirName = "annotations"
    imgDirName = "{split}{year}"

    SPLITS = ["train", "val", "test"]
    YEARS = [2017]

    def __init__(self, root: str, *_, split: str = "train", year: int = 2017, **__):
        super(COCO, self).__init__(root, *_, **__)
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


class Brainwash(Dataset, dtype="brainwash"):
    """
    This package contains the "Brainwash" dataset. The dataset consists of images capturing the everyday
    life of a busy downtown cafe and is split into the following subsets:

    training set: 10769 with 81975 annotated people
    validation set: 500 images with 3318 annotated people
    test set: 500 images with 5007 annotated people

    Bounding box annotations are provided in a simple text file format. Each line in the file contains
    image name followed by the list of annotation rectangles in the [xmin, ymin, xmax, ymax] format.

    We refer to the following arXiv submission for details on the dataset and the evaluation
    procedure:

    http://arxiv.org/abs/1506.04878
    """
    SPLITS = ["train", "test", "val"]
    imgDirName = ""
    lblDirName = ""

    def __init__(self, root: str, *_, split: str = "train", **__):
        super(Brainwash, self).__init__(root, *_, **__)
        self.lblPath = check_path(os.path.join(self.root, f"brainwash_{split}.idl"))

    @classmethod
    def check_is_correct_path(cls, root: str) -> bool:
        try:
            # Check image paths
            for date in ("10_27", "11_13", "11_24"):
                check_path(os.path.join(root, f"brainwash_{date}_2014_images"))
            # Check label paths
            for split in cls.SPLITS:
                check_path(os.path.join(root, f"brainwash_{split}.idl"))
        except FileNotFoundError:
            return False
        return True

    def load(self, clsNames: List[str] = None, nImgs: Union[int, float] = None):
        if clsNames is not None:
            sys.stderr.write("warn: brainwash dataset only has one class: head")
        if isinstance(nImgs, float):
            if nImgs <= 0.0 or nImgs > 1.0:
                msg = "nImgs must be float between (0.0, 1.0]"
                raise RuntimeError(msg)

        with open(self.lblPath, 'r') as f:
            lines = f.readlines()

        if nImgs is not None:
            if isinstance(nImgs, float):
                nImgs = int(nImgs * len(lines))
            if isinstance(nImgs, int):
                indices = random.sample(list(range(len(lines))), k=min(nImgs, len(lines)))
                lines = [lines[i] for i in indices]
            else:
                msg = f"nImgs must be an integer or float, got {nImgs}"
                raise RuntimeError(msg)

        labels = []
        ns = 0
        for line in lines:
            label, imgName = self.load_label(line)
            if label is not None:
                labels.append(label)
            else:
                ns += 1
                if imgName:
                    sys.stderr.write(f"\r[Skip :{ns:3d}]: {imgName}")
                else:
                    sys.stderr.write(f"\r[Skip :{ns:3d}]: {line}")
        self.labels = labels

    def load_label(self, line: str):

        # Case 1: line has no target: '"brainwash_10_27_2014_images/00158000_640x480.png";\n'
        lineInfo = line.strip().split(':')
        if len(lineInfo) == 1:
            imgFname = lineInfo[0][1:-2]
            return None, imgFname
        else:
            imgFname, objects = line.strip().split(':')
            imgFname = imgFname[1:-1]
        try:
            check_path(os.path.join(self.root, imgFname))
        except FileNotFoundError:
            return None, imgFname

        boxes = []
        try:
            # Case 2: image has only one label
            coordinates = ast.literal_eval(objects.strip()[:-1])
            # Only one box, line is (xmin, ymin, xmax, ymax);
            if not isinstance(coordinates[0], tuple):
                xmin, ymin, xmax, ymax = coordinates
                boxes.append(Box(xmin, ymin, xmax, ymax, "head"))

            # Case 3: image has multiple labels
            else:
                for coord in coordinates:
                    xmin, ymin, xmax, ymax = coord
                    boxes.append(Box(xmin, ymin, xmax, ymax, "head"))
            if boxes:
                return ImageLabel(imgFname, boxes, width=640, height=480, depth=3), imgFname
        except (ValueError, SyntaxError, TypeError) as e:
            sys.stderr.write(f"Current line: {line}\n")
        return None, imgFname


class HT21(Dataset, dtype="ht21"):
    """
    Head Tracking 21 dataset:  https://motchallenge.net/data/Head_Tracking_21/

    Annotation description:  https://arxiv.org/pdf/1906.04567.pdf

    Each line represents one object instance and contains 9 values:
    - The first number indicates in which frame the object appears
    - The second number identifies that object as belonging to a trajectory
      by assigning a unique ID (set to −1 in a detection file, as no ID is
      assigned yet). Each object can be assigned to only one trajectory.
    - The next four numbers indicate the position of the bounding box of the
      pedestrian in 2D image coordinates. The position is indicated by the
      top-left corner as well as width and height of the bounding box.
    - This is followed by a single number, which in case of detections denotes
      their confidence score. The last two numbers for detection files are ignored
      (set to -1).
    An example of such a 2D detection file is:
        1, -1, 794.2, 47.5, 71.2, 174.8, 67.5, -1, -1
        1, -1, 164.1, 19.6, 66.5, 163.2, 29.4, -1, -1
        1, -1, 875.4, 39.9, 25.3, 145.0, 19.6, -1, -1
        2, -1, 781.7, 25.1, 69.2, 170.2, 58.1, -1, -1

    Position    Name                    Description
    0           Frame number            Indicate at which frame the object is present
    1           Identity number         Each pedestrian trajectory is identified by a unique ID (−1 for detections)
    2           Bounding box left       Coordinate of the top-left corner of the pedestrian bounding box
    3           Bounding box top        Coordinate of the top-left corner of the pedestrian bounding box
    4           Bounding box width      Width in pixels of the pedestrian bounding box
    5           Bounding box height     Height in pixels of the pedestrian bounding box
    6           Confidence score        DET: Indicates how confident the detector is that this instance is a pedestrian.
                                        GT: It acts as a flag whether the entry is to be considered (1) or ignored (0).
    7           Class                   GT: Indicates the type of object annotated
    8           Visibility              GT: Visibility ratio, a number between 0 and 1 that says how much of that
                                            object is visible. Can be due to occlusion and due to image border cropping.
    NOTE:
        This class loads labels from `det.txt` file.
    """
    SPLITS = ["train", "test"]
    imgDirName = ""
    lblDirName = ""

    def __init__(self, root: str, *_, split: str = "train", conf: float = 60.0, **__):
        assert split in self.SPLITS, f"split must be one of {self.SPLITS}"
        super(HT21, self).__init__(os.path.join(root, split), *_, **__)
        self.conf = float(conf)
        self.imgExt = ".jpg"

    @classmethod
    def check_is_correct_path(cls, root: str) -> bool:
        try:
            root = check_path(root)
            for split in cls.SPLITS:
                splitRoot = check_path(os.path.join(root, split))
                partitions = glob.glob(os.path.join(splitRoot, cls.__name__ + '*'))
                if not partitions:
                    return False
                for part in partitions:
                    for subdir in ("det", "img1"):
                        check_path(os.path.join(part, subdir))
        except FileNotFoundError:
            return False
        return True

    def load(self, clsNames: List[str] = None, nImgs: Union[int, float] = None):
        partitions = glob.glob(os.path.join(self.root, type(self).__name__ + '*'), )
        if not partitions:
            return
        try:
            import pandas as pd
        except ImportError as e:
            sys.stderr.write("error: %s\n" % e)
            sys.exit(-1)

        labels = []
        workers = min(os.cpu_count(), len(partitions))
        futures = []
        with ProcessPoolExecutor(workers) as exe:
            for part in partitions:
                try:
                    lblFile = check_path(os.path.join(part, "det/det.txt"))
                except FileNotFoundError:
                    sys.stderr.write("warn: skip partition: %s\n" % lblFile)
                    continue

                # Check seqinfo.txt
                info = {}
                ini_conf = configparser.ConfigParser()
                if ini_conf.read(os.path.join(part, "seqinfo.ini")):
                    try:
                        info.update(width=int(ini_conf["Sequence"]["imWidth"]))
                        info.update(height=int(ini_conf["Sequence"]["imHeight"]))
                        info.update(imgExt=ini_conf["Sequence"]["imExt"])
                    except KeyError:
                        pass
                info["pName"] = os.path.basename(part)
                futures.append(exe.submit(self._load_partition_labels, lblFile, info))

            progress = tqdm(as_completed(futures), total=len(futures), desc="Load partitions")
            for future in progress:
                partLabels = future.result()
                labels.extend(partLabels)
        self.labels = labels

    def _load_partition_labels(self, lblTxt: str, imgInfo: dict):
        import pandas as pd

        df = pd.read_csv(lblTxt, header=None)
        numFrames = df[0].max()
        width = imgInfo.pop("width", -1)
        height = imgInfo.pop("height", -1)
        imgExt = imgInfo.pop("imgExt", ".jpg")
        imgNameFMT = f"{imgInfo['pName']}/img1/{{fIdx:06d}}{imgExt}"

        partLabels = []
        for fIdx in range(1, numFrames + 1):
            frameInfo = df.loc[df[0] == fIdx, :]
            frameInfo = frameInfo.loc[frameInfo[6] > self.conf, :]  # Filter by confidence
            boxes = []
            for i, rowInfo in frameInfo.iterrows():
                left, top, w, h = rowInfo[2:6]
                left -= 1   # Coordinate starts at 1
                top -= 1    # Coordinate starts at 1
                boxes.append(Box(left, top, left + w, top + h, "head"))
            if not boxes:
                continue
            imgName = imgNameFMT.format(fIdx=fIdx)
            imgLbl = ImageLabel(imgName, boxes, width, height)
            partLabels.append(imgLbl)
        return partLabels

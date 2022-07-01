import os
import sys
import inspect
import multiprocessing as mp
import xml.etree.ElementTree as ET
import xml.dom.minidom as MD
import cv2
import numpy as np

from dataclasses import dataclass
from abc import ABC, abstractmethod


from typing import List, Dict, Any, Union, Optional

_reg_ds = {}


def get_dataset(name: str, *args, **kwargs):
    assert name in _reg_ds, f"Unknown dataset: {name}"
    return _reg_ds[name](*args, **kwargs)


def register(klass):
    global _reg_ds
    assert inspect.isclass(klass), "Register classes only"
    _reg_ds[klass.__name__.lower()] = klass
    return klass


def xml_to_dict(xml_file: str) -> Dict[str, Any]:
    """
    Recursively parses xml contents to python dict.

    NOTE: `object` is the only tag that can appear multiple times.
    """
    assert os.path.exists(xml_file), f"File does not exist: {xml_file}"
    return _xml_to_dict_helper(ET.parse(xml_file).getroot())


def _xml_to_dict_helper(node: ET.Element) -> Dict[str, Any]:
    """ Helper function to parse xml_to_dict."""
    if len(node) == 0:
        return {node.tag: node.text}
    node_dict = {}
    for child in node:
        child_dict = _xml_to_dict_helper(child)
        if child.tag != "object":
            node_dict[child.tag] = child_dict[child.tag]
        else:
            node_dict.setdefault(child.tag, []).append(child_dict[child.tag])
    return {node.tag: node_dict}


def dict_to_xml(xml_dict: Dict[str, Any]) -> ET.Element:
    """Recursively parses a xml_dict to xml element tree."""
    assert isinstance(xml_dict, dict), f"Wrong type: type(xml_dict)={type(xml_dict)}"
    if "annotation" in xml_dict:
        xml_dict = xml_dict["annotation"]
    return _dict_to_xml_helper("annotation", xml_dict)


def _dict_to_xml_helper(tag: str, node: Union[str, Dict, List]) -> Union[ET.Element, List[ET.Element]]:
    """Helper function to parse xml_dict."""
    if isinstance(node, (str, int, float, type(None))):  # Elements with primal type are leaf nodes
        child = ET.Element(tag)
        child.text = str(node)
        return child
    if isinstance(node, dict):
        child = ET.Element(tag)
        for node_tag, sub_node in node.items():
            if node_tag != "object":
                child.append(_dict_to_xml_helper(node_tag, sub_node))
            else:
                for sub_child in _dict_to_xml_helper(node_tag, sub_node):
                    child.append(sub_child)
        return child
    if isinstance(node, list):
        children = []
        for sub_node in node:
            sub_child = _dict_to_xml_helper(tag, sub_node)
            children.append(sub_child)
        return children
    raise ValueError(f"[Error] Undefined node:{node} type(node)=={type(node)}")


def write_xml(file: str, node: Union[ET.ElementTree, ET.Element]):
    """Write xml_tree to file."""
    if isinstance(node, ET.Element):
        xml_str = pretty_xml(node)
        with open(file, 'w') as f:
            f.write(xml_str)
    elif isinstance(node, ET.ElementTree):
        with open(file, 'wb') as f:
            node.write(f)
    else:
        raise ValueError(f"Unknown argument `node`: type(node)==f{type(node)}!")


def pretty_xml(node: ET.Element) -> str:
    """Return a pretty indented xml string given a ET.Element node."""
    assert isinstance(node, ET.Element), f"Wrong type: type(node)=={type(node)}"
    raw_xml_str = ET.tostring(node)
    pretty_str = MD.parseString(raw_xml_str).toprettyxml()
    return pretty_str


@dataclass
class BBox:
    __slots__ = "xmin", "ymin", "xmax", "ymax", "name"

    xmin: int
    ymin: int
    xmax: int
    ymax: int
    name: str

    def scaling(self, factor: float):
        self.xmin = int(self.xmin * factor)
        self.ymin = int(self.ymin * factor)
        self.xmax = int(self.xmax * factor)
        self.ymax = int(self.ymax * factor)

    @property
    def coord(self):
        return self.xmin, self.ymin, self.xmax, self.ymax

    @classmethod
    def from_xywh(cls, left: int, top: int, width: int, height: int, cls_idx: str) -> "BBox":
        return cls(left, top, left + width, top + height, cls_idx)

    def to_voc_obj(self, truncated: bool = False, difficult: bool = False):
        d = {}
        d.update(name=self.name)
        d.update(bndbox=dict(zip(self.__slots__[:4], self.coord)))
        d.update(truncated=int(truncated))
        d.update(difficult=int(difficult))
        return d


class Dataset(ABC):

    def __init__(self, num_workers: int = 1):
        self._n_workers = num_workers
        self.img_dir: str = ""
        self.lbl_dir: str = ""

    @abstractmethod
    def create_directory(self, root_dir: str, ds_name: str):
        raise NotImplementedError

    @abstractmethod
    def save_label(self, ann_dict: Dict[str, Any], merge: bool = False):
        """
        anno_dict (Dict[str, Any]):
            Python dict containing following information:
                - folder (Optional[str]): Folder name of dataset
                - filename (str): image file name
                - size (Dict[str, int]):
                    - width  (int): Width of image
                    - height (int): Height of image
                    - depth  (int): Number channels
                - object (List[BBox]):
        """
        raise NotImplementedError

    def save_image(self, ann_dict: Dict[str, Any], image: np.ndarray):
        filename = ann_dict["filename"]
        filepath = os.path.join(self.img_dir, filename)
        cv2.imwrite(filepath, image)

    def save_all(self, ann_dict: Dict[str, Any], image: np.ndarray, merge: bool = False):
        self.save_label(ann_dict, merge)
        self.save_image(ann_dict, image)


@register
class KITTI(Dataset):
    LABEL_FMT = "{} 0 0 0 {} {} {} {} 0 0 0 0 0 0 0\n"

    def create_directory(self, root_dir: str, ds_name: str):
        self.img_dir = os.path.join(root_dir, ds_name, "images")
        self.lbl_dir = os.path.join(root_dir, ds_name, "labels")
        os.makedirs(self.img_dir, exist_ok=True)
        os.makedirs(self.lbl_dir, exist_ok=True)

    def save_label(self, ann_dict: Dict[str, Any], merge: bool = False):
        mode = 'a' if merge else 'w'
        filename = ann_dict["filename"].split('.')[0] + '.txt'
        filepath = os.path.join(self.lbl_dir, filename)
        with open(filepath, mode) as f:
            for bbox in ann_dict["object"]:
                line = self.LABEL_FMT.format(bbox.name, *bbox.coord)
                f.write(line)


@register
class VOC(Dataset):

    def create_directory(self, root_dir: str, ds_name: str):
        self.img_dir = os.path.join(root_dir, ds_name, "JPEGImages")
        self.lbl_dir = os.path.join(root_dir, ds_name, "Annotations")
        os.makedirs(self.img_dir, exist_ok=True)
        os.makedirs(self.lbl_dir, exist_ok=True)

    def save_label(self, ann_dict: Dict[str, Any], merge: bool = False):
        filename = ann_dict["filename"].split('.')[0] + '.xml'
        filepath = os.path.join(self.lbl_dir, filename)

        obj_dict_list = [obj.to_voc_obj() for obj in ann_dict["object"]]
        if merge and os.path.exists(filepath):
            old_ann_dict = xml_to_dict(filepath)["annotation"]
            old_ann_dict["object"].extent(obj_dict_list)
            write_xml(filepath, dict_to_xml({"annotation": old_ann_dict}))
        else:
            ann_dict["object"] = obj_dict_list
            write_xml(filepath, dict_to_xml({"annotation": ann_dict}))


import os
import sys
import cv2
import colorsys
import random
import numpy as np
import matplotlib.pyplot as plt

from collections.abc import Callable



sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from DSCollection.src._DSCollection import xml_to_dict




def list_files(directory: str, abspath: bool = True, recursive: bool = True, filter_pred: Callable = None):
    """
    List all files under directory.
    """
    assert os.path.exists(directory), f"Directory does not exist: {directory}"

    depth = 0
    for root, _, files in os.walk(directory):
        depth += 1
        if not recursive and depth > 1:
            return
        for file in files:
            if isinstance(filter_pred, Callable) and not filter_pred(file):
                continue
            if abspath:
                yield os.path.abspath(os.path.join(root, file))
            else:
                yield file


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


def change_lightness_color(color, factor: float):
    """
    Darken or lighten a RGB color according to factor.
    @Params:
    -------
    color (Tuple[R,G,B]):
        a tuple of RGB color.
    factor (float):
        If factor < 1, then the lightened color is returned, otherwise
        the darkened color is returned.
    """
    h, l, s = colorsys.rgb_to_hls(*color)
    return colorsys.hls_to_rgb(h, 1 - factor * (1 - l), s)


def _draw_bbox_on_image(image, bbox, color, map_id_to_cls=None, fix_color=False):
    """
    Draw one bbox on image.
    """
    cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
    if len(bbox) > 4:
        cls_id = int(bbox[4])
        cls_name = ""
        cls_conf = ""
        if map_id_to_cls is not None:
            cls_name = map_id_to_cls[cls_id]
        if len(bbox) > 5:
            cls_conf = f", {bbox[5]:.2f}"
        tag = cls_name + cls_conf
        if not fix_color: color = change_lightness_color(color, 2.)
        cv2.putText(image, tag, (int(bbox[0]), int(bbox[1]) + 15), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 1)


def visualize_detection(image, bbox=None, pred=None, map_id_to_cls=None, fix_color=False):
    """
       Visualize bbox on the original image.
       @Params
       -------
       image (PIL Image):
           Original image to visualize.
       map_id_to_cls (dict):
           Mapping from class id to class name.
       bbox (tensor)
           GT bbox, tensor of shape [N, 5], where 5 indicates
               (x_tl, y_tl, x_br, y_br, cls_id)
       pred (tensor)
           Predicted bbox, tensor of shape [N, 6], where 6 indicates
               (x_tl, y_tl, x_br, y_br, cls_id, cls_conf_score)
       """
    image = np.array(image).astype(np.uint8)
    n = 0
    if bbox is not None:
        n = max(n, bbox.shape[0])
    if pred is not None:
        n = max(n, pred.shape[0])
    cmap = get_cmap(n)

    if bbox is not None:
        # draw GT bbox
        for i in range(bbox.shape[0]):
            one_bbox = bbox[i]

            if fix_color:
                color = (255, 0, 0)  # RGB
            else:
                color = tuple(map(lambda x: int(x * 255.), cmap(i)[:3]))
            _draw_bbox_on_image(image, one_bbox, color, map_id_to_cls, fix_color=fix_color)

    if pred is not None:
        # draw predicted bbox
        for i in range(pred.shape[0]):
            one_bbox = pred[i]
            if fix_color:
                color = (0, 255, 0)  # RGB
            else:
                color = tuple(map(lambda x: int(x * 255.), cmap(i)[:3]))
                color = change_lightness_color(color, 0.5)
            _draw_bbox_on_image(image, one_bbox, color, map_id_to_cls, fix_color=fix_color)

    plt.imshow(image)
    plt.axis("off")
    plt.show()
    return image


def get_bboxes_from_xml(xml_path: str, map_cls2id: dict):
    """Extract bounding boxes from a xml file."""
    # Get labels from xml
    bboxes = []
    annotation = xml_to_dict(xml_path)["annotation"]
    objects = annotation.get("object", [])
    for obj in objects:
        xmin = float(obj["bndbox"]["xmin"])
        ymin = float(obj["bndbox"]["ymin"])
        xmax = float(obj["bndbox"]["xmax"])
        ymax = float(obj["bndbox"]["ymax"])
        try:
            clsid = float(map_cls2id[obj["name"]])
        except KeyError as e:
            msg = str(e)
            msg += f", map_cls2id keys: `{list(map_cls2id.keys())}`, xml-path: {xml_path}"
            raise KeyError(msg)

        bboxes.append([xmin, ymin, xmax, ymax, clsid])
    bboxes = np.array(bboxes, dtype=np.float32)
    return bboxes


def visualize_image_with_xml(image_path: str, xml_path: str, map_cls2id: dict, no_label=False):
    """Visualize image and all bboxes."""
    assert os.path.exists(image_path), f"Wrong path: {image_path}"
    assert os.path.exists(xml_path), f"Wrong path: {xml_path}"
    bboxes = get_bboxes_from_xml(xml_path, map_cls2id)
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    map_id2cls = None if no_label else {v: k for k, v in map_cls2id.items()}
    visualize_detection(img, bboxes, map_id_to_cls=map_id2cls)


def visualize_voc(directory: str, map_cls2id: dict, vis_num: int = 10, no_label=False):
    """Visualize VOC-like dataset."""
    assert os.path.exists(directory), f"Wrong path: {directory}"
    anns_dir = os.path.join(directory, "Annotations")
    imgs_dir = os.path.join(directory, "JPEGImages")

    xml_fnames = list(list_files(anns_dir, abspath=False, recursive=False, filter_pred=lambda p: p.endswith(".xml")))
    xml_fnames = random.sample(xml_fnames, vis_num)
    for xml_fname in xml_fnames:
        xml_file = os.path.join(anns_dir, xml_fname)
        img_file = os.path.join(imgs_dir, xml_fname.replace(".xml", ".jpg"))
        visualize_image_with_xml(img_file, xml_file, map_cls2id, no_label)


if __name__ == '__main__':
    cls2id = {"head": 0, "person": 1}
    root = "/media/sparkai/DATA2/Datasets/Head/SCUT_HEAD_Part_A"

    visualize_voc(root, cls2id, 20, True)
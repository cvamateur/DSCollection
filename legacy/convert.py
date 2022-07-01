import os
import sys
import shutil
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from voc import xml_to_dict


def voc_to_kitti(input_dir: str, output_dir: str):

    img_dir = os.path.join(input_dir, "JPEGImages")
    lbl_dir = os.path.join(input_dir, "Annotations")
    assert os.path.exists(img_dir), f"Wrong path: {img_dir}"
    assert os.path.exists(lbl_dir), f"Wrong path: {lbl_dir}"
    dst_img_dir = os.path.join(output_dir, "images")
    dst_lbl_dir = os.path.join(output_dir, "labels")
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_lbl_dir, exist_ok=True)

    def get_name(lbl_path: str):
        return lbl_path.split('.')[0]

    KITTI_FMT = "{} 0 0 0 {} {} {} {} 0 0 0 0 0 0 0\n"

    lbl_list = os.listdir(lbl_dir)
    img_list = [get_name(p) + ".jpg" for p in lbl_list]
    lbl_list = [os.path.join(lbl_dir, p) for p in lbl_list]
    img_list = [os.path.join(img_dir, p) for p in img_list]

    for img_path, lbl_path in tqdm(zip(img_list, lbl_list), total=len(img_list)):
        ann_dict = xml_to_dict(lbl_path)["annotation"]
        if "object" not in ann_dict:
            continue

        xml_name = os.path.basename(lbl_path)
        dst_lbl_name = get_name(xml_name) + ".txt"
        with open(os.path.join(dst_lbl_dir, dst_lbl_name), "w") as f:
            for obj in ann_dict["object"]:
                label_name = obj["name"]
                if not int(obj["truncated"]) and not int(obj["difficult"]):
                    bbox = obj["bndbox"]
                    xmin = bbox["xmin"]
                    ymin = bbox["ymin"]
                    xmax = bbox["xmax"]
                    ymax = bbox["ymax"]
                    line = KITTI_FMT.format(label_name, xmin, ymin, xmax, ymax)
                    f.write(line)

        shutil.copy2(img_path, os.path.join(dst_img_dir, get_name(xml_name) + ".jpg"))


if __name__ == '__main__':
    voc_to_kitti(r'/media/sparkai/DATA2/Datasets/Head/SCUT_HEAD_Part_A', "/media/sparkai/DATA2/Datasets/Head/SCUT_HEAD_Part_A_Kitti")
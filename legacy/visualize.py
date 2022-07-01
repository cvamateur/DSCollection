# @Time    : 2022/4/22 下午3:49
# @Author  : Boyang
# @Site    : 
# @File    : visualize.py
# @Software: PyCharm
import os
import random
import re
import shutil
import time
from collections import defaultdict
from glob import glob
from os.path import join as path_join
from typing import Tuple, List, Union

import cv2
import matplotlib.pyplot as plt


def visualize_wider_face(image_dir, label_path):
    """
    可视化wider face数据集
    :return:
    """
    # label_content = None
    with open(label_path) as f:
        label_content = f.readlines()

    i = 0
    while i < len(label_content):
        line = label_content[i].strip()
        if line.endswith('.jpg'):
            image_path = os.path.join(image_dir, line)
            img = cv2.imread(image_path)
            i += 1
            num_gt = int(label_content[i])
            i += 1
            for k in range(num_gt):
                label = label_content[i].strip()
                label_parts = label.split(" ")
                # filter
                blur, expression, illumination, invalid, occlusion, pose = map(int, label_parts[4:])
                if blur == 0:
                    color = (255, 0, 0)
                elif blur == 1:
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 255)

                x1 = int(label_parts[0])
                y1 = int(label_parts[1])
                x2 = int(label_parts[2]) + x1
                y2 = int(label_parts[3]) + y1
                cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=1)
                cv2.putText(img, label_parts[0], (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color)
                i += 1

            plt.imshow(img[..., ::-1])
            plt.show()
            if i >= 1000:
                break


def visualize_kitti(dataset_dir, image_extension: Union[Tuple[str], str] = None, shuffle=False, num: int = 50):
    """
    可视化kitti数据集
    """
    if image_extension is None:
        image_extension = ('png',)
    if isinstance(image_extension, str):
        image_extension = (image_extension,)
    # fig = plt.figure(figsize=(30, 30))
    images_dir = path_join(dataset_dir, 'images')
    label_dir = path_join(dataset_dir, 'labels')
    images = [filename for filename in os.listdir(images_dir) if filename.endswith(image_extension)]
    if shuffle:
        random.shuffle(images)
    for i, img_name in enumerate(images[:num]):
        img_path = path_join(images_dir, img_name)
        label_filename = os.path.splitext(img_name)[0] + '.txt'
        label_path = path_join(label_dir, label_filename)
        img = cv2.imread(img_path)
        with open(label_path) as f:
            count = 0
            for line in f.readlines():
                lines = line.split(' ')
                label, x1, y1, x2, y2 = lines[0], float(lines[4]), float(lines[5]), float(lines[6]), float(lines[7])
                color = (0, 255, 0) if label == 'mask' else (0, 0, 255)
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=2)
                count += 1
        # ax = fig.add_subplot(3, 2, i + 1)
        # ax.set_title(f"{img.shape[:2]} count object")

        plt.imshow(img[..., ::-1])

        plt.show()


def visualize_dataset_bbox_size_distribution(dataset):
    label_dir = path_join(dataset, 'labels')
    bbox_size = []
    labels = [filename for filename in os.listdir(label_dir) if filename.endswith('txt')]
    for name in labels:
        label_path = path_join(label_dir, name)
        with open(label_path) as f:
            for line in f.readlines():
                x1, y1, x2, y2 = map(float, line.split(' ')[4:8])
                bbox_size.append((x2 - x1, y2 - y1))

    width_list = list(map(lambda x: x[0], bbox_size))
    height_list = list(map(lambda x: x[1], bbox_size))
    plt.scatter(width_list, height_list, 0.4)
    plt.xlabel('Width')
    plt.ylabel('Height')
    print(len(width_list))
    plt.show()


def play_voc_dataset(dataset_path: str, classes_name: List[str] = None):
    """
    播放voc格式数据集
    :param dataset_path:
    :param classes_name:
    :return:
    """
    if classes_name is None:
        classes_name = ['default']
    sets_dir = os.path.join(dataset_path, 'ImageSets/Main')
    anns_dir = os.path.join(dataset_path, 'Annotations')
    imgs_dir = os.path.join(dataset_path, 'JPEGImages')
    for class_name in classes_name:
        class_name += '.txt'
        set_path = os.path.join(sets_dir, class_name)
        with open(set_path) as f:
            for i, line in enumerate(f.readlines()):
                img = cv2.imread(os.path.join(imgs_dir, line.strip()))
                cv2.putText(img, f"{i}", (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0,))
                cv2.imshow(class_name, img)
                print(i)
                key = cv2.waitKey(1)
                time.sleep(0.1)
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    exit(-1)

    cv2.destroyAllWindows()


def label_count(label_dir):
    """
    查看kitti数据集的标签分布情况
    :param label_dir:
    :return:
    """
    label_dict = defaultdict(int)
    for filename in os.listdir(label_dir):
        path = os.path.join(label_dir, filename)
        if not os.path.isfile(path) or not filename.endswith('.txt'):
            continue
        # only support kitti
        with open(path) as f:
            for line in f.readlines():
                label = line.strip().split(' ')[0]
                if label:
                    label_dict[label] += 1
    plt.bar(label_dict.keys(), label_dict.values())
    plt.show()


def __pack_up_some_video():
    """
    将以前的鱼眼视频存储格式转换
    :return:
    """
    pattern = re.compile('\d+-\d+-\d+\.mp4')
    root_dir = '/data/Datasets/Fisheye/video'
    out_dir = '/data/Datasets/Fisheye/video/202112'
    os.makedirs(out_dir, exist_ok=True)
    for dir_path in glob(os.path.join(root_dir, '2021-12-*')):
        for video_file in glob(dir_path + '/*/*.mp4', recursive=True):
            _p = os.path.dirname(video_file)
            _p = os.path.dirname(_p)
            date = os.path.basename(_p)
            date = date.split('-')
            date = ''.join(date)
            filename = os.path.split(video_file)[-1]
            if pattern.match(filename):
                filename = filename.split('-')
                filename = ''.join(filename)
            filename = date + filename
            dst = os.path.join(out_dir, filename)
            # shutil.move(video_file, dst)
            if os.path.exists(dst):
                continue
            shutil.copy2(video_file, dst)


if __name__ == '__main__':
    # play_voc_dataset('/data/Datasets/Fisheye/VOCDataset/20220210/saved_dataset', ['right'])
    # visualize_dataset_bbox_size_distribution('/home/sparkai/PycharmProjects/TaoFacemask/project/facenet/data/training')
    # visualize_kitti('/home/sparkai/PycharmProjects/TaoFacemask/project/facenet/data/inference', num=100)
    # visualize_kitti('/home/sparkai/PycharmProjects/ChinamobileSDK/data/GeneratedDateset')
    visualize_kitti('/home/sparkai/PycharmProjects/ChinamobileSDK/DSCollection/SCUT_HEAD_Part_A-Kitti', "jpg")

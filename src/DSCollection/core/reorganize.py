import os
import math
import shutil
import sys
import glob

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Union
from tqdm import tqdm

from ..utils.common import check_path, is_image
from .dataset import Dataset


_SUPPORT_DS = ("VOC", "KITTI")


class DatasetSplitter:
    """
    Split dataset into multiple partitions. The class uses shutil.move() by default,
    therefore all images and labels from original dataset will be moved to split datasets,
    unless pass `keep=True` explicitly.

    @Arguments:
    -----------
    Either one of the three arguments can be given:
    nSplits (int):
        Split dataset into `nSplits` partitions.

    nEach (int):
        Split dataset equally such that each part contains approximate `nEach` samples.

    nFractions (str | List[float]):
        Percentages of each partition, could be a string containing multiple floats that
        separated by ';' or a list of float number. All floats must sum up to 1.0.

    partNames (str | List[str])
        Names that will suffix to partitions directory name.
    """

    def __init__(self, nSplits: int = None,
                 nEach: int = None,
                 nFractions: Union[str, List[float]] = None,
                 partNames: Union[str, List[str]] = None):
        if not nSplits and not nEach and not nFractions:
            msg = "Split methods are all None"
            raise RuntimeError(msg)
        self._ns = nSplits
        self._ne = nEach
        if nFractions is not None:
            try:
                nFractions = [float(f) for f in nFractions]
            except ValueError:
                msg = "nFractions must be real number seperated by ';' or a list of real numbers"
                raise RuntimeError(msg)
            if sum(nFractions) != 1.0:
                msg = f"nFractions must sum up to 1.0, got: {nFractions}"
                raise RuntimeError(msg)
        self._nf = nFractions
        self.partNames = partNames

    def run(self, root: str, outDir: str = None,
            imgDirName: str = None, lblDirName: str = None,
            contiguous: bool = True, drop: bool = False):

        # Check dataset type, must be VOC or KITTI
        DataKlass = Dataset.find_dataset(root)
        if DataKlass is None:
            msg = f"Not a valid dataset {_SUPPORT_DS} path."
            raise RuntimeError(msg)
        if DataKlass.__name__ not in _SUPPORT_DS:
            msg = "Only VOC and KITTI dataset can be split."
            raise RuntimeError(msg)

        # Check output path is given, otherwise
        if imgDirName is None: imgDirName = DataKlass.imgDirName
        if lblDirName is None: lblDirName = DataKlass.lblDirName
        imgRoot = check_path(os.path.join(root, imgDirName))
        lblRoot = check_path(os.path.join(root, lblDirName))
        imgFnames = [fn for fn in os.listdir(imgRoot)]
        lblFnames = [fn.split('.')[0] + DataKlass.lblExt for fn in imgFnames]
        if not len(imgFnames):
            msg = "Not images found in dataset"
            return RuntimeError(msg)
        total = len(imgFnames)

        # calculate numbers per each part
        if self._ns:
            partitions = self._get_partition_by_ns(self._ns, total)
        elif self._ne:
            partitions = self._get_partition_by_ne(self._ne, total)
        else:
            partitions = self._get_partition_by_nf(self._nf, total)
        if self.partNames is not None and len(self.partNames) != len(partitions):
            msg = f"Number of partition names ({len(self.partNames)}) does not match " \
                  f"with number of partitions ({len(partitions)})"
            raise RuntimeError(msg)

        # Check output path
        dsName = os.path.basename(root)
        if outDir is not None:
            outDir = check_path(outDir)
        else:
            outDir = check_path(os.path.dirname(root))

        # Move files
        nameFmt = "{{:0{}d}}.{{}}".format(len(str(max(partitions))))
        workers = min(os.cpu_count()-1, len(partitions))
        futures = []
        with ThreadPoolExecutor(max_workers=workers) as exe:
            startIdx = 0
            for i, n in enumerate(partitions):
                if self.partNames is None:
                    pdName = f"{dsName}-part{i:02d}"
                else:
                    pdName = f"{dsName}-{self.partNames[i]}"
                pdRoot = DataKlass.create_structure(outDir, pdName)
                dstImgDir = os.path.join(pdRoot, DataKlass.imgDirName)
                dstLblDir = os.path.join(pdRoot, DataKlass.lblDirName)
                imgs = imgFnames[startIdx:startIdx+n]
                lbls = lblFnames[startIdx:startIdx+n]
                startIdx += n
                for j, (img, lbl) in enumerate(zip(imgs, lbls)):
                    srcImg = os.path.join(imgRoot, img)
                    srcLbl = os.path.join(lblRoot, lbl)
                    if contiguous:
                        dstImg = os.path.join(dstImgDir, nameFmt.format(j, img.split('.')[1]))
                        dstLbl = os.path.join(dstLblDir, nameFmt.format(j, lbl.split('.')[1]))
                    else:
                        dstImg = os.path.join(dstImgDir, img)
                        dstLbl = os.path.join(dstLblDir, lbl)
                    if not drop:
                        futures.append(exe.submit(shutil.copy2, srcImg, dstImg))
                        futures.append(exe.submit(shutil.copy2, srcLbl, dstLbl))
                    else:
                        futures.append(exe.submit(shutil.move, srcImg, dstImg))
                        futures.append(exe.submit(shutil.move, srcLbl, dstLbl))

            progress = tqdm(total=len(futures) // 2, desc="Split dataset")
            for i, future in enumerate(as_completed(futures)):
                future.result()
                if i % 2:
                    progress.update()

        if drop:
            try:
                os.remove(os.path.abspath(os.path.expanduser(root)))
            except PermissionError:
                sys.stdout.write("Permission denied, unable to remove: %s\n" % root)
            else:
                sys.stdout.write(f"Remove path: {root}\n")

    @staticmethod
    def _get_partition_by_ns(ns: int, total: int):
        parts = [total // ns for _ in range(ns)]
        if parts:
            parts[-1] += total % ns
        return parts

    @staticmethod
    def _get_partition_by_ne(ne: int, total: int):
        parts = []
        start = 0
        while start + ne < total:
            parts.append(ne)
            start += ne
        remain = total % ne
        if remain and parts and remain <= ne // 5:
            a = remain // len(parts)
            parts = [x + a for x in parts]
            parts[-1] += remain % len(parts)
        else:
            parts.append(remain)
        return parts

    @staticmethod
    def _get_partition_by_nf(nf: List[float], total: int):
        parts = [math.floor(f * total) for f in nf]
        if parts:
            parts[-1] += total - sum(parts)
        return parts


class DatasetCombiner:
    """
    Combine multiple datasets into a whole dataset. All datasets been combined must
    be of same type, either VOC or KITTI.
    """
    def __init__(self, inputs: List[str]):
        self.inputs = []
        self.DsKlass: Dataset = None
        for input_dir in inputs:
            input_dir = os.path.expanduser(input_dir)
            input_dir_list = glob.glob(input_dir)
            for dsRoot in input_dir_list:
                dsRoot = check_path(dsRoot)
                DataKlass = Dataset.find_dataset(dsRoot)
                if DataKlass is None:
                    msg = f"Not a valid dataset {_SUPPORT_DS} path."
                    raise RuntimeError(msg)
                if DataKlass.__name__ not in _SUPPORT_DS:
                    msg = "Only VOC and KITTI dataset can be split."
                    raise RuntimeError(msg)
                if self.DsKlass is None:
                    self.DsKlass = DataKlass
                elif self.DsKlass != DataKlass:
                    msg = "All datasets must be of same type"
                    raise RuntimeError(msg)
                self.inputs.append(dsRoot)

        self.inputs.sort()
        if not self.inputs:
            msg = "No input dataset to merge"
            raise RuntimeError(msg)

    def run(self, outDir: str, contiguous: bool = True, drop: bool = False):
        outDir = check_path(outDir, False)
        outRoot = self.DsKlass.create_structure(*os.path.split(outDir))
        dstImgDir = os.path.join(outRoot, self.DsKlass.imgDirName)
        dstLblDir = os.path.join(outRoot, self.DsKlass.lblDirName)

        imgNames_list, lblNames_list = [], []
        imgPaths_list, lblPaths_list = [], []
        for dsRoot in self.inputs:
            imgDir = os.path.join(dsRoot, self.DsKlass.imgDirName)
            lblDir = os.path.join(dsRoot, self.DsKlass.lblDirName)
            imgFnames = [name for name in os.listdir(imgDir) if is_image(name)]
            lblFnames = [name.split('.')[0] + self.DsKlass.lblExt for name in imgFnames]
            imgPaths = [os.path.join(imgDir, name) for name in imgFnames]
            lblPaths = [os.path.join(lblDir, name) for name in lblFnames]
            if not all(map(os.path.exists, lblPaths)):
                msg = f"Not all images or labels exists, dataset corrupted: {dsRoot}"
                raise RuntimeError(msg)
            imgNames_list.append(imgFnames)
            lblNames_list.append(lblFnames)
            imgPaths_list.append(imgPaths)
            lblPaths_list.append(lblPaths)

        total = sum(map(len, imgPaths_list))
        nameFmt = "{{:0{}d}}.{{}}".format(len(str(total)))
        count = 0
        futures = []
        with ThreadPoolExecutor(os.cpu_count() - 1) as exe:
            for i, (imgPaths, lblPaths) in enumerate(zip(imgPaths_list, lblPaths_list)):
                dsName = os.path.basename(self.inputs[i])
                for j, (srcImg, srcLbl) in enumerate(zip(imgPaths, lblPaths)):
                    imgName = imgNames_list[i][j]
                    lblName = lblNames_list[i][j]
                    if contiguous:
                        dstImg = os.path.join(dstImgDir, nameFmt.format(count, imgName.split('.')[1]))
                        dstLbl = os.path.join(dstLblDir, nameFmt.format(count, lblName.split('.')[1]))
                        count += 1
                    else:
                        dstImg = os.path.join(dstImgDir, f"{dsName}-{imgName}")
                        dstLbl = os.path.join(dstLblDir, f"{dsName}-{lblName}")
                    if not drop:
                        futures.append(exe.submit(shutil.copy2, srcImg, dstImg))
                        futures.append(exe.submit(shutil.copy2, srcLbl, dstLbl))
                    else:
                        futures.append(exe.submit(shutil.move, srcImg, dstImg))
                        futures.append(exe.submit(shutil.move, srcLbl, dstLbl))

            progress = tqdm(total=len(futures) // 2, desc="Combine datasets")
            for i, future in enumerate(as_completed(futures)):
                future.result()
                if i % 2:
                    progress.update()

        if drop:
            for inpDir in self.inputs:
                try:
                    shutil.rmtree(os.path.abspath(os.path.expanduser(inpDir)))
                except Exception:
                    sys.stdout.write("warn: unable to remove: %s\n" % inpDir)
                else:
                    sys.stdout.write(f"Remove path: {inpDir}\n")

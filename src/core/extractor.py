import shutil
import os.path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Union

from tqdm import tqdm

from src.utils.common import check_path
from .dataset import Dataset
from .convertor import Convertor, LabelInfo


class DataExtractor:
    """
    @Description:
    -------
    The DataExtractor class provide a general facility to extract all of or some of the
    classes from an input dataset to a new dataset. The format of the input or output
    dataset could be either { Pascal VOC, MS-COCO, KITTI }.

    @Arguments:
    -----------

    cvt (Optional[Convertor]):
        Convertor instance that implement `convert()` method, refer to Convertor for more details.
    """

    def __init__(self,  cvt: Convertor, imgExt: str = None):
        self.cvt = cvt
        self.imgExt = imgExt
        self.num_workers = os.cpu_count() - 1

    def extract(self, ds: Dataset,
                dstDir: str, dsName: str = None,
                clsNames: List[str] = None,
                nImgs: Union[int, float] = None,
                contiguous: bool = True):
        """
        @Arguments:
        -----------
        inputDataset (Dataset):
            Instance of Dataset to be extracted.

        clsNames (List[str]):
            If given, only extract classes with names in the list.

        contiguous (bool):
            Whether save images and labels as contiguous filenames.
        """
        if dsName is None:
            dstDir, dsName = os.path.split(dstDir)
            dstDir = check_path(dstDir)
        dstRoot = self.create_dataset(dstDir, dsName, self.cvt.imgDirName, self.cvt.lblDirName)
        ds.load(clsNames, nImgs)
        cvt_results = self.cvt.convert(ds)
        srcImgDir = os.path.join(ds.root, ds.imgDirName)
        dstImgDir = os.path.join(dstRoot, self.cvt.imgDirName)
        dstLblDir = os.path.join(dstRoot, self.cvt.lblDirName)

        info: LabelInfo
        nameFmt = "{{:0{}d}}{{}}".format(len(str(len(cvt_results))))
        futures = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            for info in cvt_results:
                srcImg = os.path.join(srcImgDir, info.imgName)
                if contiguous:
                    imgName, ext = info.imgName.split('.')
                    dstImg = os.path.join(dstImgDir, nameFmt.format(info.index, f".{ext}"))
                    dstLbl = os.path.join(dstLblDir, nameFmt.format(info.index, self.cvt.lblExt))
                else:
                    dstImg = os.path.join(dstImgDir, info.imgName)
                    dstLbl = os.path.join(dstLblDir, info.imgName.split('.')[0] + self.cvt.lblExt)
                future = executor.submit(self._save_to_dst, srcImg, dstImg, info.data, dstLbl)
                futures.append(future)

            for future in tqdm(as_completed(futures), total=len(futures), desc="Extract dataset"):
                future.result()

    @staticmethod
    def _save_to_dst(srcImg: str, dstImg: str, lblBytes: bytes, dstLbl: str):
        shutil.copy2(srcImg, dstImg)
        with open(dstLbl, 'wb') as f:
            f.write(lblBytes)

    @staticmethod
    def create_dataset(root: str, name: str, imgDir: str, lblDir: str):
        dstDir = check_path(root, True)
        root = os.path.join(dstDir, name)
        os.makedirs(os.path.join(root, imgDir), exist_ok=True)
        os.makedirs(os.path.join(root, lblDir), exist_ok=True)
        return root

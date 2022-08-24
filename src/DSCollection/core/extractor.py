import shutil
import os.path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Union

from tqdm import tqdm

from ..utils.common import check_path
from ..utils.imgutil import ImageUtil
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
        self.imgExt = imgExt if imgExt.startswith('.') else f".{imgExt}"
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
        dstRoot = self.cvt.create_dataset(dstDir, dsName)
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
                    dstImg = os.path.join(dstImgDir, nameFmt.format(info.index, self.imgExt))
                    dstLbl = os.path.join(dstLblDir, nameFmt.format(info.index, self.cvt.lblExt))
                else:
                    # make sure imgName is not a subdirectory structure
                    imgName, ext = info.imgName.replace('/', '-').split('.')
                    dstImg = os.path.join(dstImgDir, imgName + self.imgExt)
                    dstLbl = os.path.join(dstLblDir, imgName.split('.')[0] + self.cvt.lblExt)
                decoding = (ext != self.imgExt)
                future = executor.submit(self._save_to_dst, srcImg, dstImg, info.data, dstLbl, decoding, self.imgExt)
                futures.append(future)

            for future in tqdm(as_completed(futures), total=len(futures), desc="Extract dataset"):
                future.result()

    @staticmethod
    def _save_to_dst(srcImg: str, dstImg: str, lblBytes: bytes, dstLbl: str,
                     decoding: bool = False, ext: str = ".jpg"):
        with open(dstLbl, 'wb') as f:
            f.write(lblBytes)

        if decoding:
            with open(srcImg, "rb") as f:
                imgRaw = f.read()
            imgDec = ImageUtil.decode(imgRaw)
            imgEnc = ImageUtil.encode(imgDec, ext)
            with open(dstImg, "wb") as f:
                f.write(imgEnc)
        else:
            shutil.copy2(srcImg, dstImg)

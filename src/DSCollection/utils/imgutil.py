import cv2
import numpy as np

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Iterable, List, Union
from tqdm import tqdm


class ImageUtil:

    @classmethod
    def decode(cls, imgRaw: bytes) -> np.ndarray:
        image = np.asarray(bytearray(imgRaw), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
        return image

    @classmethod
    def encode(cls, img: np.ndarray, ext: str = ".png") -> Union[bytes, None]:
        imgRaw = cv2.imencode(ext, img)[1]
        imgData = np.array(imgRaw).tobytes()
        return imgData

    @classmethod
    def decode_many(cls, imgList: Iterable[bytes], num_workers: int = 1) -> List[np.ndarray]:
        futures = []
        with ProcessPoolExecutor(num_workers) as exe:
            for imgRaw in imgList:
                futures.append(exe.submit(cls.decode, imgRaw))

            result = []
            to_do = tqdm(as_completed(futures), total=len(futures), desc="Decode images")
            for future in to_do:
                result.append(future.result())
        return result

    @classmethod
    def encode_many(cls, imgs: Iterable[np.ndarray], ext: str = ".png",
                    num_workers: int = 1) -> List[Union[bytes, None]]:
        futures = []
        with ProcessPoolExecutor(num_workers) as exe:
            for img in imgs:
                futures.append(exe.submit(cls.encode, ext, img))

            to_do = tqdm(as_completed(futures), total=len(futures), desc="Encode images")
            result = []
            for future in to_do:
                result.append(future.result())

        return result
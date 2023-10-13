from io import BytesIO
import requests
from urllib.request import urlopen

import cv2
import numpy as np
from leptonai.photon import Photon, PNGResponse


# 需要继承 Photon 类
class Canny(Photon):
    """Canny 边缘检测算子"""

    # 用这个装饰器表示这个一个对外接口
    @Photon.handler("run")
    def run(self, url: str) -> PNGResponse:
        # 读取图像数据
        image = np.asarray(Image.open(io.BytesIO(urlopen(url).read())))

        # 进行边缘检测
        edges = cv2.Canny(image, 100, 200)

        # 将结果转换为输出字节形式
        is_success, im_buf_arr = cv2.imencode(".png", edges)
        byte_im = im_buf_arr.tobytes()

        # 返回图像格式的网络Response
        return PNGResponse(byte_im)

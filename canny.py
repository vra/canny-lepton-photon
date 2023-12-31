import io
from io import BytesIO
import requests
from urllib.request import urlopen

from leptonai.photon import Photon, PNGResponse


# 需要继承 Photon 类
class Canny(Photon):
    """Canny 边缘检测算子"""

    # 这里的依赖 Package 会在创建 Photon 时自动安装
    requirement_dependency = [
        "opencv-python",
        "numpy",
        "Pillow",
    ]

    # 用这个装饰器表示这个一个对外接口
    @Photon.handler("run")
    def run(self, url: str) -> PNGResponse:
        # 读取图像数据
        import cv2
        import numpy as np
        from PIL import Image

        image = np.asarray(Image.open(io.BytesIO(urlopen(url).read())))

        # 进行边缘检测
        edges = cv2.Canny(image, 100, 200)

        edges = Image.fromarray(edges)

        img_io = BytesIO()
        edges.save(img_io, format="PNG", quality="keep")
        img_io.seek(0)
        return PNGResponse(img_io)

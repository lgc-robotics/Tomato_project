"""由12个多点标定数据拟合得到的正式手眼标定。"""

import numpy as np


class Eye_in_hand:
    def __init__(self):
        # 相机坐标系到机械臂坐标系的旋转矩阵。
        self.R = np.array(
            [
                [-0.015397830982, -0.005080633439, 0.999868538341],
                [-0.999841801871, 0.008982998852, -0.015351773919],
                [-0.008903821197, -0.999946745030, -0.005218148390],
            ],
            dtype=np.float64,
        )

        # 固定拍照位下，相机光心在机械臂坐标系中的平移向量，单位：厘米。
        self.T = np.array(
            [
                [-27.311217643190],
                [59.063279546519],
                [48.107550541028],
            ],
            dtype=np.float64,
        )

        # 采集本次多点标定数据时使用的固定拍照位，单位：厘米。
        self.Te0 = np.array(
            [
                [3.0],
                [55.0],
                [30.0],
            ],
            dtype=np.float64,
        )

    def coordinate(self, cam_target, te):
        """把相机三维坐标转换成当前拍照位下的机械臂坐标。"""
        cam_target = np.asarray(cam_target, dtype=np.float64).reshape(3, 1)
        te = np.asarray(te, dtype=np.float64).reshape(3, 1)
        return self.R @ cam_target + self.T - self.Te0 + te

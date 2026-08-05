"""探测 OpenCV 摄像头编号，并保存预览图。

请使用和 main1.py 相同的 Python 环境运行：
    python camera_probe.py

查看 camera_probe_index_*.jpg，找到循迹摄像头对应的编号，
然后把 config1.py 里的 CHASSIS_CAMERA_INDEX 改成该编号。
"""

import time

import cv2


def main():
    for index in range(9):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        opened = cap.isOpened()
        ret = False
        frame = None

        if opened:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            time.sleep(0.3)
            ret, frame = cap.read()

        print(
            f"编号={index}, 已打开={opened}, 读取成功={ret}, "
            f"图像尺寸={None if frame is None else frame.shape}"
        )

        if ret and frame is not None:
            filename = f"camera_probe_index_{index}.jpg"
            cv2.imwrite(filename, frame)
            print(f"已保存 {filename}")

        cap.release()

    print("探测完成，请打开保存的 camera_probe_index_*.jpg 文件查看。")


if __name__ == "__main__":
    main()

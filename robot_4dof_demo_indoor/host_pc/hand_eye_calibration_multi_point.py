"""多点手眼标定：用末端标定针传递机械臂坐标，再由固定视角测量同一点。

工作流程：
1. 输入标定针需要到达的机械臂坐标 P。
2. 程序发送 P，等待机械臂到位。
3. 将平面标记片的中心固定在针尖位置，手离开机械臂运动范围。
4. 保持 Y/Z 不变，先让 X 轴单独回退一段安全距离，避开标记物。
5. X 轴脱离标记物后，程序才允许三轴回到固定拍照位 Te0。
6. 在 RGB 图像中重复点击标记中心，程序使用多帧局部深度计算相机坐标 C。
7. 保存 P <-> C；重复采集后用 SVD 最小二乘求解 P = R*C + T。

脚本不会旋转末端、控制刀片、移动底盘或执行采摘，也不会覆盖 calibration.py。
"""

import argparse
import csv
import json
import os
import re
import time
import warnings
from datetime import datetime
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np
import pyrealsense2 as rs
import serial

from config1 import (
    BAUDRATE,
    COLOR_HEIGHT,
    COLOR_WIDTH,
    DEPTH_HEIGHT,
    DEPTH_WIDTH,
    FPS,
    MAX_DEPTH,
    MIN_DEPTH,
    ROBOT_ACK_TIMEOUT_SECONDS,
    SCAN_X,
    SCAN_Y_START,
    SCAN_Z_START,
    SERIAL_PORT,
    SERIAL_TIMEOUT,
    X_MIN,
)
from robot import (
    is_robot_position_in_safe_range,
    print_robot_position_limit_error,
    send_robot_position,
)


# 每次运行的数据都会保存到该目录下的独立时间戳文件夹。
OUTPUT_ROOT_NAME = "多点手眼标定"

# 推荐使用中间扫描层作为默认固定拍照位；运行时可以重新输入。
DEFAULT_PHOTO_POSE_CM = np.array(
    [SCAN_X, SCAN_Y_START, max(SCAN_Z_START, 21.0)],
    dtype=np.float64,
)

# RealSense 启动预热和拍照参数。
CAMERA_WARMUP_FRAMES = 30
CAMERA_SETTLE_SECONDS = 1.0
CAPTURE_DISCARD_FRAMES = 10
DEPTH_SAMPLE_FRAMES = 15

# 同一中心重复点击次数。取中位像素可减小单次手点误差。
CLICK_COUNT = 3

# 每个点击位置周围取深度的半径；4 表示使用 9x9 像素邻域。
DEPTH_PATCH_RADIUS_PX = 4

# 使用中位数绝对偏差过滤局部深度离群点时，至少保留的绝对范围。
DEPTH_OUTLIER_MIN_BAND_M = 0.004
DEPTH_OUTLIER_MAD_SCALE = 3.5

# 至少获得这么多有效局部深度样本才接受一次测量。
MIN_VALID_DEPTH_SAMPLES = 30

# 稳健拟合中，单点三维残差超过该值会被列为候选异常点。
OUTLIER_RESIDUAL_THRESHOLD_CM = 1.5
MIN_POINTS_FOR_OUTLIER_REJECTION = 6

# 建议至少采集数量。少于该数量仍可求解，但终端会提示覆盖不足。
RECOMMENDED_CALIBRATION_POINTS = 12
RECOMMENDED_VALIDATION_POINTS = 3

# 标记物固定后，机械臂返回拍照位之前，X 轴必须先单独回退的距离。
# 回退阶段 Y/Z 完全保持针尖标定点坐标不变，避免三轴联动时刀具或标定针碰到标记物。
# 机械臂 X 轴的回退方向是数值减小；现场可通过 --retreat-x 修改该距离。
DEFAULT_X_CLEARANCE_RETREAT_CM = 10.0


class UserCancelled(Exception):
    """用户主动结束采集。"""


def parse_triplet(text):
    """解析逗号或空格分隔的三个浮点数。"""
    normalized = (
        text.strip()
        .replace("，", ",")
        .replace("（", "")
        .replace("）", "")
        .replace("(", "")
        .replace(")", "")
    )
    parts = [part for part in re.split(r"[,\s]+", normalized) if part]
    if len(parts) != 3:
        raise ValueError("必须输入三个数，例如 35, 40, 20")
    return np.array([float(part) for part in parts], dtype=np.float64)


def format_vector(vector, digits=3):
    values = np.asarray(vector, dtype=np.float64).reshape(3)
    return "(" + ", ".join(f"{value:.{digits}f}" for value in values) + ")"


def prompt_photo_pose(default_pose):
    """读取固定拍照位，直接回车使用默认值。"""
    while True:
        raw = input(
            "请输入固定拍照位 X,Y,Z（厘米）"
            f"，直接回车使用 {format_vector(default_pose, 2)}: "
        ).strip()
        try:
            pose = default_pose.copy() if not raw else parse_triplet(raw)
        except ValueError as exc:
            print(f"输入格式错误: {exc}")
            continue

        if validate_robot_position(pose):
            return pose


def prompt_target_position():
    """读取一个针尖坐标；DONE 表示结束采集。"""
    while True:
        raw = input(
            "\n请输入标定针目标坐标 X,Y,Z（厘米），输入 DONE 结束采集: "
        ).strip()
        if raw.lower() in {"done", "d", "结束", "完成"}:
            return None

        try:
            position = parse_triplet(raw)
        except ValueError as exc:
            print(f"输入格式错误: {exc}")
            continue

        if validate_robot_position(position):
            return position


def prompt_role():
    """选择该点用于拟合还是只用于独立验证。"""
    while True:
        raw = input(
            "该点用途：直接回车=参与标定，输入 V=仅用于独立验证: "
        ).strip().lower()
        if raw in {"", "c", "calibration", "标定"}:
            return "calibration"
        if raw in {"v", "validation", "验证"}:
            return "validation"
        print("请输入 C、V，或直接回车")


def validate_robot_position(position):
    x, y, z = map(float, position)
    if is_robot_position_in_safe_range(x, y, z):
        return True
    print_robot_position_limit_error(x, y, z)
    return False


def create_session_directory():
    root = Path(__file__).resolve().parent / OUTPUT_ROOT_NAME
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = root / f"标定_{timestamp}"
    suffix = 1
    while session_dir.exists():
        session_dir = root / f"标定_{timestamp}_{suffix:02d}"
        suffix += 1
    session_dir.mkdir(parents=True, exist_ok=False)
    return session_dir


def start_realsense():
    """按主程序相同参数启动并预热 RealSense。"""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(
        rs.stream.color,
        COLOR_WIDTH,
        COLOR_HEIGHT,
        rs.format.bgr8,
        FPS,
    )
    config.enable_stream(
        rs.stream.depth,
        DEPTH_WIDTH,
        DEPTH_HEIGHT,
        rs.format.z16,
        FPS,
    )
    pipeline.start(config)
    align = rs.align(rs.stream.color)

    print(f"RealSense 正在预热 {CAMERA_WARMUP_FRAMES} 帧...")
    for _ in range(CAMERA_WARMUP_FRAMES):
        pipeline.wait_for_frames(5000)
    print("RealSense 已启动并完成预热")
    return pipeline, align


def intrinsics_to_dict(intrinsics):
    return {
        "width": int(intrinsics.width),
        "height": int(intrinsics.height),
        "fx": float(intrinsics.fx),
        "fy": float(intrinsics.fy),
        "ppx": float(intrinsics.ppx),
        "ppy": float(intrinsics.ppy),
        "distortion_model": str(intrinsics.model),
        "coeffs": [float(value) for value in intrinsics.coeffs],
    }


def capture_aligned_rgb_depth_stack(pipeline, align):
    """丢弃缓存帧后采集彩色图和多帧对齐深度，深度单位为米。"""
    total_frames = CAPTURE_DISCARD_FRAMES + DEPTH_SAMPLE_FRAMES
    color_image = None
    intrinsics = None
    depth_images = []

    for frame_index in range(total_frames):
        frames = pipeline.wait_for_frames(5000)
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        if frame_index < CAPTURE_DISCARD_FRAMES:
            continue

        color_image = np.asanyarray(color_frame.get_data()).copy()
        intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
        raw_depth = np.asanyarray(depth_frame.get_data()).astype(np.float32)
        depth_images.append(raw_depth * float(depth_frame.get_units()))

    if color_image is None or intrinsics is None or not depth_images:
        raise RuntimeError("连续等待后仍未获得完整的彩色/深度对齐帧")

    return color_image, np.stack(depth_images, axis=0), intrinsics


def annotate_clicks(image, points):
    annotated = image.copy()
    for index, point in enumerate(points, start=1):
        u, v = map(int, point)
        cv2.circle(annotated, (u, v), 7, (0, 0, 255), 2)
        cv2.putText(
            annotated,
            str(index),
            (u + 9, max(18, v - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    if points:
        median_pixel = np.rint(np.median(np.asarray(points), axis=0)).astype(int)
        u, v = map(int, median_pixel)
        cv2.line(annotated, (u - 16, v), (u + 16, v), (0, 255, 255), 1)
        cv2.line(annotated, (u, v - 16), (u, v + 16), (0, 255, 255), 1)
    return annotated


def select_marker_center(image, click_count):
    """在同一静态图上重复点击标记中心，并返回所有点击和中位像素。"""
    window_name = "多点手眼标定 - 点击标记中心"
    state = {"points": [], "display": image.copy()}

    def on_mouse(event, x, y, _flags, _userdata):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if len(state["points"]) >= click_count:
            print("已达到点击次数；按 R 可清空后重新点击")
            return
        state["points"].append((int(x), int(y)))
        state["display"] = annotate_clicks(image, state["points"])
        print(
            f"第 {len(state['points'])}/{click_count} 次点击: "
            f"u={x}, v={y}"
        )

    print(f"\n请连续点击标记片上针尖接触的同一个中心，共 {click_count} 次")
    print("达到次数后按 Enter 或空格确认；按 R 清空重选；按 Esc 结束采集")
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        cv2.imshow(window_name, state["display"])
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10, 32) and len(state["points"]) == click_count:
            points = list(state["points"])
            median_pixel = np.rint(np.median(np.asarray(points), axis=0)).astype(int)
            annotated = annotate_clicks(image, points)
            cv2.destroyWindow(window_name)
            return points, median_pixel, annotated
        if key in (ord("r"), ord("R")):
            state["points"] = []
            state["display"] = image.copy()
            print("点击点已清空，请重新选择")
        if key == 27:
            cv2.destroyWindow(window_name)
            raise UserCancelled("用户在点击窗口中结束采集")


def estimate_camera_point(depth_stack_m, intrinsics, clicked_points, median_pixel):
    """用所有点击邻域的多帧深度中位数估计相机三维坐标。"""
    height, width = depth_stack_m.shape[1:]
    samples = []

    for point in clicked_points:
        u, v = map(int, point)
        x1 = max(0, u - DEPTH_PATCH_RADIUS_PX)
        x2 = min(width, u + DEPTH_PATCH_RADIUS_PX + 1)
        y1 = max(0, v - DEPTH_PATCH_RADIUS_PX)
        y2 = min(height, v + DEPTH_PATCH_RADIUS_PX + 1)
        patch = depth_stack_m[:, y1:y2, x1:x2].reshape(-1)
        valid = patch[
            np.isfinite(patch)
            & (patch >= float(MIN_DEPTH))
            & (patch <= float(MAX_DEPTH))
        ]
        if valid.size:
            samples.append(valid)

    if not samples:
        raise RuntimeError("点击位置附近没有任何合法深度")

    values = np.concatenate(samples).astype(np.float64)
    if values.size < MIN_VALID_DEPTH_SAMPLES:
        raise RuntimeError(
            f"有效深度样本只有 {values.size} 个，"
            f"少于要求的 {MIN_VALID_DEPTH_SAMPLES} 个"
        )

    median_depth = float(np.median(values))
    mad = float(np.median(np.abs(values - median_depth)))
    robust_sigma = 1.4826 * mad
    keep_band = max(
        DEPTH_OUTLIER_MIN_BAND_M,
        DEPTH_OUTLIER_MAD_SCALE * robust_sigma,
    )
    filtered = values[np.abs(values - median_depth) <= keep_band]
    if filtered.size < MIN_VALID_DEPTH_SAMPLES:
        raise RuntimeError(
            f"离群过滤后只剩 {filtered.size} 个深度样本，标记物深度可能不稳定"
        )

    depth_m = float(np.median(filtered))
    depth_std_m = float(np.std(filtered))
    u, v = map(float, median_pixel)
    camera_point_m = rs.rs2_deproject_pixel_to_point(
        intrinsics,
        [u, v],
        depth_m,
    )
    camera_point_cm = np.asarray(camera_point_m, dtype=np.float64) * 100.0

    return {
        "camera_point_cm": camera_point_cm,
        "depth_m": depth_m,
        "depth_std_m": depth_std_m,
        "raw_sample_count": int(values.size),
        "filtered_sample_count": int(filtered.size),
        "depth_keep_band_m": float(keep_band),
    }


def save_jpeg(path, image):
    encoded_ok, encoded_image = cv2.imencode(".jpg", image)
    if not encoded_ok:
        raise RuntimeError(f"JPEG 编码失败: {path}")
    Path(path).write_bytes(encoded_image.tobytes())


def save_depth_visualization(path, depth_stack_m, median_pixel):
    # 空旷背景中部分像素可能连续所有帧都没有深度。
    # 这些像素在可视化中填0即可，不应产生 All-NaN slice 警告。
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="All-NaN slice encountered",
            category=RuntimeWarning,
        )
        median_depth = np.nanmedian(
            np.where(depth_stack_m > 0, depth_stack_m, np.nan),
            axis=0,
        )
    scaled = np.nan_to_num(median_depth, nan=0.0)
    scaled = np.clip(scaled / max(float(MAX_DEPTH), 1e-6) * 255.0, 0, 255)
    colorized = cv2.applyColorMap(scaled.astype(np.uint8), cv2.COLORMAP_JET)
    u, v = map(int, median_pixel)
    cv2.circle(colorized, (u, v), 8, (255, 255, 255), 2)
    save_jpeg(path, colorized)


def json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def write_json(path, data):
    Path(path).write_text(
        json.dumps(json_ready(data), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


CSV_FIELDS = [
    "index",
    "role",
    "robot_x_cm",
    "robot_y_cm",
    "robot_z_cm",
    "camera_x_cm",
    "camera_y_cm",
    "camera_z_cm",
    "pixel_u",
    "pixel_v",
    "depth_m",
    "depth_std_m",
    "filtered_sample_count",
]


def write_measurements_csv(session_dir, measurements):
    rows = []
    for measurement in measurements:
        robot_point = measurement["needle_robot_cm"]
        camera_point = measurement["camera_point_cm"]
        pixel = measurement["median_pixel"]
        rows.append(
            {
                "index": measurement["index"],
                "role": measurement["role"],
                "robot_x_cm": robot_point[0],
                "robot_y_cm": robot_point[1],
                "robot_z_cm": robot_point[2],
                "camera_x_cm": camera_point[0],
                "camera_y_cm": camera_point[1],
                "camera_z_cm": camera_point[2],
                "pixel_u": pixel[0],
                "pixel_v": pixel[1],
                "depth_m": measurement["depth_m"],
                "depth_std_m": measurement["depth_std_m"],
                "filtered_sample_count": measurement["filtered_sample_count"],
            }
        )

    csv_path = Path(session_dir) / "标定点汇总.csv"
    try:
        with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    except PermissionError:
        fallback = Path(session_dir) / (
            "标定点汇总_" + datetime.now().strftime("%H%M%S") + ".csv"
        )
        with fallback.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        print(f"原CSV被占用，已改存为: {fallback.name}")


def save_session(session_dir, session_data):
    write_json(Path(session_dir) / "session.json", session_data)
    write_measurements_csv(session_dir, session_data.get("measurements", []))


def fit_rigid_transform(camera_points, robot_points):
    """Kabsch/SVD：求解 robot = R @ camera + T。"""
    camera_points = np.asarray(camera_points, dtype=np.float64)
    robot_points = np.asarray(robot_points, dtype=np.float64)
    if camera_points.shape != robot_points.shape or camera_points.ndim != 2:
        raise ValueError("相机点和机械臂点必须是形状相同的 N x 3 数组")
    if camera_points.shape[0] < 3 or camera_points.shape[1] != 3:
        raise ValueError("至少需要三个三维对应点")

    camera_center = np.mean(camera_points, axis=0)
    robot_center = np.mean(robot_points, axis=0)
    camera_zero = camera_points - camera_center
    robot_zero = robot_points - robot_center
    covariance = camera_zero.T @ robot_zero
    u, singular_values, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T
    translation = robot_center - rotation @ camera_center
    return rotation, translation, singular_values


def predict_points(rotation, translation, camera_points):
    camera_points = np.asarray(camera_points, dtype=np.float64)
    return (rotation @ camera_points.T).T + translation.reshape(1, 3)


def residual_summary(errors):
    errors = np.asarray(errors, dtype=np.float64)
    norms = np.linalg.norm(errors, axis=1)
    return {
        "count": int(errors.shape[0]),
        "mean_axis_cm": np.mean(errors, axis=0),
        "std_axis_cm": np.std(errors, axis=0),
        "mae_axis_cm": np.mean(np.abs(errors), axis=0),
        "rmse_3d_cm": float(np.sqrt(np.mean(norms ** 2))),
        "mean_3d_cm": float(np.mean(norms)),
        "max_3d_cm": float(np.max(norms)),
    }


def robust_fit(camera_points, robot_points):
    """迭代剔除明显坏点；返回最终变换和每个标定点的内点标记。"""
    camera_points = np.asarray(camera_points, dtype=np.float64)
    robot_points = np.asarray(robot_points, dtype=np.float64)
    inliers = np.ones(camera_points.shape[0], dtype=bool)

    if camera_points.shape[0] < MIN_POINTS_FOR_OUTLIER_REJECTION:
        rotation, translation, singular_values = fit_rigid_transform(
            camera_points,
            robot_points,
        )
        return rotation, translation, singular_values, inliers

    for _ in range(6):
        rotation, translation, singular_values = fit_rigid_transform(
            camera_points[inliers],
            robot_points[inliers],
        )
        predicted = predict_points(rotation, translation, camera_points)
        residual_norms = np.linalg.norm(predicted - robot_points, axis=1)
        new_inliers = residual_norms <= OUTLIER_RESIDUAL_THRESHOLD_CM

        if np.count_nonzero(new_inliers) < MIN_POINTS_FOR_OUTLIER_REJECTION:
            break
        if np.array_equal(new_inliers, inliers):
            inliers = new_inliers
            break
        inliers = new_inliers

    rotation, translation, singular_values = fit_rigid_transform(
        camera_points[inliers],
        robot_points[inliers],
    )
    return rotation, translation, singular_values, inliers


def build_generated_calibration_source(rotation, translation, photo_pose):
    def array_text(array):
        return np.array2string(
            np.asarray(array, dtype=np.float64),
            separator=", ",
            precision=12,
            suppress_small=False,
        )

    rotation_text = array_text(rotation)
    translation_text = array_text(np.asarray(translation).reshape(3, 1))
    photo_pose_text = array_text(np.asarray(photo_pose).reshape(3, 1))
    return f'''"""由 hand_eye_calibration_multi_point.py 自动生成的候选手眼标定。"""

import numpy as np


class Eye_in_hand:
    def __init__(self):
        # 相机坐标系到机械臂坐标系的旋转矩阵。
        self.R = np.array({rotation_text}, dtype=np.float64)

        # 固定拍照位下，相机光心在机械臂坐标系中的平移向量，单位：厘米。
        self.T = np.array({translation_text}, dtype=np.float64)

        # 采集标定数据时使用的固定拍照位，单位：厘米。
        self.Te0 = np.array({photo_pose_text}, dtype=np.float64)

    def coordinate(self, cam_target, te):
        """把相机三维坐标转换成当前拍照位下的机械臂坐标。"""
        cam_target = np.asarray(cam_target, dtype=np.float64).reshape(3, 1)
        te = np.asarray(te, dtype=np.float64).reshape(3, 1)
        return self.R @ cam_target + self.T - self.Te0 + te
'''


def solve_and_save(session_dir, session_data):
    measurements = session_data.get("measurements", [])
    calibration_items = [
        item for item in measurements if item.get("role") == "calibration"
    ]
    validation_items = [
        item for item in measurements if item.get("role") == "validation"
    ]

    if len(calibration_items) < 3:
        print("\n参与标定的点少于3个，数据已保存，但暂时不能计算变换矩阵")
        return None

    camera_points = np.asarray(
        [item["camera_point_cm"] for item in calibration_items],
        dtype=np.float64,
    )
    robot_points = np.asarray(
        [item["needle_robot_cm"] for item in calibration_items],
        dtype=np.float64,
    )
    rotation, translation, singular_values, inliers = robust_fit(
        camera_points,
        robot_points,
    )

    predicted = predict_points(rotation, translation, camera_points)
    errors = predicted - robot_points
    calibration_summary = residual_summary(errors[inliers])
    all_summary = residual_summary(errors)

    point_results = []
    for item, predicted_point, error, is_inlier in zip(
        calibration_items,
        predicted,
        errors,
        inliers,
    ):
        point_results.append(
            {
                "index": item["index"],
                "role": "calibration",
                "inlier": bool(is_inlier),
                "predicted_robot_cm": predicted_point,
                "error_axis_cm": error,
                "error_3d_cm": float(np.linalg.norm(error)),
            }
        )

    validation_summary = None
    if validation_items:
        validation_camera = np.asarray(
            [item["camera_point_cm"] for item in validation_items],
            dtype=np.float64,
        )
        validation_robot = np.asarray(
            [item["needle_robot_cm"] for item in validation_items],
            dtype=np.float64,
        )
        validation_predicted = predict_points(
            rotation,
            translation,
            validation_camera,
        )
        validation_errors = validation_predicted - validation_robot
        validation_summary = residual_summary(validation_errors)
        for item, predicted_point, error in zip(
            validation_items,
            validation_predicted,
            validation_errors,
        ):
            point_results.append(
                {
                    "index": item["index"],
                    "role": "validation",
                    "inlier": None,
                    "predicted_robot_cm": predicted_point,
                    "error_axis_cm": error,
                    "error_3d_cm": float(np.linalg.norm(error)),
                }
            )

    centered = camera_points[inliers] - np.mean(camera_points[inliers], axis=0)
    geometry_singular_values = np.linalg.svd(centered, compute_uv=False)
    geometry_ratio = float(
        geometry_singular_values[-1] / max(geometry_singular_values[0], 1e-12)
    )

    result = {
        "formula": "robot_cm = R_camera_to_robot @ camera_cm + T_at_photo_pose_cm",
        "photo_pose_te0_cm": session_data["photo_pose_cm"],
        "R_camera_to_robot": rotation,
        "T_at_photo_pose_cm": translation.reshape(3, 1),
        "det_R": float(np.linalg.det(rotation)),
        "fit_singular_values": singular_values,
        "geometry_singular_values": geometry_singular_values,
        "geometry_smallest_to_largest_ratio": geometry_ratio,
        "outlier_threshold_cm": OUTLIER_RESIDUAL_THRESHOLD_CM,
        "calibration_point_count": len(calibration_items),
        "calibration_inlier_count": int(np.count_nonzero(inliers)),
        "excluded_calibration_indices": [
            item["index"]
            for item, is_inlier in zip(calibration_items, inliers)
            if not is_inlier
        ],
        "calibration_inlier_error": calibration_summary,
        "calibration_all_error": all_summary,
        "validation_error": validation_summary,
        "point_results": point_results,
    }

    write_json(Path(session_dir) / "calibration_result.json", result)
    generated_source = build_generated_calibration_source(
        rotation,
        translation,
        session_data["photo_pose_cm"],
    )
    (Path(session_dir) / "calibration_generated.py").write_text(
        generated_source,
        encoding="utf-8",
    )

    print("\n================ 多点手眼标定结果 ================")
    print(f"参与标定: {len(calibration_items)} 点")
    print(f"最终内点: {np.count_nonzero(inliers)} 点")
    if result["excluded_calibration_indices"]:
        print("排除的异常点编号:", result["excluded_calibration_indices"])
    print("R =")
    print(rotation)
    print("T =", format_vector(translation))
    print(
        "标定内点平均轴误差: "
        f"{format_vector(calibration_summary['mean_axis_cm'])} 厘米"
    )
    print(
        "标定内点三维误差: "
        f"RMSE={calibration_summary['rmse_3d_cm']:.3f} 厘米, "
        f"最大={calibration_summary['max_3d_cm']:.3f} 厘米"
    )

    if validation_summary is not None:
        print(
            "独立验证平均轴误差: "
            f"{format_vector(validation_summary['mean_axis_cm'])} 厘米"
        )
        print(
            "独立验证三维误差: "
            f"RMSE={validation_summary['rmse_3d_cm']:.3f} 厘米, "
            f"最大={validation_summary['max_3d_cm']:.3f} 厘米"
        )
    else:
        print("本次没有独立验证点；建议再采集至少3个 V 点后评价泛化误差")

    if geometry_ratio < 0.02:
        print("警告: 标定点接近平面分布，请增加不同 X 深度的标定点")
    if len(calibration_items) < RECOMMENDED_CALIBRATION_POINTS:
        print(
            f"提示: 当前标定点少于建议的 {RECOMMENDED_CALIBRATION_POINTS} 个，"
            "建议继续增加空间覆盖"
        )
    print(f"\n完整结果: {Path(session_dir) / 'calibration_result.json'}")
    print(f"候选标定代码: {Path(session_dir) / 'calibration_generated.py'}")
    print("该候选文件不会自动覆盖现有 calibration.py")
    return result


def collect_measurement(
    pipeline,
    align,
    arm_ser,
    session_dir,
    point_index,
    target_position,
    photo_pose,
    role,
    x_clearance_retreat_cm,
):
    print("\n==================================================")
    print(f"采集第 {point_index} 个点，类型: {'标定' if role == 'calibration' else '验证'}")
    print(f"标定针目标坐标: {format_vector(target_position, 2)} 厘米")
    print(f"固定拍照位: {format_vector(photo_pose, 2)} 厘米")
    print("==================================================")

    confirm = input("确认针尖运动路径无障碍，输入 MOVE 开始移动: ").strip().lower()
    if confirm != "move":
        print("未输入 MOVE，本点已取消")
        return None

    if not send_robot_position(
        arm_ser,
        *target_position,
        ack_timeout_seconds=ROBOT_ACK_TIMEOUT_SECONDS,
    ):
        raise RuntimeError("标定针未能到达目标坐标")

    print("\n标定针已到位。现在将平面标记片的中心贴到针尖并可靠固定。")
    print("不要用手拿着标记物等待机械臂运动。确认标记不会移动且人员已离开运动范围。")
    fixed = input("完成后输入 FIXED，输入 SKIP 放弃本点: ").strip().lower()
    if fixed == "skip":
        return None
    if fixed != "fixed":
        print("未输入 FIXED，本点已取消")
        return None

    requested_retreat = max(0.0, float(x_clearance_retreat_cm))
    clearance_position = np.asarray(target_position, dtype=np.float64).copy()
    clearance_position[0] = max(
        float(X_MIN),
        float(target_position[0]) - requested_retreat,
    )
    actual_retreat = float(target_position[0] - clearance_position[0])

    if actual_retreat <= 1e-6:
        print("当前标定点的 X 已经位于最小位置，无法执行必需的单轴回退")
        print("请移除标记物并重新选择 X 更大的标定针目标点")
        input("移除标记物并确认路径安全后按 Enter: ")
        return None

    if actual_retreat + 1e-6 < requested_retreat:
        print(
            "警告: 受 X_MIN 限制，X 轴只能回退 "
            f"{actual_retreat:.2f} 厘米，少于设置的 {requested_retreat:.2f} 厘米"
        )
        confirm_short = input(
            "确认该距离仍足以完全脱离标记物，输入 CONTINUE；其他输入取消本点: "
        ).strip().lower()
        if confirm_short != "continue":
            print("本点已取消")
            input("移除标记物并确认路径安全后按 Enter: ")
            return None

    print("\n第一段安全脱离：只让 X 轴回退，Y/Z 保持不变")
    print(
        f"X: {target_position[0]:.2f} -> {clearance_position[0]:.2f} 厘米，"
        f"实际回退 {actual_retreat:.2f} 厘米"
    )
    print(
        "安全脱离坐标: "
        f"{format_vector(clearance_position, 2)} 厘米"
    )
    if not send_robot_position(
        arm_ser,
        *clearance_position,
        ack_timeout_seconds=ROBOT_ACK_TIMEOUT_SECONDS,
    ):
        raise RuntimeError("X 轴单独安全回退失败，禁止继续三轴返回拍照位")

    print("\nX 轴已安全脱离标记物，现在才允许三轴回到固定拍照位")
    if not send_robot_position(
        arm_ser,
        *photo_pose,
        ack_timeout_seconds=ROBOT_ACK_TIMEOUT_SECONDS,
    ):
        raise RuntimeError("机械臂未能回到固定拍照位")

    print(f"拍照位已到达，等待稳定 {CAMERA_SETTLE_SECONDS:.1f} 秒")
    time.sleep(CAMERA_SETTLE_SECONDS)

    while True:
        color_image, depth_stack, intrinsics = capture_aligned_rgb_depth_stack(
            pipeline,
            align,
        )
        clicked_points, median_pixel, annotated = select_marker_center(
            color_image,
            CLICK_COUNT,
        )
        try:
            depth_result = estimate_camera_point(
                depth_stack,
                intrinsics,
                clicked_points,
                median_pixel,
            )
        except RuntimeError as exc:
            print(f"深度计算失败: {exc}")
            retry = input("输入 R 重新拍摄，其他输入结束采集: ").strip().lower()
            if retry == "r":
                continue
            raise UserCancelled("用户在深度计算失败后结束采集")

        camera_point = depth_result["camera_point_cm"]
        print("\n本点测量结果:")
        print(f"中位像素: u={median_pixel[0]}, v={median_pixel[1]}")
        print(f"相机坐标: {format_vector(camera_point)} 厘米")
        print(
            f"深度: {depth_result['depth_m'] * 100:.2f} 厘米, "
            f"深度标准差: {depth_result['depth_std_m'] * 100:.3f} 厘米, "
            f"有效样本: {depth_result['filtered_sample_count']}"
        )
        accept = input("直接回车接受；输入 R 重新拍摄；输入 Q 结束采集: ").strip().lower()
        if accept == "r":
            continue
        if accept == "q":
            raise UserCancelled("用户结束采集")
        break

    point_dir = Path(session_dir) / f"point_{point_index:03d}"
    point_dir.mkdir(parents=True, exist_ok=False)
    raw_image_name = "rgb_raw.jpg"
    clicked_image_name = "rgb_clicked.jpg"
    depth_image_name = "depth_median.jpg"
    save_jpeg(point_dir / raw_image_name, color_image)
    save_jpeg(point_dir / clicked_image_name, annotated)
    save_depth_visualization(point_dir / depth_image_name, depth_stack, median_pixel)

    measurement = {
        "index": int(point_index),
        "role": role,
        "needle_robot_cm": target_position,
        "photo_pose_cm": photo_pose,
        "x_clearance_position_cm": clearance_position,
        "x_clearance_retreat_requested_cm": requested_retreat,
        "x_clearance_retreat_actual_cm": actual_retreat,
        "clicked_pixels": clicked_points,
        "median_pixel": median_pixel,
        "camera_point_cm": camera_point,
        "depth_m": depth_result["depth_m"],
        "depth_std_m": depth_result["depth_std_m"],
        "raw_sample_count": depth_result["raw_sample_count"],
        "filtered_sample_count": depth_result["filtered_sample_count"],
        "depth_keep_band_m": depth_result["depth_keep_band_m"],
        "intrinsics": intrinsics_to_dict(intrinsics),
        "images": {
            "raw": str(Path(point_dir.name) / raw_image_name),
            "clicked": str(Path(point_dir.name) / clicked_image_name),
            "depth": str(Path(point_dir.name) / depth_image_name),
        },
        "captured_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(point_dir / "measurement.json", measurement)
    print(f"第 {point_index} 个点已保存: {point_dir}")
    return json_ready(measurement)


def run_collection(args):
    print("\n多点手眼标定采集程序")
    print("本程序只移动机械臂并采集 RGB-D，不控制旋转、刀片或底盘")
    print("建议使用3～5厘米平面标记片；针尖接触点与点击点必须是同一个中心")

    if args.retreat_x <= 0:
        raise ValueError("X轴安全回退距离必须大于0厘米")
    print(
        "安全返回顺序: "
        f"先仅X轴回退 {args.retreat_x:.2f} 厘米，再三轴前往固定拍照位"
    )

    default_pose = (
        parse_triplet(args.photo_pose)
        if args.photo_pose is not None
        else DEFAULT_PHOTO_POSE_CM
    )
    photo_pose = prompt_photo_pose(default_pose)
    print(f"\n本次固定拍照位 Te0 = {format_vector(photo_pose, 2)} 厘米")
    print("整个标定过程中不能改变该拍照位、相机安装姿态或标定针位置")

    session_dir = create_session_directory()
    session_data = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "photo_pose_cm": photo_pose.tolist(),
        "serial_port": SERIAL_PORT,
        "camera_stream": {
            "color": [COLOR_WIDTH, COLOR_HEIGHT, FPS],
            "depth": [DEPTH_WIDTH, DEPTH_HEIGHT, FPS],
        },
        "settings": {
            "click_count": CLICK_COUNT,
            "depth_sample_frames": DEPTH_SAMPLE_FRAMES,
            "depth_patch_radius_px": DEPTH_PATCH_RADIUS_PX,
            "outlier_residual_threshold_cm": OUTLIER_RESIDUAL_THRESHOLD_CM,
            "x_clearance_retreat_cm": float(args.retreat_x),
        },
        "measurements": [],
    }
    save_session(session_dir, session_data)
    print(f"本次标定目录: {session_dir}")

    arm_ser = None
    pipeline = None
    align = None
    try:
        arm_ser = serial.Serial(
            port=SERIAL_PORT,
            baudrate=BAUDRATE,
            timeout=SERIAL_TIMEOUT,
        )
        print(f"机械臂串口已打开: {SERIAL_PORT}")
        pipeline, align = start_realsense()

        point_index = 1
        while True:
            if session_data["measurements"]:
                input(
                    "\n开始下一点前，请先移除上一个标记物并确认路径安全，"
                    "完成后按 Enter: "
                )

            target_position = prompt_target_position()
            if target_position is None:
                break
            role = prompt_role()
            measurement = collect_measurement(
                pipeline,
                align,
                arm_ser,
                session_dir,
                point_index,
                target_position,
                photo_pose,
                role,
                args.retreat_x,
            )
            if measurement is None:
                continue
            session_data["measurements"].append(measurement)
            save_session(session_dir, session_data)
            point_index += 1

    except (KeyboardInterrupt, UserCancelled) as exc:
        print(f"\n采集已结束: {exc}")
    finally:
        cv2.destroyAllWindows()
        if pipeline is not None:
            pipeline.stop()
            print("RealSense 已停止")
        if arm_ser is not None and arm_ser.is_open:
            arm_ser.close()
            print("机械臂串口已关闭")
        save_session(session_dir, session_data)

    calibration_count = sum(
        item["role"] == "calibration" for item in session_data["measurements"]
    )
    validation_count = sum(
        item["role"] == "validation" for item in session_data["measurements"]
    )
    print(
        f"\n采集完成: 标定点 {calibration_count} 个，"
        f"验证点 {validation_count} 个"
    )
    if validation_count < RECOMMENDED_VALIDATION_POINTS:
        print(
            f"建议至少准备 {RECOMMENDED_VALIDATION_POINTS} 个不参与拟合的验证点"
        )
    solve_and_save(session_dir, session_data)


def solve_existing_session(folder):
    session_dir = Path(folder).resolve()
    session_path = session_dir / "session.json"
    if not session_path.exists():
        raise FileNotFoundError(f"没有找到会话文件: {session_path}")
    session_data = json.loads(session_path.read_text(encoding="utf-8"))
    solve_and_save(session_dir, session_data)


def run_self_test():
    """用带噪声和一个坏点的合成数据验证SVD及异常点剔除。"""
    rng = np.random.default_rng(20260715)
    raw = rng.normal(size=(3, 3))
    q, _ = np.linalg.qr(raw)
    if np.linalg.det(q) < 0:
        q[:, -1] *= -1
    true_translation = np.array([31.0, 52.0, 18.0])
    camera_points = rng.uniform([-25, -20, 40], [25, 20, 100], size=(20, 3))
    robot_points = predict_points(q, true_translation, camera_points)
    robot_points += rng.normal(scale=0.02, size=robot_points.shape)
    robot_points[3] += np.array([4.0, -3.0, 2.0])

    rotation, translation, _singular_values, inliers = robust_fit(
        camera_points,
        robot_points,
    )
    clean_error = predict_points(
        rotation,
        translation,
        camera_points[inliers],
    ) - robot_points[inliers]
    summary = residual_summary(clean_error)
    assert np.count_nonzero(~inliers) == 1
    assert summary["rmse_3d_cm"] < 0.1
    assert np.linalg.norm(rotation - q) < 0.01
    assert np.linalg.norm(translation - true_translation) < 0.1
    print("多点SVD、异常点剔除和残差统计自检通过")


def build_argument_parser():
    parser = argparse.ArgumentParser(description="标定针多点手眼标定采集程序")
    parser.add_argument(
        "--photo-pose",
        help="固定拍照位 X,Y,Z，例如 --photo-pose 7,60,21",
    )
    parser.add_argument(
        "--retreat-x",
        type=float,
        default=DEFAULT_X_CLEARANCE_RETREAT_CM,
        help=(
            "标记固定后，返回拍照位前 X 轴单独回退的距离，单位厘米，"
            f"默认 {DEFAULT_X_CLEARANCE_RETREAT_CM:g}"
        ),
    )
    parser.add_argument(
        "--solve",
        metavar="SESSION_DIR",
        help="不连接硬件，仅重新计算一个已保存的标定会话",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="只运行数学自检，不连接任何硬件",
    )
    return parser


def main():
    args = build_argument_parser().parse_args()
    if args.self_test:
        run_self_test()
        return
    if args.solve:
        solve_existing_session(args.solve)
        return
    run_collection(args)


if __name__ == "__main__":
    main()

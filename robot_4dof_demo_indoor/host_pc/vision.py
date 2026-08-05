# -*- coding: utf-8 -*-
"""视觉检测和 RGB-D 定位辅助函数。"""

import math
import time

import cv2
import numpy as np
import pyrealsense2 as rs

from run_recorder import (
    allocate_depth_diagnostic_paths,
    allocate_result_image_path,
    begin_capture_session,
    register_result_image,
)

from config1 import (
    ALLOW_REFERENCE_DEPTH_FALLBACK,
    BACKGROUND_DEPTH_BAND_M,
    BACKGROUND_DEPTH_M,
    CUT_HIGH_RISK_DISTANCE_CM,
    CUT_DEPTH_BACKGROUND_PENALTY,
    CUT_DEPTH_CLUSTER_BAND_M,
    CUT_DEPTH_CONNECTED_MIN_POINTS,
    CUT_DEPTH_DISTANCE_WEIGHT,
    CUT_DEPTH_MAX_RADIUS_PX,
    CUT_DEPTH_MIN_POINTS,
    CUT_DEPTH_OUTSIDE_MASK_PENALTY,
    CUT_DEPTH_RADIUS_STEP_PX,
    CUT_DEPTH_REFERENCE_FALLBACK_TRIGGER_M,
    CUT_DEPTH_REFERENCE_WEIGHT,
    CUT_DEPTH_SEARCH_RADIUS_PX,
    CUT_MASK_BACKGROUND_RING_MIN_POINTS,
    CUT_MASK_BACKGROUND_RING_WIDTH_PX,
    CUT_MASK_EROSION_ITERATIONS,
    CUT_MASK_EROSION_KERNEL_SIZE,
    CUT_MASK_EROSION_MIN_HALF_WIDTH_PX,
    CUT_MASK_EROSION_MIN_REMAINING_POINTS,
    CUT_MASK_MIN_BACKGROUND_DEPTH_CONTRAST_M,
    CUT_SAFE_DISTANCE_CM,
    DEPTH_CONSISTENCY_THRESHOLD,
    DEPTH_STABLE_BAND_M,
    DIST_THRESHOLD,
    ALLOW_EARLY_STOP_ON_STABLE_DEPTH,
    EARLY_STOP_MIN_FRAMES,
    ENABLE_BLADE_CONTACT_OFFSET,
    ENABLE_CUT_MASK_ADAPTIVE_EROSION,
    END_EFFECTOR_CLOCKWISE_SIGN,
    END_EFFECTOR_ROTATE_LIMIT_DEG,
    FRUIT_LEFT_USES_STATIC_BLADE,
    FRUIT_STEM_CLASS_NAMES,
    FINAL_RESULT_WAIT_MS,
    GOOD_DEPTH_MODES,
    GUIDE_BLADE_CONTACT_RATIO,
    IMG_SIZE,
    MAIN_STEM_CLASS_NAMES,
    MAIN_STEM_DEPTH_MAX_RADIUS_PX,
    MAIN_STEM_DEPTH_MIN_POINTS,
    MAIN_STEM_DEPTH_MODES,
    MAIN_STEM_DEPTH_RADIUS_STEP_PX,
    MAIN_STEM_DEPTH_REFERENCE_MAX_ERROR_M,
    MAIN_STEM_DEPTH_SEARCH_RADIUS_PX,
    MAIN_STEM_DEPTH_STABLE_BAND_M,
    MASK_THRESHOLD,
    MAX_DEPTH,
    MAX_PCA_LINE_LENGTH_CM,
    MIN_DEPTH,
    MIN_GOOD_DEPTH_FRAMES,
    MIN_MASK_PIXELS,
    MIN_MAIN_STEM_DEPTH_FRAMES,
    MIN_PCA_LINEARITY,
    MIN_PCA_LINE_LENGTH_CM,
    MIN_PCA_POINTS,
    MOVING_BLADE_TIP_OFFSET_CM,
    MIN_VOTE,
    MULTI_FRAME_DEPTH_STABLE_BAND_M,
    POINT_CLOUD_DEPTH_BAND_M,
    POINT_CLOUD_DEPTH_TRIM_HIGH_PERCENT,
    POINT_CLOUD_DEPTH_TRIM_LOW_PERCENT,
    POINT_CLOUD_CLUSTER_MIN_POINTS,
    POINT_CLOUD_DISTANCE_KEEP_PERCENT,
    POINT_CLOUD_SCENE_CONSENSUS_BAND_M,
    POINT_CLOUD_SCENE_MAX_ERROR_M,
    REFERENCE_DEPTH_MIN_POINTS,
    REALSENSE_FRAME_RETRY_COUNT,
    REALSENSE_FRAME_RETRY_DELAY_SECONDS,
    REALSENSE_FRAME_TIMEOUT_MS,
    SHOW_FINAL_RESULT,
    SHOW_DETECTION_THRESHOLD_RESULT,
    SAVE_CUT_DEPTH_DIAGNOSTICS,
    STATIC_BLADE_TIP_OFFSET_CM,
    STEM_NO_ROTATE_THRESHOLD_DEG,
    STEM_CUT_REGION_END_RATIO,
    STEM_CUT_REGION_START_RATIO,
    TARGET_DEPTH_BAND_M,
    TARGET_DEPTH_REFERENCE_M,
    UNVERIFIED_FRUIT_DEPTH_MODES,
    USE_MAIN_STEM_DEPTH_FALLBACK,
    USE_REFERENCE_DEPTH_WHEN_X_OUT_OF_RANGE,
    X_MAX,
    X_MIN,
    X_RANGE_REFERENCE_RECHECK_MARGIN_CM,
    Y_MAX,
    Y_MIN,
    YOLO_CONF,
    YOLO_INFERENCE_CONF,
    YOLO_IOU,
    YOLO_RETINA_MASKS,
)


def build_result_image_summary(targets):
    """生成图文报告中每张结果图对应的一行目标摘要。"""
    if not targets:
        return "当前帧没有形成有效果梗目标"

    target_summaries = []
    for index, target in enumerate(targets, start=1):
        target_summaries.append(
            f"ID{index}: 模式={target.get('depth_mode', 'NA')}, "
            f"深度={target.get('depth_m', 0) * 100:.1f}cm, "
            f"机械臂=({target.get('Xr', 0):.1f},"
            f"{target.get('Yr', 0):.1f},{target.get('Zr', 0):.1f})cm"
        )
    return "；".join(target_summaries)


def get_fruit_stem_detections_for_display(detections):
    """返回阈值图需要显示的全部果梗；类别名未知时沿用主流程的非主茎兜底。"""
    fruit_stems = [
        detection for detection in detections if detection.get("is_fruit_stem")
    ]
    if fruit_stems:
        return fruit_stems
    return [
        detection for detection in detections if not detection.get("is_main_stem")
    ]


def build_detection_threshold_summary(detections):
    fruit_stems = get_fruit_stem_detections_for_display(detections)
    fruit_stems = sorted(
        fruit_stems,
        key=lambda item: float(item.get("score", 0.0)),
        reverse=True,
    )
    passed = [item for item in fruit_stems if item.get("score", 0.0) >= YOLO_CONF]
    scores = "，".join(
        f"S{index}={item.get('score', 0.0):.3f}"
        for index, item in enumerate(fruit_stems, start=1)
    )
    if not scores:
        scores = "无"
    return (
        f"YOLO果梗置信度图：识别到{len(fruit_stems)}个，"
        f"达到采摘阈值{len(passed)}个；置信度：{scores}"
    )


def draw_label_with_background(image, text, origin, color):
    """绘制带黑色底的ASCII标签，避免复杂背景遮住置信度。"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness,
    )
    x = max(0, min(int(origin[0]), max(0, image.shape[1] - text_width - 8)))
    y = max(text_height + baseline + 5, min(int(origin[1]), image.shape[0] - 4))
    cv2.rectangle(
        image,
        (x, y - text_height - baseline - 5),
        (x + text_width + 7, y + 3),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        image,
        text,
        (x + 3, y - baseline),
        font,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def save_detection_threshold_image(image, detections, frame_id, capture_id=0):
    """保存YOLO阈值图：显示推理阈值以上的全部果梗及每个置信度。"""
    save_image = image.copy()
    fruit_stems = get_fruit_stem_detections_for_display(detections)
    fruit_stems = sorted(
        fruit_stems,
        key=lambda item: float(item.get("score", 0.0)),
        reverse=True,
    )

    for index, detection in enumerate(fruit_stems, start=1):
        score = float(detection.get("score", 0.0))
        passed = score >= YOLO_CONF
        color = (40, 210, 40) if passed else (0, 165, 255)
        mask = detection.get("mask")
        if mask is not None and np.asarray(mask).shape == save_image.shape[:2]:
            mask_bool = np.asarray(mask, dtype=bool)
            overlay = save_image.copy()
            overlay[mask_bool] = color
            save_image = cv2.addWeighted(overlay, 0.28, save_image, 0.72, 0)

        contour = detection.get("contour")
        if contour is not None and len(contour) >= 3:
            cv2.polylines(
                save_image,
                [np.asarray(contour, dtype=np.int32).reshape(-1, 1, 2)],
                True,
                color,
                2,
            )

        x1, y1, x2, y2 = np.round(detection["box"]).astype(int).tolist()
        cv2.rectangle(save_image, (x1, y1), (x2, y2), color, 2)
        status = "PASS" if passed else "LOW"
        label = f"S{index} conf={score:.3f} {status}"
        label_y = y1 - 5 if y1 >= 28 else y1 + 24
        draw_label_with_background(save_image, label, (x1, label_y), color)

    passed_count = sum(
        float(item.get("score", 0.0)) >= YOLO_CONF for item in fruit_stems
    )
    header = (
        f"STEMS={len(fruit_stems)}  INFER>={YOLO_INFERENCE_CONF:.2f}  "
        f"PICK>={YOLO_CONF:.2f}  PASS={passed_count}"
    )
    draw_label_with_background(save_image, header, (10, 28), (255, 255, 255))

    image_id, image_path = allocate_result_image_path(
        capture_id,
        frame_id,
        image_kind="threshold",
    )
    encoded_ok, encoded_image = cv2.imencode(".jpg", save_image)
    saved = False
    if encoded_ok:
        try:
            image_path.write_bytes(encoded_image.tobytes())
            saved = True
        except OSError as exc:
            print(f"\n阈值图文件写入失败: {exc}")

    if saved:
        display_id = "未编号" if image_id is None else f"#{image_id:06d}"
        print(f"\nYOLO果梗置信度图 {display_id} 已保存: {image_path}")
        register_result_image(
            image_id,
            image_path,
            capture_id,
            frame_id,
            build_detection_threshold_summary(detections),
        )
    else:
        print(f"\nYOLO果梗置信度图保存失败: {image_path}")

    if SHOW_DETECTION_THRESHOLD_RESULT:
        cv2.imshow("检测阈值结果", save_image)
        cv2.waitKey(FINAL_RESULT_WAIT_MS)


def save_result_image(image, targets, frame_id, capture_id=0):
    """保存并显示当前定位结果，方便现场调试。"""
    save_image = image.copy()

    for i, target in enumerate(targets):
        cx = int(target["cx"])
        cy = int(target["cy"])

        contour = target.get("contour")
        if contour is not None and len(contour) >= 3:
            cv2.polylines(
                save_image,
                [np.asarray(contour, dtype=np.int32).reshape(-1, 1, 2)],
                True,
                (0, 255, 0),
                1,
            )

        if all(k in target for k in ("x1", "y1", "x2", "y2")):
            cv2.rectangle(
                save_image,
                (int(target["x1"]), int(target["y1"])),
                (int(target["x2"]), int(target["y2"])),
                (255, 0, 0),
                2,
            )

        if all(k in target for k in ("box_cx", "box_cy")):
            cv2.circle(
                save_image,
                (int(target["box_cx"]), int(target["box_cy"])),
                4,
                (0, 255, 255),
                1,
            )

        if all(k in target for k in ("depth_px", "depth_py")):
            cv2.circle(
                save_image,
                (int(target["depth_px"]), int(target["depth_py"])),
                5,
                (255, 255, 0),
                1,
            )

        if all(k in target for k in ("line_p1_uv", "line_p2_uv")):
            p1 = tuple(np.round(target["line_p1_uv"]).astype(int).tolist())
            p2 = tuple(np.round(target["line_p2_uv"]).astype(int).tolist())
            cv2.line(save_image, p1, p2, (255, 0, 255), 2)
            cv2.circle(save_image, p1, 4, (0, 255, 0), -1)
            cv2.circle(save_image, p2, 4, (0, 0, 255), -1)
            cv2.putText(
                save_image,
                "P1",
                p1,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
            )
            cv2.putText(
                save_image,
                "P2",
                p2,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                1,
            )

        cv2.circle(save_image, (cx, cy), 7, (0, 0, 255), -1)

        depth_mode = target.get("depth_mode", "NA")
        cut_mode = target.get("cut_mode", "normal")

        cv2.putText(
            save_image,
            f"ID:{i + 1} {depth_mode} {cut_mode}",
            (cx + 10, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
        )

        cv2.putText(
            save_image,
            f"Zc:{target['Zc']:.1f}cm D:{target.get('depth_m', 0) * 100:.1f}cm",
            (cx + 10, cy + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            save_image,
            f"R:({target['Xr']:.1f},{target['Yr']:.1f},{target['Zr']:.1f})cm",
            (cx + 10, cy + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )

    image_id, image_path = allocate_result_image_path(
        capture_id,
        frame_id,
        image_kind="final",
    )
    encoded_ok, encoded_image = cv2.imencode(".jpg", save_image)
    saved = False

    if encoded_ok:
        try:
            image_path.write_bytes(encoded_image.tobytes())
            saved = True
        except OSError as exc:
            print(f"\n结果图文件写入失败: {exc}")

    if saved:
        display_id = "未编号" if image_id is None else f"#{image_id:06d}"
        print(f"\n结果图 {display_id} 已保存: {image_path}")
        register_result_image(
            image_id,
            image_path,
            capture_id,
            frame_id,
            build_result_image_summary(targets),
        )
    else:
        print(f"\n结果图保存失败: {image_path}")

    if SHOW_FINAL_RESULT:
        cv2.imshow("最终结果", save_image)
        cv2.waitKey(FINAL_RESULT_WAIT_MS)


def get_stable_depth(depth_frame, cx, cy):
    """返回某个像素附近的平均深度，保留给旧版兜底方法使用。"""
    depth_list = []

    for dx in range(-3, 4):
        for dy in range(-3, 4):
            x = cx + dx
            y = cy + dy

            if (
                x < 0
                or y < 0
                or x >= depth_frame.get_width()
                or y >= depth_frame.get_height()
            ):
                continue

            depth = depth_frame.get_distance(x, y)

            if MIN_DEPTH < depth < MAX_DEPTH:
                depth_list.append(depth)

    if len(depth_list) == 0:
        return None

    depth_list.sort()

    if len(depth_list) > 4:
        depth_list = depth_list[1:-1]

    return sum(depth_list) / len(depth_list)


def get_box_depth_sample_points(x1, y1, x2, y2):
    """返回检测框垂直中心线上的 P1/P2/P3 三个采样点。"""
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    return [
        (cx, y1),
        (cx, cy),
        (cx, y2),
    ]


def get_robust_box_depth(depth_frame, x1, y1, x2, y2):
    """旧版深度策略，保留用于后续对比。"""
    points = get_box_depth_sample_points(x1, y1, x2, y2)
    valid_depths = []

    for px, py in points:
        depth = get_stable_depth(depth_frame, px, py)

        if depth is not None:
            valid_depths.append(depth)

    if len(valid_depths) < 2:
        print("P1/P2/P3 有效深度点少于 2 个")
        return None

    valid_depths.sort()
    depth_range = valid_depths[-1] - valid_depths[0]

    if depth_range > DEPTH_CONSISTENCY_THRESHOLD:
        print(
            "P1/P2/P3 深度不稳定: "
            f"范围={depth_range * 100:.1f}cm, "
            f"阈值={DEPTH_CONSISTENCY_THRESHOLD * 100:.1f}cm"
        )
        return None

    return float(np.median(valid_depths))


def get_reliable_depth_point_in_box(
    depth_frame,
    x1,
    y1,
    x2,
    y2,
    depth_reference_m=None,
):
    """在一个检测框内部搜索可靠深度像素。

    参考深度模式优先使用接近当前拍摄视角参考深度的点。
    兜底模式使用距离最近的 30% 有效深度点。
    返回 (px, py, depth_m, mode)，失败时返回 None。
    """
    reference_depth_m = resolve_depth_reference_m(depth_reference_m)
    width = depth_frame.get_width()
    height = depth_frame.get_height()

    x1 = max(0, min(width - 1, int(x1)))
    x2 = max(0, min(width - 1, int(x2)))
    y1 = max(0, min(height - 1, int(y1)))
    y2 = max(0, min(height - 1, int(y2)))

    if x2 <= x1 or y2 <= y1:
        return None

    box_w = x2 - x1
    box_h = y2 - y1

    shrink_x = max(1, int(box_w * 0.15))
    shrink_y = max(1, int(box_h * 0.15))

    sx1 = x1 + shrink_x
    sx2 = x2 - shrink_x
    sy1 = y1 + shrink_y
    sy2 = y2 - shrink_y

    if sx2 <= sx1 or sy2 <= sy1:
        sx1, sx2, sy1, sy2 = x1, x2, y1, y2

    box_cx = (x1 + x2) / 2.0
    box_cy = (y1 + y2) / 2.0

    candidates = []

    step = 1
    if box_w * box_h > 2500:
        step = 2

    for py in range(sy1, sy2 + 1, step):
        for px in range(sx1, sx2 + 1, step):
            depth = depth_frame.get_distance(px, py)

            if MIN_DEPTH < depth < MAX_DEPTH:
                candidates.append((px, py, depth))

    if len(candidates) < REFERENCE_DEPTH_MIN_POINTS:
        print("检测框内有效深度点太少，跳过目标")
        return None

    reference_candidates = [
        p for p in candidates
        if abs(p[2] - reference_depth_m) <= TARGET_DEPTH_BAND_M
    ]

    if len(reference_candidates) >= REFERENCE_DEPTH_MIN_POINTS:
        selected_candidates = reference_candidates
        mode = "REF"
    else:
        depths = np.array([p[2] for p in candidates], dtype=np.float32)
        near_threshold = float(np.percentile(depths, 30))

        selected_candidates = [
            p for p in candidates
            if p[2] <= near_threshold
        ]

        mode = "FALLBACK"

        if len(selected_candidates) < REFERENCE_DEPTH_MIN_POINTS:
            selected_candidates = candidates

    selected_depths = np.array([p[2] for p in selected_candidates], dtype=np.float32)
    median_depth = float(np.median(selected_depths))

    stable_candidates = [
        p for p in selected_candidates
        if abs(p[2] - median_depth) <= DEPTH_STABLE_BAND_M
    ]

    if len(stable_candidates) < 3:
        print("检测框内候选深度不稳定，跳过目标")
        return None

    best_point = None
    best_score = 999999.0

    for px, py, depth in stable_candidates:
        dist_to_center = ((px - box_cx) ** 2 + (py - box_cy) ** 2) ** 0.5
        depth_error_to_median = abs(depth - median_depth)

        if mode == "REF":
            depth_error_to_reference = abs(depth - reference_depth_m)
            score = (
                depth_error_to_reference * 1000.0
                + depth_error_to_median * 500.0
                + dist_to_center * 0.01
            )
        else:
            score = (
                depth_error_to_median * 1000.0
                + dist_to_center * 0.01
            )

        if score < best_score:
            best_score = score
            best_point = (px, py, depth, mode)

    if best_point is None:
        return None

    px, py, depth, mode = best_point

    mode_text = "参考深度" if mode == "REF" else "近距离兜底"

    print(
        "可靠深度点: "
        f"模式={mode_text}, "
        f"有效点={len(candidates)}, "
        f"参考点={len(reference_candidates)}, "
        f"稳定点={len(stable_candidates)}, "
        f"像素=({px},{py}), "
        f"深度={depth * 100:.1f}cm"
    )

    return best_point


def normalize_angle_180(angle):
    """把角度归一化到 [-180, 180] 度。"""
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle


def normalize_pick_angle(angle):
    """选择等价刀口方向，并把末端姿态严格限制在安全半周内。"""
    angle = normalize_angle_180(float(angle))
    limit = float(END_EFFECTOR_ROTATE_LIMIT_DEG)

    while angle > limit:
        angle -= 180.0
    while angle < -limit:
        angle += 180.0

    if abs(angle) < 1e-9:
        return 0.0
    return angle


def normalize_class_name(name):
    """把 YOLO 类别名转成便于比较的小写字符串。"""
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


def class_name_matches(class_name, expected_names):
    normalized = normalize_class_name(class_name)
    return normalized in {normalize_class_name(name) for name in expected_names}


def polygon_to_mask(mask_polygon, image_shape):
    """把 YOLO polygon mask 转成二值 mask。"""
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if mask_polygon is None or len(mask_polygon) < 3:
        return mask

    pts = np.round(mask_polygon).astype(np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def _to_numpy(value):
    """把 PyTorch/Ultralytics 数据安全转换为 NumPy 数组。"""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def make_original_size_mask(result, mask_data, mask_index, image_shape):
    """取得与对齐彩色图严格同尺寸的实例 mask。"""
    height, width = image_shape[:2]
    raw_mask = np.squeeze(np.asarray(mask_data[mask_index], dtype=np.float32))

    if raw_mask.ndim != 2:
        return None

    if raw_mask.shape == (height, width):
        return raw_mask > MASK_THRESHOLD

    polygons = getattr(getattr(result, "masks", None), "xy", None)
    if polygons is not None and mask_index < len(polygons):
        polygon = np.asarray(polygons[mask_index], dtype=np.float32).reshape(-1, 2)
        if len(polygon) >= 3:
            mask = np.zeros((height, width), dtype=np.uint8)
            points = np.round(polygon).astype(np.int32)
            points[:, 0] = np.clip(points[:, 0], 0, width - 1)
            points[:, 1] = np.clip(points[:, 1], 0, height - 1)
            cv2.fillPoly(mask, [points], 1)
            return mask.astype(bool)

    resized = cv2.resize(
        raw_mask,
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized > MASK_THRESHOLD


def depth_frame_to_meters(depth_frame):
    """使用 RealSense 帧自身单位把 Z16 深度图转换成米。"""
    raw_depth = np.asanyarray(depth_frame.get_data())
    if raw_depth.ndim == 3:
        raw_depth = raw_depth[..., 0]
    if raw_depth.ndim != 2:
        raise ValueError(f"深度图维度无效: {raw_depth.shape}")

    depth_m = raw_depth.astype(np.float32)
    if not np.issubdtype(raw_depth.dtype, np.integer):
        return depth_m

    scale = None
    get_units = getattr(depth_frame, "get_units", None)
    if callable(get_units):
        scale = float(get_units())

    if scale is None or not np.isfinite(scale) or scale <= 0:
        get_distance = getattr(depth_frame, "get_distance", None)
        rows, cols = np.nonzero(raw_depth)
        if callable(get_distance) and len(rows) > 0:
            row = int(rows[0])
            col = int(cols[0])
            scale = float(get_distance(col, row)) / float(raw_depth[row, col])

    if scale is None or not np.isfinite(scale) or scale <= 0:
        raise ValueError("无法从 RealSense 深度帧获取有效 depth scale")

    return depth_m * scale


def instance_mask_to_point_cloud(mask, depth_m, intrinsics):
    """把实例 mask 内所有有效深度像素反投影成相机坐标系点云。"""
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != depth_m.shape:
        raise ValueError(
            "RGB mask 与对齐深度图尺寸不一致: "
            f"mask={mask.shape}, depth={depth_m.shape}"
        )

    fx = float(intrinsics.fx)
    fy = float(intrinsics.fy)
    cx = float(intrinsics.ppx)
    cy = float(intrinsics.ppy)
    if fx <= 0 or fy <= 0:
        raise ValueError(f"相机内参无效: fx={fx}, fy={fy}")

    rows, cols = np.where(mask)
    if len(rows) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    z = depth_m[rows, cols].astype(np.float32)
    valid = np.isfinite(z) & (z > MIN_DEPTH) & (z < MAX_DEPTH)
    rows = rows[valid]
    cols = cols[valid]
    z = z[valid]

    if len(z) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    x = (cols.astype(np.float32) - cx) / fx * z
    y = (rows.astype(np.float32) - cy) / fy * z
    return np.column_stack((x, y, z)).astype(np.float32)


def split_point_cloud_depth_clusters(point_cloud):
    """把实例点云按深度拆成多个稳定簇，避免固定选择最近的伪深度。"""
    if point_cloud is None or len(point_cloud) == 0:
        return []

    remaining = np.asarray(point_cloud, dtype=np.float32)
    valid = np.all(np.isfinite(remaining), axis=1)
    remaining = remaining[valid]
    if len(remaining) == 0:
        return []

    trim_low = float(np.clip(POINT_CLOUD_DEPTH_TRIM_LOW_PERCENT, 0.0, 100.0))
    trim_high = float(np.clip(POINT_CLOUD_DEPTH_TRIM_HIGH_PERCENT, trim_low, 100.0))
    z_low, z_high = np.percentile(remaining[:, 2], [trim_low, trim_high])
    remaining = remaining[
        (remaining[:, 2] >= z_low) & (remaining[:, 2] <= z_high)
    ]

    min_points = max(1, int(POINT_CLOUD_CLUSTER_MIN_POINTS))
    depth_band = max(1e-4, float(POINT_CLOUD_DEPTH_BAND_M))
    clusters = []

    # 每次寻找宽度不超过 2*depth_band 的最密集窗口，取出后继续找下一簇。
    while len(remaining) >= min_points:
        order = np.argsort(remaining[:, 2])
        ordered = remaining[order]
        left = 0
        best_left = 0
        best_right = 0

        for right in range(len(ordered)):
            while ordered[right, 2] - ordered[left, 2] > 2.0 * depth_band:
                left += 1
            if right - left > best_right - best_left:
                best_left = left
                best_right = right

        dense_window = ordered[best_left:best_right + 1]
        if len(dense_window) < min_points:
            break

        center_depth = float(np.median(dense_window[:, 2]))
        selected = np.abs(remaining[:, 2] - center_depth) <= depth_band
        cluster = remaining[selected]
        if len(cluster) < min_points:
            cluster = dense_window
            selected = np.zeros(len(remaining), dtype=bool)
            selected[order[best_left:best_right + 1]] = True

        clusters.append(cluster.astype(np.float32))
        remaining = remaining[~selected]

    return sorted(clusters, key=lambda cluster: float(np.median(cluster[:, 2])))


def estimate_scene_depth_reference(point_cloud_sources):
    """用主茎和同画面果梗的候选簇投票，估计植物所在深度平面。"""
    source_clusters = []
    for point_cloud in point_cloud_sources:
        summaries = []
        for cluster in split_point_cloud_depth_clusters(point_cloud):
            summaries.append(
                {
                    "depth_m": float(np.median(cluster[:, 2])),
                    "point_count": int(len(cluster)),
                }
            )
        source_clusters.append(summaries)

    candidates = [
        summary
        for summaries in source_clusters
        for summary in summaries
    ]
    if not candidates:
        return None, 0

    consensus_band = max(0.0, float(POINT_CLOUD_SCENE_CONSENSUS_BAND_M))
    best = None
    for candidate in candidates:
        candidate_depth = candidate["depth_m"]
        matched_depths = []
        total_points = 0
        preferred_source_matched = False

        for source_index, summaries in enumerate(source_clusters):
            if not summaries:
                continue
            nearest = min(
                summaries,
                key=lambda summary: abs(summary["depth_m"] - candidate_depth),
            )
            if abs(nearest["depth_m"] - candidate_depth) <= consensus_band:
                matched_depths.append(nearest["depth_m"])
                total_points += nearest["point_count"]
                preferred_source_matched |= source_index == 0

        if not matched_depths:
            continue

        # 先看有多少个独立实例支持，再优先主茎支持；完全平票时选较近平面。
        score = (
            len(matched_depths),
            int(preferred_source_matched),
            -candidate_depth,
            total_points,
        )
        if best is None or score > best[0]:
            best = (score, float(np.median(matched_depths)), len(matched_depths))

    if best is None:
        return None, 0
    return best[1], best[2]


def filter_point_cloud(point_cloud, scene_depth_reference_m=None):
    """按场景共识选择果梗自己的深度簇，再去除三维离群点。"""
    clusters = split_point_cloud_depth_clusters(point_cloud)
    if not clusters:
        return np.zeros((0, 3), dtype=np.float32)

    if scene_depth_reference_m is None:
        point_cloud = max(clusters, key=len)
    else:
        max_error = max(0.0, float(POINT_CLOUD_SCENE_MAX_ERROR_M))
        eligible_clusters = [
            cluster
            for cluster in clusters
            if abs(float(np.median(cluster[:, 2])) - scene_depth_reference_m)
            <= max_error
        ]
        if not eligible_clusters:
            return np.zeros((0, 3), dtype=np.float32)
        point_cloud = min(
            eligible_clusters,
            key=lambda cluster: (
                abs(float(np.median(cluster[:, 2])) - scene_depth_reference_m),
                -len(cluster),
            ),
        )

    if len(point_cloud) < MIN_PCA_POINTS:
        return point_cloud.astype(np.float32)

    center = np.mean(point_cloud, axis=0, keepdims=True)
    distances = np.linalg.norm(point_cloud - center, axis=1)
    keep_percent = float(np.clip(POINT_CLOUD_DISTANCE_KEEP_PERCENT, 0.0, 100.0))
    distance_high = float(np.percentile(distances, keep_percent))
    return point_cloud[distances <= distance_high].astype(np.float32)


def fit_point_cloud_line(point_cloud):
    """用 SVD/PCA 拟合三维果梗主轴和稳健 P1/P2 端点。"""
    if point_cloud is None or len(point_cloud) < MIN_PCA_POINTS:
        return None

    point_cloud = np.asarray(point_cloud, dtype=np.float32)
    center = np.mean(point_cloud, axis=0).astype(np.float32)
    centered = point_cloud - center[None, :]

    try:
        _u, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None

    if len(vh) == 0 or not np.all(np.isfinite(vh[0])):
        return None

    second_value = float(singular_values[1]) if len(singular_values) > 1 else 0.0
    linearity = float(singular_values[0]) / (second_value + 1e-12)
    if not np.isfinite(linearity) or linearity < MIN_PCA_LINEARITY:
        return None

    main_axis = vh[0].astype(np.float32)
    axis_norm = float(np.linalg.norm(main_axis))
    if axis_norm <= 1e-12:
        return None
    main_axis /= axis_norm

    projections = centered @ main_axis
    t_min, t_max = np.percentile(projections, [5, 95])
    p1 = (center + float(t_min) * main_axis).astype(np.float32)
    p2 = (center + float(t_max) * main_axis).astype(np.float32)
    line_length_cm = float(np.linalg.norm(p2 - p1)) * 100.0
    if not (MIN_PCA_LINE_LENGTH_CM <= line_length_cm <= MAX_PCA_LINE_LENGTH_CM):
        return None

    if p1[1] > p2[1] or (
        abs(float(p1[1] - p2[1])) < 1e-6 and p1[0] > p2[0]
    ):
        p1, p2 = p2, p1
        main_axis = (p2 - p1).astype(np.float32)
        main_axis /= float(np.linalg.norm(main_axis)) + 1e-12

    if not (
        np.all(np.isfinite(center))
        and np.all(np.isfinite(p1))
        and np.all(np.isfinite(p2))
    ):
        return None

    return {
        "center": center,
        "p1": p1,
        "p2": p2,
        "main_axis": main_axis,
        "point_count": int(len(point_cloud)),
        "linearity": linearity,
        "line_length_cm": line_length_cm,
    }


def project_camera_point_to_pixel(point, intrinsics):
    """把米制相机三维点投影回对齐后的彩色图。"""
    point = np.asarray(point, dtype=np.float32).reshape(3)
    z = float(point[2])
    if not np.isfinite(z) or z <= 0:
        return None

    u = float(point[0]) * float(intrinsics.fx) / z + float(intrinsics.ppx)
    v = float(point[1]) * float(intrinsics.fy) / z + float(intrinsics.ppy)
    if not np.isfinite(u) or not np.isfinite(v):
        return None
    return np.array([u, v], dtype=np.float32)


def prepare_cut_depth_masks(mask_polygon, image_shape):
    """生成原始果梗 mask 和用于深度搜索的自适应内部 mask。"""
    original_mask = polygon_to_mask(mask_polygon, image_shape)
    search_mask = original_mask.copy()
    info = {
        "erosion_applied": False,
        "max_half_width_px": 0.0,
        "original_points": int(np.count_nonzero(original_mask)),
        "search_points": int(np.count_nonzero(search_mask)),
    }

    if not ENABLE_CUT_MASK_ADAPTIVE_EROSION or info["original_points"] == 0:
        return original_mask, search_mask, info

    distance_map = cv2.distanceTransform(original_mask, cv2.DIST_L2, 5)
    max_half_width = float(np.max(distance_map))
    info["max_half_width_px"] = max_half_width

    if max_half_width < float(CUT_MASK_EROSION_MIN_HALF_WIDTH_PX):
        return original_mask, search_mask, info

    kernel_size = max(3, int(CUT_MASK_EROSION_KERNEL_SIZE))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    eroded_mask = cv2.erode(
        original_mask,
        kernel,
        iterations=max(1, int(CUT_MASK_EROSION_ITERATIONS)),
    )
    remaining_points = int(np.count_nonzero(eroded_mask))

    if remaining_points < max(1, int(CUT_MASK_EROSION_MIN_REMAINING_POINTS)):
        return original_mask, search_mask, info

    search_mask = eroded_mask
    info["erosion_applied"] = True
    info["search_points"] = remaining_points
    return original_mask, search_mask, info


def filter_connected_cut_depth_candidates(candidates):
    """只保留形成空间连通且深度相近簇的 mask 内候选。"""
    in_mask_candidates = {
        (candidate[1], candidate[2]): candidate
        for candidate in candidates
        if candidate[4]
    }
    required_points = max(1, int(CUT_DEPTH_CONNECTED_MIN_POINTS))
    depth_band = max(0.0, float(CUT_DEPTH_CLUSTER_BAND_M))
    remaining = set(in_mask_candidates)
    qualified_pixels = set()
    largest_cluster = 0

    while remaining:
        start = remaining.pop()
        cluster = [start]
        stack = [start]

        while stack:
            current = stack.pop()
            current_depth = in_mask_candidates[current][3]

            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    neighbor = (current[0] + dx, current[1] + dy)
                    if neighbor not in remaining:
                        continue
                    neighbor_depth = in_mask_candidates[neighbor][3]
                    if abs(neighbor_depth - current_depth) > depth_band:
                        continue
                    remaining.remove(neighbor)
                    cluster.append(neighbor)
                    stack.append(neighbor)

        largest_cluster = max(largest_cluster, len(cluster))
        if len(cluster) >= required_points:
            qualified_pixels.update(cluster)

    filtered = [
        candidate for candidate in candidates
        if not candidate[4] or (candidate[1], candidate[2]) in qualified_pixels
    ]
    return filtered, largest_cluster


def save_cut_depth_diagnostic(
    depth_frame,
    original_mask,
    search_mask,
    cut_px,
    cut_py,
    depth_result,
    capture_id,
    frame_id,
    target_index,
    mask_info,
):
    """保存果梗 mask 内原始深度热力图和可复算的压缩数值。"""
    if not SAVE_CUT_DEPTH_DIAGNOSTICS:
        return

    image_path, data_path = allocate_depth_diagnostic_paths(
        capture_id,
        frame_id,
        target_index,
    )
    if image_path is None or data_path is None:
        return

    try:
        raw_depth = np.asanyarray(depth_frame.get_data())
        depth_scale = float(depth_frame.get_units())
        depth_m = raw_depth.astype(np.float32) * depth_scale
        height = min(depth_m.shape[0], original_mask.shape[0])
        width = min(depth_m.shape[1], original_mask.shape[1])
        depth_m = depth_m[:height, :width]
        original_mask = original_mask[:height, :width]
        search_mask = search_mask[:height, :width]
        mask_pixels = original_mask > 0
        valid_pixels = (
            mask_pixels
            & (depth_m > MIN_DEPTH)
            & (depth_m < MAX_DEPTH)
        )
        valid_depths = depth_m[valid_pixels]

        ys, xs = np.nonzero(mask_pixels)
        if len(xs) == 0:
            return

        padding = 20
        x0 = max(0, int(xs.min()) - padding)
        x1 = min(width, int(xs.max()) + padding + 1)
        y0 = max(0, int(ys.min()) - padding)
        y1 = min(height, int(ys.max()) + padding + 1)

        normalized = np.zeros((height, width), dtype=np.uint8)
        if np.any(valid_pixels):
            clipped = np.clip(depth_m, MIN_DEPTH, MAX_DEPTH)
            normalized[valid_pixels] = np.round(
                (clipped[valid_pixels] - MIN_DEPTH)
                / max(1e-6, MAX_DEPTH - MIN_DEPTH)
                * 255.0
            ).astype(np.uint8)

        color_map_id = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
        heatmap = cv2.applyColorMap(normalized, color_map_id)
        heatmap[~mask_pixels] = 0
        heatmap[mask_pixels & ~valid_pixels] = (45, 45, 45)

        original_contours, _ = cv2.findContours(
            original_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        search_contours, _ = cv2.findContours(
            search_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(heatmap, original_contours, -1, (0, 255, 0), 1)
        cv2.drawContours(heatmap, search_contours, -1, (255, 255, 0), 1)
        cv2.circle(
            heatmap,
            (int(round(cut_px)), int(round(cut_py))),
            3,
            (0, 0, 255),
            -1,
        )

        selected_values = None
        if depth_result is not None:
            selected_values = (
                int(depth_result[0]),
                int(depth_result[1]),
                float(depth_result[2]),
                str(depth_result[3]),
            )
            cv2.circle(
                heatmap,
                (selected_values[0], selected_values[1]),
                4,
                (255, 255, 255),
                1,
            )

        crop = heatmap[y0:y1, x0:x1]
        scale = max(1, min(6, int(round(360 / max(1, max(crop.shape[:2]))))))
        crop = cv2.resize(
            crop,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_NEAREST,
        )
        header = np.zeros((90, max(520, crop.shape[1]), 3), dtype=np.uint8)
        stats_text = "valid=0"
        if len(valid_depths) > 0:
            stats_text = (
                f"valid={len(valid_depths)} min={valid_depths.min() * 100:.1f}cm "
                f"median={np.median(valid_depths) * 100:.1f}cm "
                f"max={valid_depths.max() * 100:.1f}cm"
            )
        mode_text = "selected=NONE"
        if selected_values is not None:
            mode_text = (
                f"selected={selected_values[3]} "
                f"depth={selected_values[2] * 100:.1f}cm"
            )
        erosion_text = (
            f"erosion={mask_info['erosion_applied']} "
            f"half_width={mask_info['max_half_width_px']:.1f}px "
            f"mask={mask_info['original_points']}->{mask_info['search_points']}"
        )
        cv2.putText(header, stats_text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
        cv2.putText(header, mode_text, (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
        cv2.putText(header, erosion_text, (8, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
        if crop.shape[1] < header.shape[1]:
            crop = cv2.copyMakeBorder(
                crop,
                0,
                0,
                0,
                header.shape[1] - crop.shape[1],
                cv2.BORDER_CONSTANT,
            )
        diagnostic_image = np.vstack((header, crop))
        cv2.imwrite(str(image_path), diagnostic_image)

        selected_array = np.array(
            selected_values[:3] if selected_values is not None else [-1, -1, np.nan],
            dtype=np.float32,
        )
        np.savez_compressed(
            str(data_path),
            depth_m=depth_m[y0:y1, x0:x1],
            original_mask=original_mask[y0:y1, x0:x1],
            search_mask=search_mask[y0:y1, x0:x1],
            crop_origin=np.array([x0, y0], dtype=np.int32),
            cut_point=np.array([cut_px, cut_py], dtype=np.float32),
            selected_point_depth=selected_array,
            selected_mode=np.array(
                selected_values[3] if selected_values is not None else "NONE"
            ),
            depth_scale=np.array(depth_scale, dtype=np.float32),
        )
        print(f"果梗深度诊断已保存: {image_path.name}")
    except Exception as exc:
        print(f"保存果梗深度诊断失败，检测流程继续: {exc}")


def get_mask_points(mask):
    coords = cv2.findNonZero(mask)
    if coords is None:
        return None
    return coords[:, 0, :].astype(np.float32)


def resolve_depth_reference_m(depth_reference_m=None):
    """返回当前拍摄视角使用的相机相对参考深度。"""
    if depth_reference_m is None:
        return float(TARGET_DEPTH_REFERENCE_M)

    return float(depth_reference_m)


def is_background_depth(depth, depth_reference_m=None):
    """判断某个深度是否接近已知背景墙深度。"""
    if BACKGROUND_DEPTH_M <= 0 or BACKGROUND_DEPTH_BAND_M <= 0:
        return False

    reference_depth_m = resolve_depth_reference_m(depth_reference_m)
    camera_forward_shift_m = TARGET_DEPTH_REFERENCE_M - reference_depth_m
    current_background_depth_m = BACKGROUND_DEPTH_M - camera_forward_shift_m

    if current_background_depth_m <= 0:
        return False

    return abs(depth - current_background_depth_m) <= BACKGROUND_DEPTH_BAND_M


def score_cut_depth_candidate(
    px,
    py,
    depth,
    cut_px,
    cut_py,
    in_mask,
    depth_reference_m=None,
):
    """给剪切点附近的深度候选打分，分数越低越可信。"""
    reference_depth_m = resolve_depth_reference_m(depth_reference_m)
    pixel_distance = ((px - cut_px) ** 2 + (py - cut_py) ** 2) ** 0.5
    reference_error = abs(depth - reference_depth_m)
    reference_penalty = max(0.0, reference_error - TARGET_DEPTH_BAND_M)

    score = (
        pixel_distance * CUT_DEPTH_DISTANCE_WEIGHT
        + reference_penalty * CUT_DEPTH_REFERENCE_WEIGHT
    )

    if not in_mask:
        score += CUT_DEPTH_OUTSIDE_MASK_PENALTY

    if is_background_depth(depth, reference_depth_m):
        score += CUT_DEPTH_BACKGROUND_PENALTY

    return score


def measure_mask_background_depth_contrast(
    depth_frame,
    search_mask,
    cut_px,
    cut_py,
    candidate_depth_m,
    search_radius_px,
    ring_width_px,
    min_ring_points,
):
    """比较果梗 mask 内候选与 mask 外圈背景的深度差。"""
    if search_mask is None:
        return None

    ring_width_px = max(1, int(ring_width_px))
    min_ring_points = max(1, int(min_ring_points))
    kernel_size = ring_width_px * 2 + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    dilated_mask = cv2.dilate(search_mask, kernel)
    ring_mask = (dilated_mask > 0) & (search_mask == 0)

    mask_h, mask_w = search_mask.shape[:2]
    width = min(depth_frame.get_width(), mask_w)
    height = min(depth_frame.get_height(), mask_h)
    local_radius = max(1, int(search_radius_px)) + ring_width_px
    center_x = int(round(cut_px))
    center_y = int(round(cut_py))
    x_start = max(0, center_x - local_radius)
    x_end = min(width - 1, center_x + local_radius)
    y_start = max(0, center_y - local_radius)
    y_end = min(height - 1, center_y + local_radius)
    ring_depths = []

    for py in range(y_start, y_end + 1):
        for px in range(x_start, x_end + 1):
            if not ring_mask[py, px]:
                continue

            depth = depth_frame.get_distance(px, py)
            if MIN_DEPTH < depth < MAX_DEPTH:
                ring_depths.append(depth)

    if len(ring_depths) < min_ring_points:
        return None

    background_depth_m = float(np.median(ring_depths))
    return {
        "background_depth_m": background_depth_m,
        "contrast_m": background_depth_m - float(candidate_depth_m),
        "point_count": len(ring_depths),
    }


def estimate_depth_from_main_stem(
    depth_frame,
    nearest_main_point,
    main_stem_mask,
    depth_reference_m=None,
    enabled=None,
):
    """果梗深度不可靠时，借用最近主茎局部的稳定深度。"""
    use_main_stem_fallback = (
        USE_MAIN_STEM_DEPTH_FALLBACK
        if enabled is None
        else bool(enabled)
    )
    if not use_main_stem_fallback:
        return None

    if nearest_main_point is None or main_stem_mask is None:
        return None

    reference_depth_m = resolve_depth_reference_m(depth_reference_m)
    mask_h, mask_w = main_stem_mask.shape[:2]
    width = min(depth_frame.get_width(), mask_w)
    height = min(depth_frame.get_height(), mask_h)

    center_x = float(nearest_main_point[0])
    center_y = float(nearest_main_point[1])

    if not (0 <= center_x < width and 0 <= center_y < height):
        return None

    start_radius = max(1, int(MAIN_STEM_DEPTH_SEARCH_RADIUS_PX))
    max_radius = max(start_radius, int(MAIN_STEM_DEPTH_MAX_RADIUS_PX))
    step = max(1, int(MAIN_STEM_DEPTH_RADIUS_STEP_PX))
    min_points = max(1, int(MAIN_STEM_DEPTH_MIN_POINTS))
    stable_band = max(0.0, float(MAIN_STEM_DEPTH_STABLE_BAND_M))
    reference_limit = max(0.0, float(MAIN_STEM_DEPTH_REFERENCE_MAX_ERROR_M))

    for radius in range(start_radius, max_radius + 1, step):
        x_start = max(0, int(round(center_x)) - radius)
        x_end = min(width - 1, int(round(center_x)) + radius)
        y_start = max(0, int(round(center_y)) - radius)
        y_end = min(height - 1, int(round(center_y)) + radius)

        candidates = []

        for py in range(y_start, y_end + 1):
            for px in range(x_start, x_end + 1):
                pixel_distance = ((px - center_x) ** 2 + (py - center_y) ** 2) ** 0.5
                if pixel_distance > radius:
                    continue

                if main_stem_mask[py, px] == 0:
                    continue

                depth = depth_frame.get_distance(px, py)
                if not (MIN_DEPTH < depth < MAX_DEPTH):
                    continue

                if is_background_depth(depth, reference_depth_m):
                    continue

                if (
                    reference_limit > 0
                    and abs(depth - reference_depth_m) > reference_limit
                ):
                    continue

                candidates.append((px, py, depth, pixel_distance))

        if len(candidates) < min_points:
            continue

        depths = np.array([candidate[2] for candidate in candidates], dtype=np.float32)
        median_depth = float(np.median(depths))

        stable_candidates = candidates
        if stable_band > 0:
            stable_candidates = [
                candidate for candidate in candidates
                if abs(candidate[2] - median_depth) <= stable_band
            ]

        if len(stable_candidates) < min_points:
            continue

        stable_depths = np.array(
            [candidate[2] for candidate in stable_candidates],
            dtype=np.float32,
        )
        depth = float(np.median(stable_depths))
        px, py, _sample_depth, _pixel_distance = min(
            stable_candidates,
            key=lambda candidate: (
                abs(candidate[2] - depth),
                candidate[3],
            ),
        )

        print(
            "使用最近主茎局部深度兜底: "
            f"像素=({px},{py}), "
            f"深度={depth * 100:.1f}cm, "
            f"半径={radius}px, "
            f"有效点={len(stable_candidates)}"
        )
        return px, py, depth, "MAIN_STEM"

    return None


def estimate_depth_at_cut_point(
    depth_frame,
    cut_px,
    cut_py,
    mask_polygon,
    image_shape,
    nearest_main_point=None,
    main_stem_mask=None,
    depth_reference_m=None,
    reference_fallback_trigger_m=None,
    candidate_reference_max_error_m=None,
    use_main_stem_depth_fallback=None,
    mask_background_ring_width_px=None,
    mask_background_ring_min_points=None,
    min_mask_background_depth_contrast_m=None,
):
    """围绕最终剪切点估计深度；找不到可信深度时可用人工参考深度兜底。"""
    reference_depth_m = resolve_depth_reference_m(depth_reference_m)
    fallback_trigger_m = (
        CUT_DEPTH_REFERENCE_FALLBACK_TRIGGER_M
        if reference_fallback_trigger_m is None
        else max(0.0, float(reference_fallback_trigger_m))
    )
    candidate_reference_limit_m = (
        None
        if candidate_reference_max_error_m is None
        else max(0.0, float(candidate_reference_max_error_m))
    )
    width = depth_frame.get_width()
    height = depth_frame.get_height()
    cut_px = float(cut_px)
    cut_py = float(cut_py)

    original_mask = None
    search_mask = None
    mask_info = None
    if mask_polygon is not None and image_shape is not None:
        original_mask, search_mask, mask_info = prepare_cut_depth_masks(
            mask_polygon,
            image_shape,
        )
        if mask_info["erosion_applied"]:
            print(
                "果梗 mask 已轻度腐蚀后搜索深度: "
                f"半宽={mask_info['max_half_width_px']:.1f}px, "
                f"像素={mask_info['original_points']}->{mask_info['search_points']}"
            )

    start_radius = max(1, int(CUT_DEPTH_SEARCH_RADIUS_PX))
    max_radius = max(start_radius, int(CUT_DEPTH_MAX_RADIUS_PX))
    step = max(1, int(CUT_DEPTH_RADIUS_STEP_PX))
    min_points = max(1, int(CUT_DEPTH_MIN_POINTS))

    best_candidate = None
    best_score = None
    best_radius = None
    best_count = 0
    reference_rejected_pixels = set()
    nearest_rejected_depth = None
    largest_mask_cluster_seen = 0

    for radius in range(start_radius, max_radius + 1, step):
        x_start = max(0, int(round(cut_px)) - radius)
        x_end = min(width - 1, int(round(cut_px)) + radius)
        y_start = max(0, int(round(cut_py)) - radius)
        y_end = min(height - 1, int(round(cut_py)) + radius)

        candidates = []

        for py in range(y_start, y_end + 1):
            for px in range(x_start, x_end + 1):
                pixel_distance = ((px - cut_px) ** 2 + (py - cut_py) ** 2) ** 0.5
                if pixel_distance > radius:
                    continue

                depth = depth_frame.get_distance(px, py)
                if not (MIN_DEPTH < depth < MAX_DEPTH):
                    continue

                if (
                    candidate_reference_limit_m is not None
                    and abs(depth - reference_depth_m)
                    > candidate_reference_limit_m
                ):
                    reference_rejected_pixels.add((px, py))
                    if (
                        nearest_rejected_depth is None
                        or abs(depth - reference_depth_m)
                        < abs(nearest_rejected_depth - reference_depth_m)
                    ):
                        nearest_rejected_depth = depth
                    continue

                in_mask = bool(search_mask is not None and search_mask[py, px] > 0)
                score = score_cut_depth_candidate(
                    px,
                    py,
                    depth,
                    cut_px,
                    cut_py,
                    in_mask,
                    reference_depth_m,
                )
                candidates.append((score, px, py, depth, in_mask))

        candidates, largest_cluster = filter_connected_cut_depth_candidates(
            candidates,
        )
        largest_mask_cluster_seen = max(
            largest_mask_cluster_seen,
            largest_cluster,
        )

        if len(candidates) < min_points:
            continue

        candidates.sort(key=lambda item: item[0])
        best_score, px, py, depth, in_mask = candidates[0]
        best_candidate = (px, py, depth, "CUT_MASK" if in_mask else "CUT_NEAR")
        best_radius = radius
        best_count = len(candidates)
        break

    if best_candidate is not None:
        px, py, depth, mode = best_candidate
        minimum_contrast_m = max(
            0.0,
            float(min_mask_background_depth_contrast_m),
        )

        if mode == "CUT_MASK" and minimum_contrast_m > 0:
            contrast_result = measure_mask_background_depth_contrast(
                depth_frame,
                original_mask,
                cut_px,
                cut_py,
                depth,
                best_radius,
                mask_background_ring_width_px,
                mask_background_ring_min_points,
            )

            if contrast_result is None:
                print(
                    "果梗 mask 外圈有效深度不足，"
                    "当前深度仅标记为未验证前景"
                )
            else:
                print(
                    "果梗 mask 内外深度对比: "
                    f"内部={depth * 100:.1f}cm, "
                    f"外圈={contrast_result['background_depth_m'] * 100:.1f}cm, "
                    f"前景差={contrast_result['contrast_m'] * 100:.1f}cm, "
                    f"外圈点={contrast_result['point_count']}"
                )

                if contrast_result["contrast_m"] < minimum_contrast_m:
                    print("mask 内外深度几乎相同，判定为背景深度泄漏")
                    best_candidate = None
                else:
                    best_candidate = (px, py, depth, "CUT_MASK_FG")
                    print("mask 内存在连续深度簇且明显近于外圈，已验证为果梗前景")

    if best_candidate is not None:
        px, py, depth, mode = best_candidate
        reference_error = abs(depth - reference_depth_m)
        should_use_reference = (
            mode != "CUT_MASK_FG"
            and
            ALLOW_REFERENCE_DEPTH_FALLBACK
            and (
                reference_error > fallback_trigger_m
                or is_background_depth(depth, reference_depth_m)
            )
        )

        if should_use_reference:
            main_stem_depth = estimate_depth_from_main_stem(
                depth_frame,
                nearest_main_point,
                main_stem_mask,
                reference_depth_m,
                enabled=use_main_stem_depth_fallback,
            )
            if main_stem_depth is not None:
                print(
                    "剪切点附近深度疑似背景，改用最近主茎深度兜底: "
                    f"候选={depth * 100:.1f}cm, "
                    f"当前视角参考={reference_depth_m * 100:.1f}cm"
                )
                return main_stem_depth

            print(
                "剪切点附近深度疑似背景，且没有可用主茎深度，改用人工参考深度兜底: "
                f"候选={depth * 100:.1f}cm, "
                f"当前视角参考={reference_depth_m * 100:.1f}cm, "
                f"半径={best_radius}px, "
                f"候选点={best_count}"
            )
            return int(round(cut_px)), int(round(cut_py)), reference_depth_m, "REF_FALLBACK"

        if mode != "CUT_MASK_FG":
            main_stem_depth = estimate_depth_from_main_stem(
                depth_frame,
                nearest_main_point,
                main_stem_mask,
                reference_depth_m,
                enabled=use_main_stem_depth_fallback,
            )
            if main_stem_depth is not None:
                print(
                    "果梗候选数值可用但尚未验证为独立前景，"
                    "优先采用主茎深度"
                )
                return main_stem_depth

            print(
                "果梗候选尚未验证为独立前景，且没有可用主茎深度，"
                "保留为未验证候选"
            )

        print(
            "剪切点附近深度: "
            f"模式={mode}, "
            f"像素=({px},{py}), "
            f"深度={depth * 100:.1f}cm, "
            f"半径={best_radius}px, "
            f"候选点={best_count}, "
            f"评分={best_score:.1f}"
        )
        return best_candidate

    if nearest_rejected_depth is not None:
        print(
            "剪切点附近候选均偏离当前视角参考深度: "
            f"最近候选={nearest_rejected_depth * 100:.1f}cm, "
            f"参考={reference_depth_m * 100:.1f}cm, "
            f"允许误差={candidate_reference_limit_m * 100:.1f}cm, "
            f"排除像素={len(reference_rejected_pixels)}"
        )

    if (
        search_mask is not None
        and 0 < largest_mask_cluster_seen < max(
            1,
            int(CUT_DEPTH_CONNECTED_MIN_POINTS),
        )
    ):
        print(
            "果梗 mask 内没有达到要求的连续深度簇: "
            f"最大簇={largest_mask_cluster_seen}, "
            f"要求={max(1, int(CUT_DEPTH_CONNECTED_MIN_POINTS))}"
        )

    main_stem_depth = estimate_depth_from_main_stem(
        depth_frame,
        nearest_main_point,
        main_stem_mask,
        reference_depth_m,
        enabled=use_main_stem_depth_fallback,
    )
    if main_stem_depth is not None:
        print("剪切点附近没有有效深度，改用最近主茎深度兜底")
        return main_stem_depth

    if ALLOW_REFERENCE_DEPTH_FALLBACK:
        print(
            "剪切点附近没有有效深度，且没有可用主茎深度，使用人工参考深度兜底: "
            f"当前视角参考={reference_depth_m * 100:.1f}cm"
        )
        return int(round(cut_px)), int(round(cut_py)), reference_depth_m, "REF_FALLBACK"

    return None


def unit_vector(vector):
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-6:
        return None
    return vector / norm


def get_stem_tangent(points):
    """用 PCA 拟合果梗局部主方向。"""
    if points is None or len(points) < 10:
        return None

    _mean, eigenvectors = cv2.PCACompute(points.astype(np.float32), mean=None)
    vx, vy = eigenvectors[0]

    if vy < 0:
        vx = -vx
        vy = -vy

    return unit_vector((vx, vy))


def static_vector_to_rotate_angle(static_vector):
    """根据图像里的静刀方向计算旋转角。

    0 度时静刀在图像右侧。图像坐标系 y 轴向下，
    atan2(y, x) 的正方向是顺时针；实际电机方向用配置符号修正。
    """
    sx, sy = static_vector
    clockwise_angle = math.degrees(math.atan2(sy, sx))
    return normalize_pick_angle(clockwise_angle * END_EFFECTOR_CLOCKWISE_SIGN)


def choose_static_blade_vector(tangent, toward_main=None, contact_blade="static"):
    """选择静刀方向。

    contact_blade 为 static 时，让静刀侧朝主茎；
    contact_blade 为 moving 时，让动刀侧朝主茎。
    没有主茎参考时选旋转幅度更小的一侧。
    """
    tx, ty = tangent
    candidates = [
        unit_vector((ty, -tx)),
        unit_vector((-ty, tx)),
    ]
    candidates = [candidate for candidate in candidates if candidate is not None]

    if not candidates:
        return None

    if toward_main is not None:
        toward_main = unit_vector(toward_main)
        if toward_main is not None:
            def contact_score(candidate):
                contact_vector = candidate
                if contact_blade == "moving":
                    contact_vector = -candidate
                return float(np.dot(contact_vector, toward_main))

            return max(candidates, key=contact_score)

    def score(candidate):
        angle = static_vector_to_rotate_angle(candidate)
        clockwise_bonus = 0 if angle <= 0 else 0.01
        return abs(angle) + clockwise_bonus

    return min(candidates, key=score)


def get_fruit_side(cut_point, nearest_main_point):
    """判断果梗剪切点在主茎左侧还是右侧。"""
    if nearest_main_point is None:
        return "unknown"

    if cut_point[0] < nearest_main_point[0]:
        return "left"

    return "right"


def choose_contact_blade(fruit_side):
    """根据果梗相对主茎左右侧选择贴近刀片。"""
    if fruit_side == "left":
        return "static" if FRUIT_LEFT_USES_STATIC_BLADE else "moving"

    if fruit_side == "right":
        return "moving" if FRUIT_LEFT_USES_STATIC_BLADE else "static"

    return "static"


def get_contact_blade_offset(contact_blade):
    """返回导入时从旋转中心偏向所选刀片的距离。"""
    if contact_blade == "moving":
        return MOVING_BLADE_TIP_OFFSET_CM * GUIDE_BLADE_CONTACT_RATIO

    return STATIC_BLADE_TIP_OFFSET_CM * GUIDE_BLADE_CONTACT_RATIO


def get_cut_region_points(points, tangent):
    """取 P1/P2 之间的果梗有效剪切区域。"""
    projections = points @ tangent
    start_ratio = max(0.0, min(1.0, STEM_CUT_REGION_START_RATIO))
    end_ratio = max(start_ratio, min(1.0, STEM_CUT_REGION_END_RATIO))
    low = float(np.quantile(projections, start_ratio))
    high = float(np.quantile(projections, end_ratio))
    region_points = points[(projections >= low) & (projections <= high)]

    if len(region_points) == 0:
        return points

    return region_points


def find_center_cut_point(stem_points, main_stem_points):
    """在果梗有效区域中取中心剪切点，并返回离它最近的主茎点。"""
    if stem_points is None or len(stem_points) == 0:
        return None, None, None

    center = np.mean(stem_points, axis=0)
    stem_distances = np.sum((stem_points - center) ** 2, axis=1)
    cut_point = stem_points[int(np.argmin(stem_distances))]

    if main_stem_points is None or len(main_stem_points) == 0:
        return cut_point, None, None

    main_points = main_stem_points
    if len(main_points) > 6000:
        step = max(1, len(main_points) // 6000)
        main_points = main_points[::step]

    main_distances = np.sum((main_points - cut_point) ** 2, axis=1)
    nearest_index = int(np.argmin(main_distances))
    nearest_main_point = main_points[nearest_index]

    return cut_point, nearest_main_point, float(main_distances[nearest_index] ** 0.5)


def pixel_vector_distance_cm(pixel_vector, depth_m, depth_intrin):
    """把图像像素向量粗略换算成当前深度下的厘米距离。"""
    if pixel_vector is None or depth_m is None:
        return None

    dx, dy = pixel_vector
    x_cm = dx * depth_m * 100.0 / depth_intrin.fx
    y_cm = dy * depth_m * 100.0 / depth_intrin.fy
    return float((x_cm * x_cm + y_cm * y_cm) ** 0.5)


def pixel_to_robot_coordinate(depth_intrin, eye, te, px, py, depth):
    point_3d = rs.rs2_deproject_pixel_to_point(
        depth_intrin,
        [float(px), float(py)],
        float(depth),
    )
    cam_target = np.array([
        [point_3d[0] * 100],
        [point_3d[1] * 100],
        [point_3d[2] * 100],
    ])
    robot_target = eye.coordinate(cam_target, te)
    return cam_target, robot_target


def should_recheck_with_reference_depth(xr):
    """X 轴越界时，把当前深度视为可疑并触发参考深度重算。"""
    margin = max(0.0, float(X_RANGE_REFERENCE_RECHECK_MARGIN_CM))
    return xr > X_MAX + margin or xr < X_MIN - margin


def reproject_target_with_reference_depth(
    depth_intrin,
    eye,
    te,
    cx,
    cy,
    xr,
    yr,
    zr,
    cut_xr,
    cut_yr,
    cut_zr,
    depth,
    depth_mode,
    depth_reference_m=None,
    enabled=None,
):
    """用人工参考深度重算剪切点坐标，并保留导入策略产生的小偏置。"""
    use_reference_recheck = (
        USE_REFERENCE_DEPTH_WHEN_X_OUT_OF_RANGE
        if enabled is None
        else bool(enabled)
    )
    if not use_reference_recheck:
        return None

    if not should_recheck_with_reference_depth(xr):
        return None

    reference_depth_m = resolve_depth_reference_m(depth_reference_m)
    cam_target, robot_target = pixel_to_robot_coordinate(
        depth_intrin,
        eye,
        te,
        cx,
        cy,
        reference_depth_m,
    )

    ref_cut_xr = float(robot_target[0][0])
    ref_cut_yr = float(robot_target[1][0])
    ref_cut_zr = float(robot_target[2][0])

    offset_x = xr - cut_xr
    offset_y = yr - cut_yr
    offset_z = zr - cut_zr

    ref_xr = ref_cut_xr + offset_x
    ref_yr = ref_cut_yr + offset_y
    ref_zr = ref_cut_zr + offset_z

    print(
        "X 轴触边，疑似深度异常，使用人工参考深度重算坐标: "
        f"原深度={depth * 100:.1f}cm, "
        f"当前视角参考深度={reference_depth_m * 100:.1f}cm, "
        f"原坐标=({xr:.2f}, {yr:.2f}, {zr:.2f}), "
        f"重算=({ref_xr:.2f}, {ref_yr:.2f}, {ref_zr:.2f})"
    )

    return {
        "Xc": float(cam_target[0][0]),
        "Yc": float(cam_target[1][0]),
        "Zc": float(cam_target[2][0]),
        "Xr": ref_xr,
        "Yr": ref_yr,
        "Zr": ref_zr,
        "cut_Xr": ref_cut_xr,
        "cut_Yr": ref_cut_yr,
        "cut_Zr": ref_cut_zr,
        "depth_m": reference_depth_m,
        "depth_mode": f"{depth_mode}->REF_X_RANGE",
    }


def image_vector_to_robot_yz_unit(depth_intrin, eye, te, px, py, depth, image_vector):
    """把图像方向换成机械臂 Y/Z 平面方向。"""
    image_vector = unit_vector(image_vector)
    if image_vector is None:
        return None

    probe_pixels = 20.0
    _cam0, robot0 = pixel_to_robot_coordinate(depth_intrin, eye, te, px, py, depth)
    _cam1, robot1 = pixel_to_robot_coordinate(
        depth_intrin,
        eye,
        te,
        px + image_vector[0] * probe_pixels,
        py + image_vector[1] * probe_pixels,
        depth,
    )

    delta = (robot1 - robot0).reshape(3)
    delta[0] = 0.0
    norm = float(np.linalg.norm(delta))

    if norm < 1e-6:
        return None

    return delta / norm


def analyze_cut_strategy(
    mask_polygon,
    main_stem_points,
    main_stem_mask,
    image_shape,
    depth_intrin,
    depth_frame,
    eye,
    te,
    depth_reference_m=None,
    reference_fallback_trigger_m=None,
    candidate_reference_max_error_m=None,
    use_main_stem_depth_fallback=None,
    mask_background_ring_width_px=0,
    mask_background_ring_min_points=0,
    min_mask_background_depth_contrast_m=0.0,
    capture_id=0,
    frame_id=0,
    target_index=0,
):
    """分析果梗 mask，返回剪切点、姿态和近主茎导入策略。"""
    stem_mask, depth_search_mask, depth_mask_info = prepare_cut_depth_masks(
        mask_polygon,
        image_shape,
    )
    stem_points = get_mask_points(stem_mask)
    tangent = get_stem_tangent(stem_points)

    if tangent is None:
        return None

    cut_region_points = get_cut_region_points(stem_points, tangent)
    cut_point, nearest_main_point, _pixel_distance = find_center_cut_point(
        cut_region_points,
        main_stem_points,
    )

    if cut_point is None:
        return None

    cut_px, cut_py = float(cut_point[0]), float(cut_point[1])
    cut_depth_point = estimate_depth_at_cut_point(
        depth_frame,
        cut_px,
        cut_py,
        mask_polygon,
        image_shape,
        nearest_main_point=nearest_main_point,
        main_stem_mask=main_stem_mask,
        depth_reference_m=depth_reference_m,
        reference_fallback_trigger_m=reference_fallback_trigger_m,
        candidate_reference_max_error_m=candidate_reference_max_error_m,
        use_main_stem_depth_fallback=use_main_stem_depth_fallback,
        mask_background_ring_width_px=mask_background_ring_width_px,
        mask_background_ring_min_points=mask_background_ring_min_points,
        min_mask_background_depth_contrast_m=(
            min_mask_background_depth_contrast_m
        ),
    )

    save_cut_depth_diagnostic(
        depth_frame,
        stem_mask,
        depth_search_mask,
        cut_px,
        cut_py,
        cut_depth_point,
        capture_id,
        frame_id,
        target_index,
        depth_mask_info,
    )

    if cut_depth_point is None:
        return None

    depth_px, depth_py, depth_m, depth_mode = cut_depth_point

    toward_main = None
    main_distance_cm = None

    if nearest_main_point is not None:
        toward_main = nearest_main_point - cut_point
        main_distance_cm = pixel_vector_distance_cm(
            toward_main,
            depth_m,
            depth_intrin,
        )

    cut_mode = "normal"
    if main_distance_cm is not None and main_distance_cm < CUT_SAFE_DISTANCE_CM:
        cut_mode = "guide"

    fruit_side = get_fruit_side(cut_point, nearest_main_point)
    contact_blade = choose_contact_blade(fruit_side)

    static_vector = choose_static_blade_vector(
        tangent,
        toward_main if cut_mode == "guide" else None,
        contact_blade=contact_blade if cut_mode == "guide" else "static",
    )

    if static_vector is None:
        return None

    target_angle = static_vector_to_rotate_angle(static_vector)
    stem_parallel_angle = -math.degrees(math.atan2(tangent[0], tangent[1]))
    stem_tilt_angle = abs(stem_parallel_angle)

    if cut_mode == "normal" and stem_tilt_angle <= STEM_NO_ROTATE_THRESHOLD_DEG:
        target_angle = 0

    cam_target, robot_target = pixel_to_robot_coordinate(
        depth_intrin,
        eye,
        te,
        cut_px,
        cut_py,
        depth_m,
    )

    xr = float(robot_target[0][0])
    yr = float(robot_target[1][0])
    zr = float(robot_target[2][0])

    guide_away_y = 0.0
    guide_away_z = 0.0

    if cut_mode == "guide":
        contact_vector = static_vector
        if contact_blade == "moving":
            contact_vector = -static_vector

        contact_robot_unit = image_vector_to_robot_yz_unit(
            depth_intrin,
            eye,
            te,
            cut_px,
            cut_py,
            depth_m,
            contact_vector,
        )

        if contact_robot_unit is not None:
            contact_robot_unit = np.asarray(contact_robot_unit, dtype=np.float32)

            guide_away_y = float(-contact_robot_unit[1])
            guide_away_z = float(-contact_robot_unit[2])

            if ENABLE_BLADE_CONTACT_OFFSET:
                lateral_offset_cm = get_contact_blade_offset(contact_blade)

                xr = xr - contact_robot_unit[0] * lateral_offset_cm
                yr = yr - contact_robot_unit[1] * lateral_offset_cm
                zr = zr - contact_robot_unit[2] * lateral_offset_cm

    high_risk = (
        main_distance_cm is not None
        and main_distance_cm < CUT_HIGH_RISK_DISTANCE_CM
    )

    return {
        "cx": int(round(cut_px)),
        "cy": int(round(cut_py)),
        "depth_px": int(depth_px),
        "depth_py": int(depth_py),
        "depth_m": float(depth_m),
        "depth_mode": depth_mode,
        "target_angle": target_angle,
        "stem_tilt_angle": stem_tilt_angle,
        "cut_mode": cut_mode,
        "fruit_side": fruit_side,
        "contact_blade": contact_blade,
        "main_stem_distance_cm": main_distance_cm,
        "main_stem_high_risk": high_risk,
        "guide_away_y": guide_away_y,
        "guide_away_z": guide_away_z,
        "Xc": float(cam_target[0][0]),
        "Yc": float(cam_target[1][0]),
        "Zc": float(cam_target[2][0]),
        "Xr": xr,
        "Yr": yr,
        "Zr": zr,
        "cut_Xr": float(robot_target[0][0]),
        "cut_Yr": float(robot_target[1][0]),
        "cut_Zr": float(robot_target[2][0]),
    }


def analyze_point_cloud_cut_strategy(
    detection,
    localization,
    main_stem_points,
    depth_intrin,
    eye,
    te,
):
    """用 mask 三维点云中心定位，并复用现有姿态与近主茎动作字段。"""
    center_m = np.asarray(localization["center"], dtype=np.float64).reshape(3)
    p1_m = np.asarray(localization["p1"], dtype=np.float64).reshape(3)
    p2_m = np.asarray(localization["p2"], dtype=np.float64).reshape(3)

    center_uv = project_camera_point_to_pixel(center_m, depth_intrin)
    p1_uv = project_camera_point_to_pixel(p1_m, depth_intrin)
    p2_uv = project_camera_point_to_pixel(p2_m, depth_intrin)
    if center_uv is None or p1_uv is None or p2_uv is None:
        return None

    tangent = unit_vector(p2_uv - p1_uv)
    if tangent is None:
        return None
    if tangent[1] < 0:
        tangent = -tangent

    nearest_main_point = None
    if main_stem_points is not None and len(main_stem_points) > 0:
        candidate_points = main_stem_points
        if len(candidate_points) > 6000:
            step = max(1, len(candidate_points) // 6000)
            candidate_points = candidate_points[::step]
        distances = np.sum((candidate_points - center_uv) ** 2, axis=1)
        nearest_main_point = candidate_points[int(np.argmin(distances))]

    toward_main = None
    main_distance_cm = None
    if nearest_main_point is not None:
        toward_main = nearest_main_point - center_uv
        main_distance_cm = pixel_vector_distance_cm(
            toward_main,
            float(center_m[2]),
            depth_intrin,
        )

    cut_mode = "normal"
    if main_distance_cm is not None and main_distance_cm < CUT_SAFE_DISTANCE_CM:
        cut_mode = "guide"

    fruit_side = get_fruit_side(center_uv, nearest_main_point)
    contact_blade = choose_contact_blade(fruit_side)
    static_vector = choose_static_blade_vector(
        tangent,
        toward_main if cut_mode == "guide" else None,
        contact_blade=contact_blade if cut_mode == "guide" else "static",
    )
    if static_vector is None:
        return None

    target_angle = static_vector_to_rotate_angle(static_vector)
    stem_parallel_angle = -math.degrees(math.atan2(tangent[0], tangent[1]))
    stem_tilt_angle = abs(stem_parallel_angle)
    if cut_mode == "normal" and stem_tilt_angle <= STEM_NO_ROTATE_THRESHOLD_DEG:
        target_angle = 0

    center_camera_cm = center_m * 100.0
    p1_camera_cm = p1_m * 100.0
    p2_camera_cm = p2_m * 100.0
    robot_target = np.asarray(
        eye.coordinate(center_camera_cm.reshape(3, 1), te),
        dtype=np.float64,
    ).reshape(3)
    p1_robot_cm = np.asarray(
        eye.coordinate(p1_camera_cm.reshape(3, 1), te),
        dtype=np.float64,
    ).reshape(3)
    p2_robot_cm = np.asarray(
        eye.coordinate(p2_camera_cm.reshape(3, 1), te),
        dtype=np.float64,
    ).reshape(3)
    if not np.all(np.isfinite(robot_target)):
        return None

    cut_robot_target = robot_target.copy()
    guide_away_y = 0.0
    guide_away_z = 0.0
    if cut_mode == "guide":
        contact_vector = static_vector
        if contact_blade == "moving":
            contact_vector = -static_vector

        contact_robot_unit = image_vector_to_robot_yz_unit(
            depth_intrin,
            eye,
            te,
            float(center_uv[0]),
            float(center_uv[1]),
            float(center_m[2]),
            contact_vector,
        )
        if contact_robot_unit is not None:
            contact_robot_unit = np.asarray(contact_robot_unit, dtype=np.float64)
            guide_away_y = float(-contact_robot_unit[1])
            guide_away_z = float(-contact_robot_unit[2])
            if ENABLE_BLADE_CONTACT_OFFSET:
                lateral_offset_cm = get_contact_blade_offset(contact_blade)
                robot_target = robot_target - contact_robot_unit * lateral_offset_cm

    high_risk = (
        main_distance_cm is not None
        and main_distance_cm < CUT_HIGH_RISK_DISTANCE_CM
    )
    center_x = int(round(float(center_uv[0])))
    center_y = int(round(float(center_uv[1])))

    return {
        "cx": center_x,
        "cy": center_y,
        "depth_px": center_x,
        "depth_py": center_y,
        "depth_m": float(center_m[2]),
        "depth_mode": "MASK_PCA",
        "depth_quality": "mask三维点云PCA",
        "target_angle": target_angle,
        "stem_tilt_angle": stem_tilt_angle,
        "cut_mode": cut_mode,
        "fruit_side": fruit_side,
        "contact_blade": contact_blade,
        "main_stem_distance_cm": main_distance_cm,
        "main_stem_high_risk": high_risk,
        "guide_away_y": guide_away_y,
        "guide_away_z": guide_away_z,
        "Xc": float(center_camera_cm[0]),
        "Yc": float(center_camera_cm[1]),
        "Zc": float(center_camera_cm[2]),
        "Xr": float(robot_target[0]),
        "Yr": float(robot_target[1]),
        "Zr": float(robot_target[2]),
        "cut_Xr": float(cut_robot_target[0]),
        "cut_Yr": float(cut_robot_target[1]),
        "cut_Zr": float(cut_robot_target[2]),
        "raw_point_count": int(localization.get("raw_point_count", 0)),
        "point_count": int(localization["point_count"]),
        "linearity": float(localization["linearity"]),
        "line_length_cm": float(localization["line_length_cm"]),
        "line_p1_camera_m": p1_m.astype(np.float32),
        "line_p2_camera_m": p2_m.astype(np.float32),
        "line_p1_robot_cm": p1_robot_cm,
        "line_p2_robot_cm": p2_robot_cm,
        "line_p1_uv": p1_uv,
        "line_p2_uv": p2_uv,
    }


def estimate_rotation_angle_from_mask(mask_polygon, image_shape):
    """根据 YOLO 分割 mask 估计带符号果梗旋转角。"""
    h, w = image_shape[:2]

    if mask_polygon is None or len(mask_polygon) < 5:
        return 0, 0

    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.round(mask_polygon).astype(np.int32)
    cv2.fillPoly(mask, [pts], 255)

    coords = cv2.findNonZero(mask)
    if coords is None or len(coords) < 10:
        return 0, 0

    points = coords[:, 0, :].astype(np.float32)
    _mean, eigenvectors = cv2.PCACompute(points, mean=None)
    vx, vy = eigenvectors[0]

    if vy < 0:
        vx = -vx
        vy = -vy

    stem_parallel_angle = -math.degrees(math.atan2(vx, vy))
    stem_tilt_angle = abs(stem_parallel_angle)

    if stem_tilt_angle <= STEM_NO_ROTATE_THRESHOLD_DEG:
        return 0, stem_tilt_angle

    cut_perpendicular_angle = normalize_pick_angle(stem_parallel_angle + 90)

    print(
        "果梗角度转换: "
        f"平行角={stem_parallel_angle:.1f} 度, "
        f"切割垂直角={cut_perpendicular_angle:.1f} 度"
    )

    return cut_perpendicular_angle, stem_tilt_angle


TARGET_OBSERVATION_KEYS = (
    "cx",
    "cy",
    "x1",
    "y1",
    "x2",
    "y2",
    "box_cx",
    "box_cy",
    "depth_px",
    "depth_py",
    "depth_m",
    "depth_mode",
    "depth_quality",
    "depth_foreground_verified",
    "depth_reference_m",
    "scene_depth_reference_m",
    "class_name",
    "score",
    "mask",
    "contour",
    "angle",
    "stem_tilt_angle",
    "cut_mode",
    "fruit_side",
    "contact_blade",
    "main_stem_distance_cm",
    "main_stem_high_risk",
    "guide_away_y",
    "guide_away_z",
    "cut_Xr",
    "cut_Yr",
    "cut_Zr",
    "Xc",
    "Yc",
    "Zc",
    "Xr",
    "Yr",
    "Zr",
    "raw_point_count",
    "point_count",
    "linearity",
    "line_length_cm",
    "line_p1_camera_m",
    "line_p2_camera_m",
    "line_p1_robot_cm",
    "line_p2_robot_cm",
    "line_p1_uv",
    "line_p2_uv",
    "last_frame_id",
)


def copy_target_observation(target):
    """复制一帧目标观测，避免后续融合时互相污染。"""
    return {
        key: target[key]
        for key in TARGET_OBSERVATION_KEYS
        if key in target
    }


def is_reference_depth_mode(depth_mode):
    text = str(depth_mode)
    return (
        "REF_FALLBACK" in text
        or "REF_X_RANGE" in text
        or text == "REF"
        or text == "FALLBACK"
    )


def is_good_depth_mode(depth_mode):
    text = str(depth_mode)
    if is_reference_depth_mode(text):
        return False
    return text in GOOD_DEPTH_MODES


def is_unverified_fruit_depth_mode(depth_mode):
    """判断深度是否来自果梗附近、但尚未通过前景验证。"""
    text = str(depth_mode)
    if is_reference_depth_mode(text):
        return False
    return text in UNVERIFIED_FRUIT_DEPTH_MODES


def is_main_stem_depth_mode(depth_mode):
    text = str(depth_mode)
    if is_reference_depth_mode(text):
        return False
    return text in MAIN_STEM_DEPTH_MODES


def get_depth_quality_score(observation):
    depth_mode = observation.get("depth_mode", "")

    if is_good_depth_mode(depth_mode):
        return 4

    if is_main_stem_depth_mode(depth_mode):
        return 3

    if is_unverified_fruit_depth_mode(depth_mode):
        return 2

    if str(depth_mode) == "REF":
        return 1

    return 0


def find_stable_depth_observation(observations, mode_checker, min_count):
    """在指定深度模式里找一组稳定观测，并返回最接近中位深度的那一帧。"""
    candidates = [
        observation for observation in observations
        if mode_checker(observation.get("depth_mode", ""))
        and observation.get("depth_m") is not None
    ]

    if len(candidates) < min_count:
        return None, 0

    candidates.sort(key=lambda observation: observation["depth_m"])
    stable_band = max(0.0, float(MULTI_FRAME_DEPTH_STABLE_BAND_M))
    best_group = []

    for start_index, start_observation in enumerate(candidates):
        group = []
        start_depth = start_observation["depth_m"]

        for observation in candidates[start_index:]:
            if observation["depth_m"] - start_depth <= stable_band:
                group.append(observation)
            else:
                break

        if len(group) > len(best_group):
            best_group = group

    if len(best_group) < min_count:
        return None, 0

    median_depth = float(np.median([
        observation["depth_m"] for observation in best_group
    ]))

    best_observation = min(
        best_group,
        key=lambda observation: (
            abs(observation["depth_m"] - median_depth),
            -get_depth_quality_score(observation),
        ),
    )

    return best_observation, len(best_group)


def select_best_observation(observations):
    """按深度质量选择当前目标最可信的一帧观测。"""
    good_observation, good_count = find_stable_depth_observation(
        observations,
        is_good_depth_mode,
        max(1, int(MIN_GOOD_DEPTH_FRAMES)),
    )
    if good_observation is not None:
        return good_observation, "稳定已验证果梗前景", good_count

    main_observation, main_count = find_stable_depth_observation(
        observations,
        is_main_stem_depth_mode,
        max(1, int(MIN_MAIN_STEM_DEPTH_FRAMES)),
    )
    if main_observation is not None:
        return main_observation, "稳定主茎深度", main_count

    good_candidates = [
        observation for observation in observations
        if is_good_depth_mode(observation.get("depth_mode", ""))
    ]
    if good_candidates:
        return good_candidates[-1], "单帧已验证果梗前景", len(good_candidates)

    main_candidates = [
        observation for observation in observations
        if is_main_stem_depth_mode(observation.get("depth_mode", ""))
    ]
    if main_candidates:
        return main_candidates[-1], "单帧主茎深度", len(main_candidates)

    unverified_observation, unverified_count = find_stable_depth_observation(
        observations,
        is_unverified_fruit_depth_mode,
        max(1, int(MIN_GOOD_DEPTH_FRAMES)),
    )
    if unverified_observation is not None:
        return (
            unverified_observation,
            "稳定但未验证果梗前景",
            unverified_count,
        )

    unverified_candidates = [
        observation for observation in observations
        if is_unverified_fruit_depth_mode(
            observation.get("depth_mode", "")
        )
    ]
    if unverified_candidates:
        return (
            unverified_candidates[-1],
            "单帧未验证果梗前景",
            len(unverified_candidates),
        )

    best_observation = max(
        observations,
        key=lambda observation: get_depth_quality_score(observation),
    )
    return best_observation, "参考/兜底深度", 1


def apply_best_observation(target):
    """把多帧观测里质量最高的结果写回目标。"""
    observations = target.get("observations", [])
    if not observations:
        return

    count = target.get("count", len(observations))
    pca_observations = [
        observation
        for observation in observations
        if str(observation.get("depth_mode", "")) == "MASK_PCA"
        and observation.get("depth_m") is not None
    ]
    if pca_observations:
        stable_band = max(0.0, float(MULTI_FRAME_DEPTH_STABLE_BAND_M))
        ordered_observations = sorted(
            pca_observations,
            key=lambda observation: float(observation["depth_m"]),
        )
        candidate_groups = []
        for start_index, start_observation in enumerate(ordered_observations):
            start_depth = float(start_observation["depth_m"])
            group = [
                observation
                for observation in ordered_observations[start_index:]
                if float(observation["depth_m"]) - start_depth <= stable_band
            ]
            if group:
                candidate_groups.append(group)

        # 深度组数量相同时，优先点云点数更多、线性度更高的一组。
        # 这样一帧 37.8cm 伪深度和一帧 76cm 真深度冲突时，不再固定偏向小深度。
        stable_observations = max(
            candidate_groups,
            key=lambda group: (
                len(group),
                sum(int(observation.get("point_count", 0)) for observation in group),
                max(float(observation.get("linearity", 0.0)) for observation in group),
            ),
        )

        best_observation = max(
            stable_observations,
            key=lambda observation: (
                int(observation.get("point_count", 0)),
                float(observation.get("linearity", 0.0)),
            ),
        )
        for key, value in best_observation.items():
            target[key] = value

        average_keys = (
            "depth_m",
            "Xc",
            "Yc",
            "Zc",
            "Xr",
            "Yr",
            "Zr",
            "cut_Xr",
            "cut_Yr",
            "cut_Zr",
            "score",
            "linearity",
            "line_length_cm",
        )
        for key in average_keys:
            values = [
                float(observation[key])
                for observation in stable_observations
                if observation.get(key) is not None
            ]
            if values:
                target[key] = float(np.mean(values))

        for key in ("cx", "cy", "depth_px", "depth_py"):
            values = [
                float(observation[key])
                for observation in stable_observations
                if observation.get(key) is not None
            ]
            if values:
                target[key] = int(round(float(np.mean(values))))

        target["count"] = count
        target["observations"] = observations
        target["depth_quality"] = (
            "MASK点云PCA多帧平均"
            if len(stable_observations) > 1
            else "MASK点云PCA单帧"
        )
        target["stable_depth_count"] = len(stable_observations)
        return

    best_observation, depth_quality, stable_count = select_best_observation(
        observations,
    )

    for key, value in best_observation.items():
        target[key] = value

    target["count"] = count
    target["observations"] = observations
    target["depth_quality"] = depth_quality
    target["stable_depth_count"] = stable_count


def initialize_target_tracking(target):
    """初始化一个新目标的多帧观测记录。"""
    target["count"] = max(1, int(target.get("count", 1)))
    target["observations"] = [copy_target_observation(target)]
    apply_best_observation(target)


def has_stable_good_depth(
    target,
    accepted_modes=None,
    min_stable_frames=None,
    expected_depth_m=None,
    max_depth_error_m=None,
):
    """判断目标是否具有满足指定来源和帧数要求的稳定深度。"""
    observations = target.get("observations", [])

    if expected_depth_m is not None and max_depth_error_m is not None:
        allowed_error = max(0.0, float(max_depth_error_m))
        observations = [
            observation for observation in observations
            if observation.get("depth_m") is not None
            and abs(
                float(observation["depth_m"]) - float(expected_depth_m)
            ) <= allowed_error
        ]

    if accepted_modes is None:
        mode_checker = is_good_depth_mode
    else:
        accepted_mode_set = {str(mode) for mode in accepted_modes}
        mode_checker = lambda mode: str(mode) in accepted_mode_set

    required_count = (
        MIN_GOOD_DEPTH_FRAMES
        if min_stable_frames is None
        else min_stable_frames
    )

    good_observation, _good_count = find_stable_depth_observation(
        observations,
        mode_checker,
        max(1, int(required_count)),
    )
    return good_observation is not None


def should_stop_on_stable_depth(
    targets,
    finished_frames,
    accepted_modes=None,
    min_stable_frames=None,
    expected_depth_m=None,
    max_depth_error_m=None,
):
    """所有已跟踪目标都有稳定果梗深度时，允许提前结束多帧检测。"""
    if not ALLOW_EARLY_STOP_ON_STABLE_DEPTH:
        return False

    if finished_frames < max(1, int(EARLY_STOP_MIN_FRAMES)):
        return False

    valid_targets = [
        target for target in targets
        if target.get("count", 0) >= MIN_VOTE
    ]

    if not valid_targets:
        return False

    return all(
        has_stable_good_depth(
            target,
            accepted_modes=accepted_modes,
            min_stable_frames=min_stable_frames,
            expected_depth_m=expected_depth_m,
            max_depth_error_m=max_depth_error_m,
        )
        for target in valid_targets
    )


def update_target(new_target, targets, distance_threshold=None):
    """把单帧目标合并到多帧投票列表，并优先保留高质量稳定深度。"""
    merge_distance = (
        DIST_THRESHOLD
        if distance_threshold is None
        else max(1.0, float(distance_threshold))
    )

    for target in targets:
        if target.get("last_frame_id") == new_target.get("last_frame_id"):
            continue
        if target.get("class_name") != new_target.get("class_name"):
            continue

        old_x = target["cx"]
        old_y = target["cy"]

        dist = math.sqrt(
            (new_target["cx"] - old_x) ** 2
            + (new_target["cy"] - old_y) ** 2
        )

        if dist < merge_distance:
            if "observations" not in target:
                target["observations"] = [copy_target_observation(target)]

            target["count"] += 1
            target["observations"].append(copy_target_observation(new_target))
            apply_best_observation(target)
            # 最优观测可能来自较早帧；单独保留最新帧号，防止同帧候选重复合并。
            target["last_frame_id"] = new_target.get("last_frame_id")

            return True

    return False


def get_class_name(names, class_id):
    if names is None:
        return str(class_id)

    if isinstance(names, dict):
        return str(names.get(class_id, class_id))

    if 0 <= class_id < len(names):
        return str(names[class_id])

    return str(class_id)


def build_detections(result, model, image_shape):
    """整理 YOLO segmentation 输出，并恢复原图尺寸二值实例 mask。"""
    boxes_obj = getattr(result, "boxes", None)
    masks_obj = getattr(result, "masks", None)
    if boxes_obj is None or getattr(boxes_obj, "xyxy", None) is None:
        return []

    boxes = _to_numpy(boxes_obj.xyxy).reshape(-1, 4)
    class_ids = np.zeros(len(boxes), dtype=np.int32)
    if getattr(boxes_obj, "cls", None) is not None:
        class_ids = _to_numpy(boxes_obj.cls).reshape(-1).astype(np.int32)

    scores = np.ones(len(boxes), dtype=np.float32)
    if getattr(boxes_obj, "conf", None) is not None:
        scores = _to_numpy(boxes_obj.conf).reshape(-1).astype(np.float32)

    mask_data = None
    if masks_obj is not None and getattr(masks_obj, "data", None) is not None:
        mask_data = _to_numpy(masks_obj.data)
        if mask_data.ndim == 2:
            mask_data = mask_data[None, ...]

    mask_polygons = getattr(masks_obj, "xy", None) if masks_obj is not None else None
    names = getattr(result, "names", None)
    if names is None:
        names = getattr(model, "names", None)

    detections = []
    for box_index, box in enumerate(boxes):
        class_id = int(class_ids[box_index]) if box_index < len(class_ids) else -1
        class_name = get_class_name(names, class_id)
        score = float(scores[box_index]) if box_index < len(scores) else 1.0
        mask_polygon = None
        if mask_polygons is not None and box_index < len(mask_polygons):
            mask_polygon = np.asarray(mask_polygons[box_index], dtype=np.float32)

        mask = None
        if mask_data is not None and box_index < len(mask_data):
            mask = make_original_size_mask(
                result,
                mask_data,
                box_index,
                image_shape,
            )
        elif mask_polygon is not None and len(mask_polygon) >= 3:
            mask = polygon_to_mask(mask_polygon, image_shape) > 0

        if mask is not None and int(np.count_nonzero(mask)) < MIN_MASK_PIXELS:
            mask = None

        contour = None
        if mask is not None:
            contours, _ = cv2.findContours(
                mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            if contours:
                contour = max(contours, key=cv2.contourArea).reshape(-1, 2)

        detections.append({
            "box": box.astype(np.float32),
            "score": score,
            "class_id": class_id,
            "class_name": class_name,
            "mask": mask,
            "contour": contour,
            "mask_polygon": mask_polygon,
            "is_main_stem": class_name_matches(class_name, MAIN_STEM_CLASS_NAMES),
            "is_fruit_stem": class_name_matches(class_name, FRUIT_STEM_CLASS_NAMES),
        })

    return detections


def wait_for_aligned_color_depth_frames(pipeline, align):
    """等待一组对齐后的 RealSense 彩色 / 深度帧，失败时重试。"""
    retry_count = max(1, int(REALSENSE_FRAME_RETRY_COUNT))

    for attempt in range(1, retry_count + 1):
        try:
            frames = pipeline.wait_for_frames(int(REALSENSE_FRAME_TIMEOUT_MS))
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if color_frame and depth_frame:
                return color_frame, depth_frame

            print(
                "RealSense 帧不完整: "
                f"第 {attempt}/{retry_count} 次尝试"
            )

        except RuntimeError as exc:
            print(
                "RealSense 等帧超时: "
                f"第 {attempt}/{retry_count} 次尝试, {exc}"
            )

        time.sleep(REALSENSE_FRAME_RETRY_DELAY_SECONDS)

    return None, None


def _detect_targets_legacy(
    pipeline,
    align,
    model,
    eye,
    te,
    num_frames,
    early_stop_depth_modes=None,
    early_stop_min_stable_frames=None,
    target_merge_distance_px=None,
    depth_reference_m=None,
    depth_reference_fallback_trigger_m=None,
    depth_candidate_reference_max_error_m=None,
    use_main_stem_depth_fallback=None,
    mask_background_ring_width_px=0,
    mask_background_ring_min_points=0,
    min_mask_background_depth_contrast_m=0.0,
    early_stop_reference_max_error_m=None,
    use_reference_x_recheck=None,
    capture_label=None,
):
    """连续检测多帧 RGB-D 图像，并返回机械臂坐标系下的目标。"""
    all_targets = []
    reference_depth_m = resolve_depth_reference_m(depth_reference_m)
    if mask_background_ring_width_px is None:
        mask_background_ring_width_px = CUT_MASK_BACKGROUND_RING_WIDTH_PX
    if mask_background_ring_min_points is None:
        mask_background_ring_min_points = CUT_MASK_BACKGROUND_RING_MIN_POINTS
    if min_mask_background_depth_contrast_m is None:
        min_mask_background_depth_contrast_m = (
            CUT_MASK_MIN_BACKGROUND_DEPTH_CONTRAST_M
        )
    capture_id = begin_capture_session(
        capture_label or f"视觉检测_参考深度{reference_depth_m * 100:.1f}cm"
    )

    print(f"\n开始连续检测: {num_frames} 帧\n")
    print(f"当前视角深度参考: {reference_depth_m * 100:.1f}cm\n")

    for frame_id in range(num_frames):
        print(f"处理第 {frame_id + 1} 帧")

        color_frame, depth_frame = wait_for_aligned_color_depth_frames(
            pipeline,
            align,
        )

        if not color_frame or not depth_frame:
            print("RealSense 帧不可用，跳过当前检测帧")
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics

        results = model(
            color_image,
            imgsz=IMG_SIZE,
            conf=YOLO_CONF,
            verbose=False,
        )

        result = results[0]
        detections = build_detections(result, model, color_image.shape)

        if len(detections) == 0:
            print("未检测到目标")
            save_result_image(color_image, all_targets, frame_id, capture_id)
            continue

        main_stem_mask = np.zeros(color_image.shape[:2], dtype=np.uint8)
        main_stem_count = 0

        for detection in detections:
            if detection["is_main_stem"] and detection["mask_polygon"] is not None:
                main_stem_mask = cv2.bitwise_or(
                    main_stem_mask,
                    polygon_to_mask(detection["mask_polygon"], color_image.shape),
                )
                main_stem_count += 1

        main_stem_points = None
        if main_stem_count > 0:
            main_stem_points = get_mask_points(main_stem_mask)

        target_detections = [
            detection for detection in detections
            if detection["is_fruit_stem"]
        ]

        if len(target_detections) == 0:
            target_detections = [
                detection for detection in detections
                if not detection["is_main_stem"]
            ]

        if len(target_detections) == 0:
            print("只检测到主茎，未检测到可采摘果梗")
            save_result_image(color_image, all_targets, frame_id, capture_id)
            continue

        print(
            f"检测到 {len(detections)} 个目标，"
            f"果梗候选 {len(target_detections)} 个，"
            f"主茎 {main_stem_count} 个"
        )

        for target_index, detection in enumerate(target_detections):
            box = detection["box"]
            x1, y1, x2, y2 = map(int, box)
            box_cx = (x1 + x2) // 2
            box_cy = (y1 + y2) // 2

            strategy = None

            if detection["mask_polygon"] is not None:
                strategy = analyze_cut_strategy(
                    detection["mask_polygon"],
                    main_stem_points,
                    main_stem_mask,
                    color_image.shape,
                    depth_intrin,
                    depth_frame,
                    eye,
                    te,
                    depth_reference_m=reference_depth_m,
                    reference_fallback_trigger_m=(
                        depth_reference_fallback_trigger_m
                    ),
                    candidate_reference_max_error_m=(
                        depth_candidate_reference_max_error_m
                    ),
                    use_main_stem_depth_fallback=(
                        use_main_stem_depth_fallback
                    ),
                    mask_background_ring_width_px=(
                        mask_background_ring_width_px
                    ),
                    mask_background_ring_min_points=(
                        mask_background_ring_min_points
                    ),
                    min_mask_background_depth_contrast_m=(
                        min_mask_background_depth_contrast_m
                    ),
                    capture_id=capture_id,
                    frame_id=frame_id,
                    target_index=target_index,
                )

            if strategy is not None:
                cx = strategy["cx"]
                cy = strategy["cy"]
                depth_cx = strategy["depth_px"]
                depth_cy = strategy["depth_py"]
                depth = strategy["depth_m"]
                depth_mode = strategy["depth_mode"]
                target_angle = strategy["target_angle"]
                stem_tilt_angle = strategy["stem_tilt_angle"]
                xc = strategy["Xc"]
                yc = strategy["Yc"]
                zc = strategy["Zc"]
                xr = strategy["Xr"]
                yr = strategy["Yr"]
                zr = strategy["Zr"]
                cut_mode = strategy["cut_mode"]
                fruit_side = strategy["fruit_side"]
                contact_blade = strategy["contact_blade"]
                main_stem_distance_cm = strategy["main_stem_distance_cm"]
                main_stem_high_risk = strategy["main_stem_high_risk"]
                guide_away_y = strategy["guide_away_y"]
                guide_away_z = strategy["guide_away_z"]
                cut_xr = strategy["cut_Xr"]
                cut_yr = strategy["cut_Yr"]
                cut_zr = strategy["cut_Zr"]
            else:
                depth_point = get_reliable_depth_point_in_box(
                    depth_frame,
                    x1,
                    y1,
                    x2,
                    y2,
                    depth_reference_m=reference_depth_m,
                )

                if depth_point is None:
                    print(f"({box_cx},{box_cy}) 检测框内没有可靠深度点")
                    continue

                depth_cx, depth_cy, depth, depth_mode = depth_point

                if detection["mask_polygon"] is not None:
                    target_angle, stem_tilt_angle = estimate_rotation_angle_from_mask(
                        detection["mask_polygon"],
                        color_image.shape,
                    )
                else:
                    print("模型未输出分割 mask，旋转角暂设为 0")
                    target_angle = 0
                    stem_tilt_angle = 0

                cx, cy = depth_cx, depth_cy
                cam_target, robot_target = pixel_to_robot_coordinate(
                    depth_intrin,
                    eye,
                    te,
                    cx,
                    cy,
                    depth,
                )
                xc = float(cam_target[0][0])
                yc = float(cam_target[1][0])
                zc = float(cam_target[2][0])
                xr = float(robot_target[0][0])
                yr = float(robot_target[1][0])
                zr = float(robot_target[2][0])
                cut_mode = "normal"
                fruit_side = "unknown"
                contact_blade = "static"
                main_stem_distance_cm = None
                main_stem_high_risk = False
                guide_away_y = 0.0
                guide_away_z = 0.0
                cut_xr = xr
                cut_yr = yr
                cut_zr = zr

            distance_text = "无主茎参考"
            if main_stem_distance_cm is not None:
                distance_text = f"{main_stem_distance_cm:.1f} 厘米"

            if main_stem_high_risk:
                print(
                    "高风险近主茎剪切: "
                    f"距离={main_stem_distance_cm:.1f} 厘米，先按导入策略尝试"
                )

            positioning_text = (
                f"{contact_blade}刀片贴近"
                if ENABLE_BLADE_CONTACT_OFFSET
                else "旋转中心（刀片偏置已关闭）"
            )

            print(
                f"果梗类别={detection['class_name']}, "
                f"剪切模式={cut_mode}, "
                f"果梗侧={fruit_side}, "
                f"定位基准={positioning_text}, "
                f"主茎距离={distance_text}, "
                f"目标角={target_angle:.1f} 度, "
                f"倾角={stem_tilt_angle:.1f} 度"
            )

            reference_projection = reproject_target_with_reference_depth(
                depth_intrin,
                eye,
                te,
                cx,
                cy,
                xr,
                yr,
                zr,
                cut_xr,
                cut_yr,
                cut_zr,
                depth,
                depth_mode,
                depth_reference_m=reference_depth_m,
                enabled=use_reference_x_recheck,
            )

            if reference_projection is not None:
                xc = reference_projection["Xc"]
                yc = reference_projection["Yc"]
                zc = reference_projection["Zc"]
                xr = reference_projection["Xr"]
                yr = reference_projection["Yr"]
                zr = reference_projection["Zr"]
                cut_xr = reference_projection["cut_Xr"]
                cut_yr = reference_projection["cut_Yr"]
                cut_zr = reference_projection["cut_Zr"]
                depth = reference_projection["depth_m"]
                depth_mode = reference_projection["depth_mode"]
                depth_cx = cx
                depth_cy = cy

            if xr > X_MAX:
                print(f"X 超过最大范围: {xr:.2f}，指令 X 已限幅到 {X_MAX:.2f}")
                xr = X_MAX
            elif xr < X_MIN:
                print(f"X 超出范围: {xr:.2f}")
                continue

            if not (Y_MIN <= yr <= Y_MAX):
                print(f"Y 超出范围: {yr:.2f}")
                continue

            target = {
                "cx": cx,
                "cy": cy,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "box_cx": box_cx,
                "box_cy": box_cy,
                "depth_px": depth_cx,
                "depth_py": depth_cy,
                "depth_m": depth,
                "depth_mode": depth_mode,
                "depth_foreground_verified": depth_mode == "CUT_MASK_FG",
                "depth_reference_m": reference_depth_m,
                "class_name": detection["class_name"],
                "angle": target_angle,
                "stem_tilt_angle": stem_tilt_angle,
                "cut_mode": cut_mode,
                "fruit_side": fruit_side,
                "contact_blade": contact_blade,
                "main_stem_distance_cm": main_stem_distance_cm,
                "main_stem_high_risk": main_stem_high_risk,
                "guide_away_y": guide_away_y,
                "guide_away_z": guide_away_z,
                "cut_Xr": cut_xr,
                "cut_Yr": cut_yr,
                "cut_Zr": cut_zr,
                "Xc": xc,
                "Yc": yc,
                "Zc": zc,
                "Xr": xr,
                "Yr": yr,
                "Zr": zr,
                "count": 1,
            }

            if update_target(
                target,
                all_targets,
                distance_threshold=target_merge_distance_px,
            ):
                merge_distance = (
                    DIST_THRESHOLD
                    if target_merge_distance_px is None
                    else max(1.0, float(target_merge_distance_px))
                )
                matched_target = next(
                    (
                        current for current in all_targets
                        if math.sqrt(
                            (current["cx"] - target["cx"]) ** 2
                            + (current["cy"] - target["cy"]) ** 2
                        ) < merge_distance
                    ),
                    None,
                )
                if matched_target is not None:
                    print(
                        f"已更新目标 ({cx},{cy})，"
                        f"深度选择={matched_target.get('depth_quality', '未知')}, "
                        f"稳定帧={matched_target.get('stable_depth_count', 1)}, "
                        f"当前模式={matched_target.get('depth_mode', 'NA')}"
                    )
                else:
                    print(f"已更新目标 ({cx},{cy})")
            else:
                initialize_target_tracking(target)
                all_targets.append(target)
                print(
                    f"新增目标 ({cx},{cy})，"
                    f"深度选择={target.get('depth_quality', '未知')}, "
                    f"当前模式={target.get('depth_mode', 'NA')}"
                )

        save_result_image(color_image, all_targets, frame_id, capture_id)

        if should_stop_on_stable_depth(
            all_targets,
            frame_id + 1,
            accepted_modes=early_stop_depth_modes,
            min_stable_frames=early_stop_min_stable_frames,
            expected_depth_m=reference_depth_m,
            max_depth_error_m=early_stop_reference_max_error_m,
        ):
            print(
                "已拍到稳定果梗深度，提前结束当前扫描位检测: "
                f"已处理 {frame_id + 1}/{num_frames} 帧"
            )
            break

    return [target for target in all_targets if target["count"] >= MIN_VOTE]


def detect_targets(
    pipeline,
    align,
    model,
    eye,
    te,
    num_frames,
    early_stop_depth_modes=None,
    early_stop_min_stable_frames=None,
    target_merge_distance_px=None,
    depth_reference_m=None,
    depth_reference_fallback_trigger_m=None,
    depth_candidate_reference_max_error_m=None,
    use_main_stem_depth_fallback=None,
    mask_background_ring_width_px=0,
    mask_background_ring_min_points=0,
    min_mask_background_depth_contrast_m=0.0,
    early_stop_reference_max_error_m=None,
    use_reference_x_recheck=None,
    capture_label=None,
):
    """连续检测多帧，只用实例 mask 三维点云/PCA 生成采摘坐标。"""
    # 保留旧参数只是为了兼容 main1/pick1 的调用接口；点云定位不再使用任何
    # 人工参考深度、主茎借深度、背景环或 X 触边重算策略。
    _ = (
        depth_reference_m,
        depth_reference_fallback_trigger_m,
        depth_candidate_reference_max_error_m,
        use_main_stem_depth_fallback,
        mask_background_ring_width_px,
        mask_background_ring_min_points,
        min_mask_background_depth_contrast_m,
        early_stop_reference_max_error_m,
        use_reference_x_recheck,
    )

    all_targets = []
    capture_id = begin_capture_session(capture_label or "视觉检测_MASK点云PCA")
    print(f"\n开始连续检测: {num_frames} 帧")
    print(
        "定位方式: 果梗实例 mask -> 三维点云 -> 离群过滤 -> PCA 中心\n"
        f"YOLO参数: imgsz={IMG_SIZE}, 推理阈值={YOLO_INFERENCE_CONF:.2f}, "
        f"采摘阈值={YOLO_CONF:.2f}, iou={YOLO_IOU:.2f}\n"
    )

    for frame_id in range(num_frames):
        print(f"处理第 {frame_id + 1} 帧")
        color_frame, depth_frame = wait_for_aligned_color_depth_frames(
            pipeline,
            align,
        )
        if not color_frame or not depth_frame:
            print("RealSense 帧不可用，跳过当前检测帧")
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        try:
            depth_image_m = depth_frame_to_meters(depth_frame)
        except ValueError as exc:
            print(f"深度帧无效，本帧跳过: {exc}")
            save_result_image(color_image, all_targets, frame_id, capture_id)
            continue

        prediction_options = {
            "imgsz": IMG_SIZE,
            "conf": YOLO_INFERENCE_CONF,
            "iou": YOLO_IOU,
            "retina_masks": YOLO_RETINA_MASKS,
            "verbose": False,
        }
        predict = getattr(model, "predict", None)
        if callable(predict):
            results = predict(source=color_image, **prediction_options)
        else:
            results = model(color_image, **prediction_options)

        if not results:
            print("YOLO 未返回检测结果")
            save_detection_threshold_image(
                color_image,
                [],
                frame_id,
                capture_id,
            )
            save_result_image(color_image, all_targets, frame_id, capture_id)
            continue

        result = results[0]
        detections = build_detections(result, model, color_image.shape)
        save_detection_threshold_image(
            color_image,
            detections,
            frame_id,
            capture_id,
        )
        if not detections:
            print("YOLO 没有输出可用检测框")
            save_result_image(color_image, all_targets, frame_id, capture_id)
            continue

        qualified_detections = [
            detection
            for detection in detections
            if detection["score"] >= YOLO_CONF and detection["mask"] is not None
        ]
        print(
            f"YOLO输出={len(detections)}, "
            f"高于采摘阈值且有mask={len(qualified_detections)}"
        )
        if not qualified_detections:
            print("本帧没有可进入三维点云定位的实例")
            save_result_image(color_image, all_targets, frame_id, capture_id)
            continue

        main_stem_mask = np.zeros(color_image.shape[:2], dtype=np.uint8)
        main_stem_count = 0
        for detection in qualified_detections:
            if detection["is_main_stem"]:
                main_stem_mask = cv2.bitwise_or(
                    main_stem_mask,
                    detection["mask"].astype(np.uint8) * 255,
                )
                main_stem_count += 1

        main_stem_points = get_mask_points(main_stem_mask)
        target_detections = [
            detection
            for detection in qualified_detections
            if detection["is_fruit_stem"]
        ]
        if not target_detections:
            target_detections = [
                detection
                for detection in qualified_detections
                if not detection["is_main_stem"]
            ]

        if not target_detections:
            print("本帧只有主茎，没有可采摘果梗")
            save_result_image(color_image, all_targets, frame_id, capture_id)
            continue

        print(
            f"果梗候选={len(target_detections)}, 主茎={main_stem_count}"
        )
        localized_count = 0

        try:
            main_stem_point_cloud = instance_mask_to_point_cloud(
                main_stem_mask > 0,
                depth_image_m,
                depth_intrin,
            )
        except ValueError:
            main_stem_point_cloud = np.zeros((0, 3), dtype=np.float32)

        target_point_clouds = []
        for detection in target_detections:
            try:
                raw_point_cloud = instance_mask_to_point_cloud(
                    detection["mask"],
                    depth_image_m,
                    depth_intrin,
                )
            except ValueError as exc:
                print(f"实例点云生成失败，已跳过: {exc}")
                raw_point_cloud = np.zeros((0, 3), dtype=np.float32)
            target_point_clouds.append(raw_point_cloud)

        scene_depth_reference_m, scene_support_count = estimate_scene_depth_reference(
            [main_stem_point_cloud, *target_point_clouds]
        )
        if scene_depth_reference_m is None:
            print("本帧没有形成场景深度共识，各果梗将选择自身最大深度簇")
        else:
            print(
                "本帧植物平面深度共识: "
                f"{scene_depth_reference_m * 100:.2f}cm，"
                f"支持实例={scene_support_count}"
            )

        for detection, raw_point_cloud in zip(
            target_detections,
            target_point_clouds,
        ):
            box = detection["box"]
            x1, y1, x2, y2 = np.round(box).astype(int).tolist()
            box_cx = (x1 + x2) // 2
            box_cy = (y1 + y2) // 2

            if len(raw_point_cloud) == 0:
                continue

            raw_point_count = int(len(raw_point_cloud))
            candidate_depths = [
                float(np.median(cluster[:, 2]))
                for cluster in split_point_cloud_depth_clusters(raw_point_cloud)
            ]
            point_cloud = filter_point_cloud(
                raw_point_cloud,
                scene_depth_reference_m=scene_depth_reference_m,
            )
            localization = fit_point_cloud_line(point_cloud)
            if localization is None:
                candidate_text = ", ".join(
                    f"{depth * 100:.1f}cm" for depth in candidate_depths
                ) or "无"
                print(
                    f"果梗 {detection['class_name']} 点云不足、线性度过低或线段退化，"
                    f"已跳过: 候选深度=[{candidate_text}], "
                    f"滤波前={raw_point_count}, 滤波后={len(point_cloud)}"
                )
                continue

            localization["raw_point_count"] = raw_point_count
            strategy = analyze_point_cloud_cut_strategy(
                detection,
                localization,
                main_stem_points,
                depth_intrin,
                eye,
                te,
            )
            if strategy is None:
                print("点云中心、P1/P2 或手眼转换无效，本实例已跳过")
                continue

            cx = strategy["cx"]
            cy = strategy["cy"]
            image_height, image_width = color_image.shape[:2]
            if not (0 <= cx < image_width and 0 <= cy < image_height):
                print(f"点云中心投影越界: ({cx},{cy})，本实例已跳过")
                continue

            xr = strategy["Xr"]
            yr = strategy["Yr"]
            zr = strategy["Zr"]
            if not (X_MIN <= xr <= X_MAX):
                print(
                    f"MASK_PCA 的 X 超出安全范围 [{X_MIN:.2f}, {X_MAX:.2f}]: "
                    f"{xr:.2f}，拒绝限幅并跳过"
                )
                continue
            if not (Y_MIN <= yr <= Y_MAX):
                print(
                    f"MASK_PCA 的 Y 超出安全范围 [{Y_MIN:.2f}, {Y_MAX:.2f}]: "
                    f"{yr:.2f}，已跳过"
                )
                continue
            if zr < 0:
                print(f"MASK_PCA 的 Z 为负数: {zr:.2f}，已跳过")
                continue

            target = {
                **strategy,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "box_cx": box_cx,
                "box_cy": box_cy,
                "depth_foreground_verified": True,
                "depth_reference_m": None,
                "scene_depth_reference_m": scene_depth_reference_m,
                "class_name": detection["class_name"],
                "score": detection["score"],
                "mask": detection["mask"],
                "contour": detection["contour"],
                "angle": strategy["target_angle"],
                "last_frame_id": frame_id,
                "count": 1,
            }
            target.pop("target_angle", None)
            localized_count += 1

            distance_text = "无主茎参考"
            if target["main_stem_distance_cm"] is not None:
                distance_text = f"{target['main_stem_distance_cm']:.2f}cm"
            print(
                "MASK点云定位成功: "
                f"类别={target['class_name']}, 置信度={target['score']:.2f}, "
                f"点数={target['point_count']}/{target['raw_point_count']}, "
                f"线性度={target['linearity']:.2f}, "
                f"线长={target['line_length_cm']:.2f}cm, "
                f"深度={target['depth_m'] * 100:.2f}cm, "
                f"主茎距离={distance_text}, "
                f"机械臂=({target['Xr']:.2f}, {target['Yr']:.2f}, {target['Zr']:.2f})"
            )

            if update_target(
                target,
                all_targets,
                distance_threshold=target_merge_distance_px,
            ):
                print(f"已更新目标 ({cx},{cy})")
            else:
                initialize_target_tracking(target)
                all_targets.append(target)
                print(f"新增目标 ({cx},{cy})")

        print(f"本帧三维定位成功: {localized_count}")
        save_result_image(color_image, all_targets, frame_id, capture_id)

        if should_stop_on_stable_depth(
            all_targets,
            frame_id + 1,
            accepted_modes=early_stop_depth_modes,
            min_stable_frames=early_stop_min_stable_frames,
            expected_depth_m=None,
            max_depth_error_m=None,
        ):
            print(
                "MASK_PCA 已获得稳定多帧点云，提前结束当前扫描位检测: "
                f"已处理 {frame_id + 1}/{num_frames} 帧"
            )
            break

    return [target for target in all_targets if target["count"] >= MIN_VOTE]

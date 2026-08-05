"""采摘流程模块。

这里负责目标的二次视觉校正和最终动作执行。
第一次检测得到的是拍照位下的粗定位；移动到预瞄位后再检测一次，
可以减小由于相机视角、机械臂运动误差和深度噪声带来的偏差。
"""

import math
import time

import numpy as np

from config1 import (
    ACTIVE_RESCAN_ACCEPT_DEPTH_MODES,
    ACTIVE_RESCAN_BACKGROUND_RING_MIN_POINTS,
    ACTIVE_RESCAN_BACKGROUND_RING_WIDTH_PX,
    ACTIVE_RESCAN_CAMERA_Z_OFFSET_CM,
    ACTIVE_RESCAN_FRAME_MATCH_DISTANCE_PX,
    ACTIVE_RESCAN_MATCH_MAX_YZ_DISTANCE_CM,
    ACTIVE_RESCAN_MIN_STABLE_FRAMES,
    ACTIVE_RESCAN_MIN_MASK_DEPTH_CONTRAST_M,
    ACTIVE_RESCAN_REFERENCE_MAX_ERROR_M,
    ACTIVE_RESCAN_REFERENCE_X_SCALE,
    ACTIVE_RESCAN_SETTLE_SECONDS,
    ACTIVE_RESCAN_TARGET_STANDOFF_CM,
    ACTIVE_RESCAN_USE_FALLBACK_AFTER_FAILURE,
    ACTIVE_RESCAN_X_ADVANCE_CM,
    ENABLE_BLADE_CONTACT_OFFSET,
    ENABLE_PICK_PREAIM,
    GUIDE_FORWARD_X_CM,
    GUIDE_INSERT_DEPTH_CM,
    GUIDE_PUSH_AWAY_CM,
    MAX_DEPTH,
    MIN_DEPTH,
    PICK_TARGET_Y_CALIBRATION_OFFSET_CM,
    PICK_TARGET_Z_CALIBRATION_OFFSET_CM,
    PICK_PREAIM_SETTLE_SECONDS,
    PICK_PREAIM_X_OFFSET,
    REFINE_FRAMES,
    RETREAT_Y_LOWER_MAX,
    RETREAT_Y_LOWER_MIN,
    RETREAT_Y_UPPER_MAX,
    RETREAT_Y_UPPER_MIN,
    RETREAT_Z_MAX,
    RETREAT_Z_MIN,
    ROTATE_MOVE_SEND_INTERVAL_SECONDS,
    TARGET_DEPTH_REFERENCE_M,
    X_MAX,
    X_MIN,
    Y_MAX,
    Y_MIN,
)
from robot import (
    has_negative_coordinate,
    send_end_effector_action,
    send_robot_position,
    send_rotate_and_robot_position,
)
from vision import detect_targets, find_stable_depth_observation

RETREAT_X_STEP = 20.0
RETREAT_STEP_INTERVAL_SECONDS = 0.2


def clamp_retreat_position(y, z):
    """把释放点限制在两个合法 Y 区间，X 轴仍沿用原有逻辑。"""
    retreat_y = max(RETREAT_Y_LOWER_MIN, min(RETREAT_Y_UPPER_MAX, y))
    if RETREAT_Y_LOWER_MAX < retreat_y < RETREAT_Y_UPPER_MIN:
        lower_distance = retreat_y - RETREAT_Y_LOWER_MAX
        upper_distance = RETREAT_Y_UPPER_MIN - retreat_y
        retreat_y = (
            RETREAT_Y_LOWER_MAX
            if lower_distance <= upper_distance
            else RETREAT_Y_UPPER_MIN
        )
    retreat_z = max(RETREAT_Z_MIN, min(RETREAT_Z_MAX, z))
    return retreat_y, retreat_z


def clamp_pick_position(x, y, z):
    """限制导入剪切过程中的机械臂目标点。"""
    clamped_x = max(X_MIN, min(X_MAX, x))
    clamped_y = max(Y_MIN, min(Y_MAX, y))
    clamped_z = max(0.0, z)
    return clamped_x, clamped_y, clamped_z


def select_stable_fruit_depth_target(target, expected_depth_m=None):
    """从目标多帧观测中选出稳定的果梗自身深度。"""
    observations = target.get("observations", [])
    if not observations:
        observations = [target]

    if expected_depth_m is not None:
        allowed_error = max(0.0, float(ACTIVE_RESCAN_REFERENCE_MAX_ERROR_M))
        observations = [
            observation for observation in observations
            if observation.get("depth_m") is not None
            and abs(
                float(observation["depth_m"]) - float(expected_depth_m)
            ) <= allowed_error
        ]

    accepted_modes = {str(mode) for mode in ACTIVE_RESCAN_ACCEPT_DEPTH_MODES}
    best_observation, stable_count = find_stable_depth_observation(
        observations,
        lambda mode: str(mode) in accepted_modes,
        max(1, int(ACTIVE_RESCAN_MIN_STABLE_FRAMES)),
    )

    if best_observation is None:
        return None

    stable_target = dict(target)
    stable_target.update(best_observation)
    stable_target["depth_quality"] = "稳定果梗自身深度"
    stable_target["stable_depth_count"] = stable_count
    return stable_target


def stem_angle_difference(angle_a, angle_b):
    """计算两条无方向果梗轴线之间的最小角度差。"""
    difference = (float(angle_a) - float(angle_b) + 90.0) % 180.0 - 90.0
    return abs(difference)


def target_yz_distance(candidate, reference):
    """只比较 Y/Z，避免第一次错误深度造成的 X 偏差干扰目标匹配。"""
    return math.hypot(
        float(candidate["Yr"]) - float(reference["Yr"]),
        float(candidate["Zr"]) - float(reference["Zr"]),
    )


def find_matching_rescan_target(
    candidates,
    reference,
    require_stable_depth,
    expected_depth_m=None,
):
    """在新视角中匹配同一果梗，不使用第一次可能错误的 X 坐标。"""
    best_target = None
    best_score = None
    reference_class = str(reference.get("class_name", ""))
    reference_angle = float(reference.get("angle", 0.0))

    for candidate in candidates:
        selected = candidate
        if require_stable_depth:
            selected = select_stable_fruit_depth_target(
                candidate,
                expected_depth_m=expected_depth_m,
            )
            if selected is None:
                continue

        yz_distance = target_yz_distance(selected, reference)
        if yz_distance > ACTIVE_RESCAN_MATCH_MAX_YZ_DISTANCE_CM:
            continue

        class_mismatch = int(
            str(selected.get("class_name", "")) != reference_class
        )
        angle_difference = stem_angle_difference(
            selected.get("angle", 0.0),
            reference_angle,
        )
        score = (class_mismatch, yz_distance, angle_difference)

        if best_score is None or score < best_score:
            best_score = score
            best_target = selected

    return best_target


def build_active_rescan_position(scan_position, target):
    """根据首次扫描点和粗目标坐标生成安全的近距离二次扫描点。"""
    scan_x, _scan_y, _scan_z = map(float, scan_position)
    rough_x = float(target["Xr"])
    rough_y = float(target["Yr"])
    rough_z = float(target["Zr"])

    x_from_scan = scan_x + ACTIVE_RESCAN_X_ADVANCE_CM
    x_with_standoff = rough_x - ACTIVE_RESCAN_TARGET_STANDOFF_CM
    rescan_x = min(x_from_scan, x_with_standoff)

    # 粗目标过近时不向扫描点后方移动，只保持原扫描 X。
    rescan_x = max(scan_x, rescan_x)
    rescan_x = max(X_MIN, min(X_MAX, rescan_x))
    rescan_y = max(Y_MIN, min(Y_MAX, rough_y))
    rescan_z = max(0.0, rough_z - ACTIVE_RESCAN_CAMERA_Z_OFFSET_CM)

    return rescan_x, rescan_y, rescan_z


def get_target_navigation_depth(target):
    """返回该目标自己的粗深度，优先使用主茎观测而不是全局参考值。"""
    observations = target.get("observations", [])
    main_observation, main_count = find_stable_depth_observation(
        observations,
        lambda mode: str(mode) == "MAIN_STEM",
        1,
    )

    if main_observation is not None:
        return (
            float(main_observation["depth_m"]),
            f"目标主茎深度({main_count}帧)",
        )

    if target.get("depth_m") is not None:
        return (
            float(target["depth_m"]),
            f"首次目标深度({target.get('depth_mode', '未知')})",
        )

    return float(TARGET_DEPTH_REFERENCE_M), "全局人工参考深度"


def calculate_active_rescan_reference_depth(
    scan_position,
    rescan_position,
    target,
):
    """用目标自己的粗深度减去相机 X 前进量，得到近拍参考深度。"""
    scan_x = float(scan_position[0])
    rescan_x = float(rescan_position[0])
    forward_distance_m = max(0.0, rescan_x - scan_x) / 100.0
    base_depth_m, base_depth_source = get_target_navigation_depth(target)
    reference_depth_m = (
        base_depth_m
        - forward_distance_m * ACTIVE_RESCAN_REFERENCE_X_SCALE
    )

    lower_limit = float(MIN_DEPTH) + 1e-3
    upper_limit = float(MAX_DEPTH) - 1e-3
    reference_depth_m = max(lower_limit, min(upper_limit, reference_depth_m))
    return reference_depth_m, base_depth_m, base_depth_source


def capture_targets_at_rescan_position(
    pipeline,
    align,
    model,
    eye,
    ser,
    rescan_position,
    depth_reference_m,
    label,
):
    """移动到指定复拍位，使用该位置对应的新 te 重新检测目标。"""
    rescan_x, rescan_y, rescan_z = rescan_position
    print(
        f"\n{label}: "
        f"({rescan_x:.2f}, {rescan_y:.2f}, {rescan_z:.2f})"
    )
    print(f"当前复拍动态参考深度: {depth_reference_m * 100:.1f}cm")

    if not send_robot_position(ser, rescan_x, rescan_y, rescan_z):
        print(f"{label}移动失败")
        return None

    print(f"等待机械臂和相机稳定 {ACTIVE_RESCAN_SETTLE_SECONDS:.1f} 秒")
    time.sleep(ACTIVE_RESCAN_SETTLE_SECONDS)

    te_rescan = np.array([[rescan_x], [rescan_y], [rescan_z]])
    return detect_targets(
        pipeline,
        align,
        model,
        eye,
        te_rescan,
        num_frames=REFINE_FRAMES,
        early_stop_depth_modes=ACTIVE_RESCAN_ACCEPT_DEPTH_MODES,
        early_stop_min_stable_frames=ACTIVE_RESCAN_MIN_STABLE_FRAMES,
        target_merge_distance_px=ACTIVE_RESCAN_FRAME_MATCH_DISTANCE_PX,
        depth_reference_m=depth_reference_m,
        depth_reference_fallback_trigger_m=ACTIVE_RESCAN_REFERENCE_MAX_ERROR_M,
        depth_candidate_reference_max_error_m=(
            ACTIVE_RESCAN_REFERENCE_MAX_ERROR_M
        ),
        mask_background_ring_width_px=ACTIVE_RESCAN_BACKGROUND_RING_WIDTH_PX,
        mask_background_ring_min_points=(
            ACTIVE_RESCAN_BACKGROUND_RING_MIN_POINTS
        ),
        min_mask_background_depth_contrast_m=(
            ACTIVE_RESCAN_MIN_MASK_DEPTH_CONTRAST_M
        ),
        early_stop_reference_max_error_m=ACTIVE_RESCAN_REFERENCE_MAX_ERROR_M,
        use_reference_x_recheck=False,
        use_main_stem_depth_fallback=False,
        capture_label=(
            f"主动近拍_X{rescan_x:.2f}_Y{rescan_y:.2f}_Z{rescan_z:.2f}"
        ),
    )


def refine_target(pipeline, align, model, eye, ser, target, scan_position):
    """深度不可靠时主动靠近复拍，并优先返回果梗自身真实深度。"""
    stable_initial_target = select_stable_fruit_depth_target(
        target,
        expected_depth_m=None,
    )
    if stable_initial_target is not None:
        print(
            "\n首次扫描已有稳定果梗自身深度，"
            f"共 {stable_initial_target['stable_depth_count']} 帧，无需靠近复拍"
        )
        stable_initial_target["active_rescan_status"] = "首次扫描真实深度"
        return stable_initial_target

    print("\n首次扫描未获得稳定果梗自身深度，开始主动靠近复拍")
    print(
        f"首次深度来源: {target.get('depth_mode', '未知')}, "
        f"质量: {target.get('depth_quality', '未知')}"
    )

    primary_position = build_active_rescan_position(scan_position, target)
    (
        primary_reference_depth_m,
        navigation_depth_m,
        navigation_depth_source,
    ) = calculate_active_rescan_reference_depth(
        scan_position,
        primary_position,
        target,
    )
    print(
        "二次扫描参考基准: "
        f"{navigation_depth_source}={navigation_depth_m * 100:.1f}cm"
    )
    primary_targets = capture_targets_at_rescan_position(
        pipeline,
        align,
        model,
        eye,
        ser,
        primary_position,
        primary_reference_depth_m,
        "移动到近距离二次扫描点",
    )
    if primary_targets is None:
        print("二次扫描点未确认到位，为避免使用未知机械臂位置，已跳过该目标")
        return None

    stable_target = find_matching_rescan_target(
        primary_targets,
        target,
        require_stable_depth=True,
        expected_depth_m=primary_reference_depth_m,
    )
    if stable_target is not None:
        stable_target["active_rescan_status"] = "二次扫描真实深度"
        print(
            "\n二次扫描已获得稳定果梗自身深度: "
            f"{stable_target['stable_depth_count']} 帧, "
            f"坐标=({stable_target['Xr']:.2f}, "
            f"{stable_target['Yr']:.2f}, {stable_target['Zr']:.2f})"
        )
        return stable_target

    print("\n主动复拍仍未获得稳定果梗自身深度")

    if not ACTIVE_RESCAN_USE_FALLBACK_AFTER_FAILURE:
        print("当前配置禁止使用兜底深度，已跳过该目标")
        return None

    fallback_target = dict(target)
    fallback_target["active_rescan_status"] = "复拍失败，沿用首次结果"
    print(
        "二次扫描的主茎/参考深度不作为新结果，直接沿用第一次结果: "
        f"来源={fallback_target.get('depth_mode', '未知')}, "
        f"坐标=({fallback_target['Xr']:.2f}, "
        f"{fallback_target['Yr']:.2f}, {fallback_target['Zr']:.2f})"
    )
    return fallback_target


def execute_pick(ser, target):
    """按校正后的坐标和姿态执行一次采摘动作。"""
    x = target["Xr"]
    raw_y = target["Yr"]
    y = raw_y + PICK_TARGET_Y_CALIBRATION_OFFSET_CM
    raw_z = target["Zr"]
    z = raw_z + PICK_TARGET_Z_CALIBRATION_OFFSET_CM
    angle = target["angle"]
    cut_mode = target.get("cut_mode", "normal")
    fruit_side = target.get("fruit_side", "unknown")
    contact_blade = target.get("contact_blade", "static")
    guide_away_y = float(target.get("guide_away_y", 0.0))
    guide_away_z = float(target.get("guide_away_z", 0.0))

    if has_negative_coordinate(x, y, z):
        print("\n采摘目标坐标出现负数，已跳过该目标:")
        print(f"x={x:.2f}, y={y:.2f}, z={z:.2f}")
        return False

    print("\n开始执行采摘...")
    if abs(PICK_TARGET_Y_CALIBRATION_OFFSET_CM) > 1e-9:
        print(
            "Y轴标定偏置已应用: "
            f"{raw_y:.2f} + ({PICK_TARGET_Y_CALIBRATION_OFFSET_CM:.2f}) "
            f"= {y:.2f} 厘米"
        )
    if abs(PICK_TARGET_Z_CALIBRATION_OFFSET_CM) > 1e-9:
        print(
            "Z轴标定偏置已应用: "
            f"{raw_z:.2f} + ({PICK_TARGET_Z_CALIBRATION_OFFSET_CM:.2f}) "
            f"= {z:.2f} 厘米"
        )
    print(
        "动作顺序: "
        "05旋转+04预瞄并行 -> 04采摘 -> 06闭合 -> "
        f"04沿X退{RETREAT_X_STEP:g}厘米 -> "
        f"05复位+04退回({X_MIN:.1f},"
        f"Y限{RETREAT_Y_LOWER_MIN:g}-{RETREAT_Y_LOWER_MAX:g}或"
        f"{RETREAT_Y_UPPER_MIN:g}-{RETREAT_Y_UPPER_MAX:g},"
        f"Z限{RETREAT_Z_MIN:g}-{RETREAT_Z_MAX:g}) -> 06张开"
    )
    print(f"剪切模式: {cut_mode}")
    print(f"果梗相对主茎: {fruit_side}")
    if ENABLE_BLADE_CONTACT_OFFSET:
        print(f"贴近刀片: {contact_blade}")
    else:
        print("定位基准: 旋转中心（静刀/动刀横向偏置已关闭）")

    if ENABLE_PICK_PREAIM:
        x_pre = max(X_MIN, x - PICK_PREAIM_X_OFFSET)

        print(
            "\n移动到采摘预瞄位: "
            f"({x_pre:.2f}, {y:.2f}, {z:.2f})"
        )

        if not send_rotate_and_robot_position(
            ser,
            angle,
            x_pre,
            y,
            z,
            send_interval_seconds=ROTATE_MOVE_SEND_INTERVAL_SECONDS,
        ):
            return False

        print(f"\n预瞄位稳定等待 {PICK_PREAIM_SETTLE_SECONDS:.1f} 秒")
        time.sleep(PICK_PREAIM_SETTLE_SECONDS)

        print(f"\n移动到采摘目标点: ({x:.2f}, {y:.2f}, {z:.2f})")
        if not send_robot_position(ser, x, y, z):
            return False
    else:
        print(f"\n无预瞄：旋转与采摘目标移动并行: ({x:.2f}, {y:.2f}, {z:.2f})")
        if not send_rotate_and_robot_position(
            ser,
            angle,
            x,
            y,
            z,
            send_interval_seconds=ROTATE_MOVE_SEND_INTERVAL_SECONDS,
        ):
            return False

    if cut_mode == "guide":
        insert_x, insert_y, insert_z = clamp_pick_position(
            x + GUIDE_INSERT_DEPTH_CM,
            y,
            z,
        )

        print(
            "\n近主茎导入: 先插入刀口 "
            f"({insert_x:.2f}, {insert_y:.2f}, {insert_z:.2f})"
        )
        if not send_robot_position(ser, insert_x, insert_y, insert_z):
            return False

        guide_x, guide_y, guide_z = clamp_pick_position(
            insert_x + GUIDE_FORWARD_X_CM,
            insert_y + guide_away_y * GUIDE_PUSH_AWAY_CM,
            insert_z + guide_away_z * GUIDE_PUSH_AWAY_CM,
        )

        print(
            "\n近主茎导入: 推离主茎并前送 X "
            f"({guide_x:.2f}, {guide_y:.2f}, {guide_z:.2f})"
        )
        if not send_robot_position(ser, guide_x, guide_y, guide_z):
            return False

        x, y, z = guide_x, guide_y, guide_z

    # 到达目标点后闭合末端，夹住果梗；这里不能立刻张开。
    print("\n末端闭合，夹住果梗")
    if not send_end_effector_action(ser, 0x02):
        return False

    # 剪完果梗后先只让 X 轴按配置距离后退，Y/Z 保持采摘点不变。
    step_retreat_x = max(X_MIN, x - RETREAT_X_STEP)

    print(
        "\nX 轴先单独回退: "
        f"({step_retreat_x:.2f}, {y:.2f}, {z:.2f})"
    )
    if not send_robot_position(ser, step_retreat_x, y, z):
        return False

    time.sleep(RETREAT_STEP_INTERVAL_SECONDS)

    # 采摘后让 X 轴保持原回退逻辑，Y/Z 限制在释放圣女果的安全范围内。
    retreat_y, retreat_z = clamp_retreat_position(y, z)

    if retreat_y != y or retreat_z != z:
        print(
            "\n回退释放点已限幅: "
            f"Y {y:.2f}->{retreat_y:.2f}, "
            f"Z {z:.2f}->{retreat_z:.2f}"
        )

    print(
        "\n旋转关节复位，同时机械臂退回释放点: "
        f"({X_MIN:.1f}, {retreat_y:.2f}, {retreat_z:.2f})，"
        "末端保持闭合"
    )
    if not send_rotate_and_robot_position(
        ser,
        0,
        X_MIN,
        retreat_y,
        retreat_z,
        send_interval_seconds=ROTATE_MOVE_SEND_INTERVAL_SECONDS,
    ):
        return False

    # 到达回退位后再张开/松开，释放果梗。
    print("\n末端张开，松开果梗")
    if not send_end_effector_action(ser, 0x01):
        return False

    return True

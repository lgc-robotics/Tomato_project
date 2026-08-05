"""整机联调入口。

主流程：
1. 将机械臂停到安全扫描位。
2. 让 STM32 执行一次定时底盘移动，同时 Python 持续发送循迹偏差。
3. 在当前停车点用机械臂和 RealSense 扫描。
4. 对检测到的有效圣女果目标执行采摘。
5. 按规划间距重复，直到覆盖设定轨道长度。
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time

import cv2
import numpy as np
import pyrealsense2 as rs
import serial
from ultralytics import YOLO

from calibration import Eye_in_hand
from config1 import (
    BAUDRATE,
    CHASSIS_BAUDRATE,
    CHASSIS_CAMERA_HEIGHT,
    CHASSIS_CAMERA_INDEX,
    CHASSIS_CAMERA_WARMUP_FRAMES,
    CHASSIS_CAMERA_WIDTH,
    CHASSIS_DEBUG_DISPLAY_INTERVAL_SECONDS,
    CHASSIS_ERROR_FILTER_WINDOW,
    CHASSIS_LOG_INTERVAL_SECONDS,
    CHASSIS_MAX_CONTOUR_AREA_RATIO,
    CHASSIS_MAX_ERROR_STEP_PX,
    CHASSIS_MIN_CONTOUR_AREA_PX,
    CHASSIS_OPEN_CAMERA_AT_START,
    CHASSIS_RELEASE_CAMERA_DURING_PICK,
    CHASSIS_SEND_INTERVAL,
    CHASSIS_SERIAL_PORT,
    CHASSIS_SERIAL_TIMEOUT,
    CHASSIS_SHOW_DEBUG,
    CHASSIS_START_STABLE_FRAMES,
    CHASSIS_START_STABLE_SPREAD_PX,
    CHASSIS_START_STABLE_TIMEOUT_SECONDS,
    CHASSIS_MOVE_ACK_TIMEOUT_SECONDS,
    COLOR_HEIGHT,
    COLOR_WIDTH,
    DEPTH_HEIGHT,
    DEPTH_WIDTH,
    ENABLE_BLADE_CONTACT_OFFSET,
    ENABLE_RUN_RECORDING,
    FPS,
    MODEL_PATH,
    NUM_FRAMES,
    PICK_BEFORE_FIRST_MOVE,
    SCAN_SETTLE_SECONDS,
    SCAN_X,
    SCAN_Y_MAX,
    SCAN_Y_START,
    SCAN_Y_STEP,
    SCAN_Z_MAX,
    SCAN_Z_START,
    SCAN_Z_STEP,
    SERIAL_PORT,
    SERIAL_TIMEOUT,
    ROBOT_ACK_TIMEOUT_SECONDS,
    RUN_RECORD_FOLDER_NAME,
    RUN_REPORT_MAX_LOG_CHARS_PER_IMAGE,
    STATION_SPACING_METERS,
    STATION_SETTLE_SECONDS,
    TRACK_TOTAL_METERS,
    USE_REFINE_PICK,
)
from pick1 import execute_pick, refine_target
from robot import send_robot_position
from run_recorder import finalize_run_recorder, start_run_recorder
from test8 import LineFollowerChassis
from vision import detect_targets


def create_pipeline():
    """创建并启动 RealSense RGB-D 数据流。"""
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
    print("RealSense 已启动")
    return pipeline, align


def create_arm_serial():
    """打开机械臂串口。"""
    arm_ser = serial.Serial(
        port=SERIAL_PORT,
        baudrate=BAUDRATE,
        timeout=SERIAL_TIMEOUT,
    )
    print(f"机械臂串口已打开: {SERIAL_PORT}")
    return arm_ser


def create_chassis(shared_ser=None):
    """创建底盘循迹控制对象。"""
    use_shared_serial = (
        shared_ser is not None
        and SERIAL_PORT.upper() == CHASSIS_SERIAL_PORT.upper()
    )

    chassis = LineFollowerChassis(
        port=CHASSIS_SERIAL_PORT,
        baudrate=CHASSIS_BAUDRATE,
        timeout=CHASSIS_SERIAL_TIMEOUT,
        camera_index=CHASSIS_CAMERA_INDEX,
        frame_width=CHASSIS_CAMERA_WIDTH,
        frame_height=CHASSIS_CAMERA_HEIGHT,
        send_interval=CHASSIS_SEND_INTERVAL,
        camera_warmup_frames=CHASSIS_CAMERA_WARMUP_FRAMES,
        start_stable_frames=CHASSIS_START_STABLE_FRAMES,
        start_stable_spread_px=CHASSIS_START_STABLE_SPREAD_PX,
        start_stable_timeout_seconds=CHASSIS_START_STABLE_TIMEOUT_SECONDS,
        error_filter_window=CHASSIS_ERROR_FILTER_WINDOW,
        max_error_step_px=CHASSIS_MAX_ERROR_STEP_PX,
        min_contour_area_px=CHASSIS_MIN_CONTOUR_AREA_PX,
        max_contour_area_ratio=CHASSIS_MAX_CONTOUR_AREA_RATIO,
        log_interval_seconds=CHASSIS_LOG_INTERVAL_SECONDS,
        show_debug=CHASSIS_SHOW_DEBUG,
        debug_display_interval_seconds=CHASSIS_DEBUG_DISPLAY_INTERVAL_SECONDS,
        ser=shared_ser if use_shared_serial else None,
        close_serial_on_close=not use_shared_serial,
    )

    if CHASSIS_OPEN_CAMERA_AT_START:
        chassis.open()
    else:
        print(
            "底盘摄像头仅在定时移动时打开: "
            f"编号={CHASSIS_CAMERA_INDEX}"
        )

    if CHASSIS_SHOW_DEBUG:
        print("底盘调试窗口已启用: 摄像头 + 掩膜")
    return chassis


def iter_axis_positions(start, max_value, step):
    """按步长生成轴向位置，并确保包含配置的终点。"""
    if step <= 0:
        raise ValueError("扫描步长必须大于 0")

    value = start
    last_value = None

    while value <= max_value + 1e-9:
        yield round(value, 4)
        last_value = value
        value += step

    if last_value is None or abs(last_value - max_value) > 1e-6:
        yield round(max_value, 4)


def iter_scan_positions():
    """生成一个停车点内的机械臂扫描位。"""
    z_positions = list(iter_axis_positions(SCAN_Z_START, SCAN_Z_MAX, SCAN_Z_STEP))
    y_positions = list(iter_axis_positions(SCAN_Y_START, SCAN_Y_MAX, SCAN_Y_STEP))

    for row_index, z in enumerate(z_positions):
        current_y_positions = y_positions
        if row_index % 2 == 1:
            current_y_positions = reversed(y_positions)

        for y in current_y_positions:
            yield SCAN_X, y, z


def iter_station_moves(total_meters, station_spacing_meters):
    """生成直线轨道上的移动 / 采摘循环。"""
    if total_meters <= 0:
        raise ValueError("轨道总长度必须大于 0")

    if station_spacing_meters <= 0:
        raise ValueError("停车间距必须大于 0")

    planned_covered = 0.0
    move_index = 1

    while planned_covered < total_meters - 1e-9:
        planned_distance = min(station_spacing_meters, total_meters - planned_covered)
        planned_covered += planned_distance
        yield move_index, planned_distance, planned_covered
        move_index += 1


def print_targets(targets):
    print("\n===================================")
    print("当前扫描位有效目标")
    print("===================================")

    for i, target in enumerate(targets):
        print(f"\n目标 {i + 1}")
        print(f"投票次数: {target['count']}")
        print(f"剪切模式: {target.get('cut_mode', 'normal')}")
        print(f"果梗相对主茎: {target.get('fruit_side', 'unknown')}")
        if ENABLE_BLADE_CONTACT_OFFSET:
            print(f"贴近刀片: {target.get('contact_blade', 'static')}")
        else:
            print("定位基准: 旋转中心（静刀/动刀横向偏置已关闭）")
        print(
            "深度来源: "
            f"{target.get('depth_mode', '未知')} / "
            f"{target.get('depth_quality', '未知')}"
        )
        if target.get("main_stem_distance_cm") is not None:
            print(f"主茎最近距离: {target['main_stem_distance_cm']:.1f} 厘米")
        print(f"果梗倾角: {target.get('stem_tilt_angle', 0):.1f} 度")
        print(f"末端旋转角: {target['angle']} 度")
        print(
            "机械臂坐标: "
            f"({target['Xr']:.2f}, "
            f"{target['Yr']:.2f}, "
            f"{target['Zr']:.2f}) 厘米"
        )


def close_final_window():
    try:
        cv2.destroyWindow("最终结果")
    except cv2.error:
        pass


def park_arm_for_chassis_motion(arm_ser):
    """底盘移动前，将机械臂移动到第一个扫描位。"""
    travel_pose = (SCAN_X, SCAN_Y_START, SCAN_Z_START)
    print(
        "\n底盘移动前停放机械臂: "
        f"({travel_pose[0]:.2f}, {travel_pose[1]:.2f}, {travel_pose[2]:.2f}) 厘米"
    )
    return send_robot_position(
        arm_ser,
        *travel_pose,
        ack_timeout_seconds=ROBOT_ACK_TIMEOUT_SECONDS,
    )


def scan_and_pick_at_position(
    pipeline,
    align,
    model,
    eye,
    arm_ser,
    scan_position,
    use_refine_pick=False,
):
    """移动到一个扫描位，检测目标，并立即执行采摘。"""
    scan_x, scan_y, scan_z = scan_position

    print("\n===================================")
    print(f"机械臂扫描位: ({scan_x:.2f}, {scan_y:.2f}, {scan_z:.2f}) 厘米")
    print("===================================")

    if not send_robot_position(
        arm_ser,
        scan_x,
        scan_y,
        scan_z,
        ack_timeout_seconds=ROBOT_ACK_TIMEOUT_SECONDS,
    ):
        print("扫描位无效或未收到到位确认信号，已跳过")
        return {
            "detected": 0,
            "picked": 0,
        }

    print("\n扫描位已到达，等待相机和机械臂稳定")
    time.sleep(SCAN_SETTLE_SECONDS)

    te = np.array([[scan_x], [scan_y], [scan_z]])

    final_targets = detect_targets(
        pipeline,
        align,
        model,
        eye,
        te,
        num_frames=NUM_FRAMES,
        capture_label=(
            f"首次扫描_X{scan_x:.2f}_Y{scan_y:.2f}_Z{scan_z:.2f}"
        ),
    )

    if len(final_targets) == 0:
        print("当前扫描位未检测到目标")
        return {
            "detected": 0,
            "picked": 0,
        }

    print_targets(final_targets)

    picked = 0

    for target in final_targets:
        pick_target = target

        if use_refine_pick:
            refined_target = refine_target(
                pipeline,
                align,
                model,
                eye,
                arm_ser,
                target,
                scan_position,
            )

            if refined_target is None:
                continue

            pick_target = refined_target

        if execute_pick(arm_ser, pick_target):
            picked += 1

    return {
        "detected": len(final_targets),
        "picked": picked,
    }


def pick_current_station(pipeline, align, model, eye, arm_ser, station_index):
    """在一个底盘停车点执行所有配置的扫描位。"""
    print("\n##################################################")
    print(f"停车点 {station_index}: 开始扫描和采摘")
    print("##################################################")

    station_detected = 0
    station_picked = 0

    for scan_position in iter_scan_positions():
        result = scan_and_pick_at_position(
            pipeline,
            align,
            model,
            eye,
            arm_ser,
            scan_position,
            use_refine_pick=USE_REFINE_PICK,
        )
        station_detected += result["detected"]
        station_picked += result["picked"]

    print("\n停车点汇总:")
    print(f"检测目标数: {station_detected}")
    print(f"采摘目标数: {station_picked}")

    return {
        "detected": station_detected,
        "picked": station_picked,
    }


def prepare_vision_scan(chassis):
    """采摘前释放底盘摄像头，减少 USB 占用。"""
    if CHASSIS_RELEASE_CAMERA_DURING_PICK and chassis is not None:
        print("RealSense 扫描前释放底盘摄像头")
        chassis.close_camera()


def main():
    print("\n正在加载 YOLO 模型...")
    model = YOLO(MODEL_PATH)
    print("YOLO 模型加载完成")

    arm_ser = None
    chassis = None
    pipeline = None
    align = None

    total_detected = 0
    total_picked = 0
    station_index = 0

    try:
        arm_ser = create_arm_serial()
        chassis = create_chassis(shared_ser=arm_ser)
        eye = Eye_in_hand()
        pipeline, align = create_pipeline()

        print("\n整机联调已启动")
        print(f"轨道总长度: {TRACK_TOTAL_METERS:.2f} 米")
        print(f"计划停车间距: {STATION_SPACING_METERS:.2f} 米")
        print(
            "底盘移动确认信号超时: "
            f"{CHASSIS_MOVE_ACK_TIMEOUT_SECONDS:.1f} 秒"
        )

        if PICK_BEFORE_FIRST_MOVE:
            station_index += 1
            prepare_vision_scan(chassis)
            station_result = pick_current_station(
                pipeline,
                align,
                model,
                eye,
                arm_ser,
                station_index,
            )
            total_detected += station_result["detected"]
            total_picked += station_result["picked"]

        for move_index, planned_distance_m, planned_covered_m in iter_station_moves(
            TRACK_TOTAL_METERS,
            STATION_SPACING_METERS,
        ):
            if not park_arm_for_chassis_motion(arm_ser):
                print(
                    "警告: 机械臂停放确认信号异常，继续整机流程"
                )

            print("\n==================================================")
            print(
                f"轨道移动 {move_index}: "
                f"计划间距 {planned_distance_m:.2f} 米, "
                f"计划已覆盖 {planned_covered_m:.2f}/{TRACK_TOTAL_METERS:.2f} 米"
            )
            print("==================================================")

            chassis_result = chassis.run_timed_move(
                ack_timeout_seconds=CHASSIS_MOVE_ACK_TIMEOUT_SECONDS,
            )

            if not chassis_result.get("started", True):
                print("警告: 循迹黑线未稳定，本次底盘移动已取消，结束后续站点流程")
                break

            if not chassis_result["ack_received"]:
                print("警告: 底盘定时移动因超时结束，未收到确认信号")

            print(
                f"\n底盘定时移动结束，等待 {STATION_SETTLE_SECONDS:.1f} 秒后开始采摘"
            )
            time.sleep(STATION_SETTLE_SECONDS)

            station_index += 1
            prepare_vision_scan(chassis)
            station_result = pick_current_station(
                pipeline,
                align,
                model,
                eye,
                arm_ser,
                station_index,
            )
            total_detected += station_result["detected"]
            total_picked += station_result["picked"]

        print("\n整机任务完成")
        print(f"总检测目标数: {total_detected}")
        print(f"总采摘目标数: {total_picked}")

    except KeyboardInterrupt:
        print("\n用户停止程序")

    finally:
        if chassis is not None:
            chassis.close()

        if pipeline is not None:
            pipeline.stop()

        cv2.destroyAllWindows()

        if arm_ser is not None:
            arm_ser.close()

        print("\n程序已结束")


if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.abspath(__file__))
    run_dir = start_run_recorder(
        project_dir,
        folder_name=RUN_RECORD_FOLDER_NAME,
        enabled=ENABLE_RUN_RECORDING,
        max_log_chars_per_image=RUN_REPORT_MAX_LOG_CHARS_PER_IMAGE,
    )

    try:
        main()
    finally:
        if run_dir is not None:
            print("\n正在生成本次运行的图文报告...")

        report_path = finalize_run_recorder()
        if report_path is not None:
            print(f"运行记录已保存: {report_path.parent}")
            print(f"图文报告: {report_path}")

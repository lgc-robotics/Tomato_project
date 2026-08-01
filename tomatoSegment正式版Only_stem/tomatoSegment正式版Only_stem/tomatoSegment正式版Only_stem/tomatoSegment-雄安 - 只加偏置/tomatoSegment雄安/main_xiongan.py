"""
雄安走停走版本：
1. 沿地面黑色胶带前进一段距离
2. 停下来遍历所有扫描点
3. 每个扫描点识别到果实就采摘，没有识别到就换下一个点
4. 扫描点遍历完，机械臂回初始位置，然后继续前进
"""
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import ctypes
import time

import cv2
import numpy as np
import pyrealsense2 as rs

from config import Config
from controller import RobotController
from locator import Locator
from navigation import Navigation
from robot import RobotKinematics
from xiongan_utils import XionganUtils


class RealSenseFrameError(RuntimeError):
    """RealSense连续采集或对齐失败，提示上层重启相机管线。"""


def cal_traj_for_fruit(
    robot,
    current_joints_rad,
    current_pose,
    target_pose,
    previous_pose,
    steps=3,
    log_context=None,
):
    prefix = XionganUtils.path_log_prefix(log_context)  # 打印日志
    use_previous_pose = bool(Config.use_previous_pose)

    print(f"\n{prefix} 开始直线插补逆解")
    print(f"{prefix} 初始扫描点pose[mm,rad]: {XionganUtils.format_pose(current_pose)}")
    print(f"{prefix} 初始扫描点关节角[deg]: {XionganUtils.format_joints_deg(current_joints_rad)}")
    if use_previous_pose:
        print(f"{prefix} 预采摘点pose[mm,rad]: {XionganUtils.format_pose(previous_pose)}")
    else:
        print(f"{prefix} 预采摘点已关闭，从初始扫描点直线插补到采摘目标点")
    print(f"{prefix} 采摘目标点pose[mm,rad]: {XionganUtils.format_pose(target_pose)}")

    if use_previous_pose:
        traj_poses = robot.linear_interpolation(current_pose, previous_pose, steps=steps)
        traj_poses.append(target_pose)
    else:
        traj_poses = robot.linear_interpolation(current_pose, target_pose, steps=steps)
    traj_joints = []
    successful_pose_records = []
    last_joints_rad = current_joints_rad
    ik_success = True
    ik_failed_step = 0

    for i, pose in enumerate(traj_poses):
        if i == len(traj_poses) - 1:
            step_role = "采摘目标点"
        elif use_previous_pose and i == len(traj_poses) - 2:
            step_role = "预采摘点"
        elif use_previous_pose:
            step_role = "预采摘路径点"
        else:
            step_role = "直线插补路径点"
        if use_previous_pose:
            step_name = XionganUtils.traj_step_name(i, len(traj_poses))
        else:
            step_name = f"插补点 {i + 1}/{len(traj_poses)}（{step_role}）"
        initial_joints_rad = last_joints_rad
        joint_rad, success = robot.Gn_ik(pose, initial_q=last_joints_rad)
        ik_info = {}

        if not success:
            XionganUtils.print_ik_failure(prefix, step_name, pose, initial_joints_rad, joint_rad, ik_info)
            ik_success = False
            ik_failed_step = i + 1
            break

        traj_joints.append(joint_rad.tolist())
        successful_pose_records.append({
            "pose": np.array(pose, dtype=float).copy(),
            "role": step_role,
            "step_name": step_name,
        })
        last_joints_rad = joint_rad

        print(f"{prefix} {step_name} 逆解成功")
        print(f"{prefix}   当前插补点pose[mm,rad]: {XionganUtils.format_pose(pose)}")
        print(f"{prefix}   逆解初值关节角[deg]: {XionganUtils.format_joints_deg(initial_joints_rad)}")
        print(f"{prefix}   逆解结果关节角[deg]: {XionganUtils.format_joints_deg(joint_rad)}")
        if ik_info:
            print(
                f"{prefix}   IK方法: {ik_info.get('method')}；"
                f"关节1-6最大突变量={ik_info.get('max_jump_deg')} deg；"
                f"关节1-6突变量={ik_info.get('limit_joint_jump_deg')} deg"
            )

    if not ik_success:
        return None, False, ik_failed_step

    print(f"{prefix} 直线插补逆解完成")
    print(f"{prefix} 目标点最终关节角[deg]: {XionganUtils.format_joints_deg(traj_joints[-1])}")
    robot.visualize_ik_pose_path_3d(successful_pose_records, target_pose, f"{prefix} 直线插补逆解pose三维可视化")
    return traj_joints, True, 0


def plan_traj_with_pose_retry(
    robot,
    locator,
    tomato_info,
    current_joints_rad,
    current_pose,
    max_angle,
    picking_offset,
    steps=3,
    log_context=None,
):
    prefix = XionganUtils.path_log_prefix(log_context)
    fallback_enabled = bool(Config.enable_fallback_picking_strategy)

    target_pose, previous_pose = locator.cal_picking_pose(
        tomato_info,
        max_angle=max_angle,
        picking_offset=picking_offset,
    )
    first_fallback_info = dict(tomato_info.get("vertical_x_fallback_info", {}))
    first_pose_is_fallback = bool(first_fallback_info.get("enabled", False))

    if first_pose_is_fallback:
        print(f"{prefix} TCP Z俯仰角已超过阈值，直接使用最低点备用pose进行逆解")
    elif not fallback_enabled:
        print(f"{prefix} 最低点备用采摘策略已关闭，使用正常pose进行逆解")
    else:
        print(f"{prefix} TCP Z俯仰角未超过阈值，优先使用正常pose进行逆解")

    traj_joints, ik_success, failed_step = cal_traj_for_fruit(
        robot,
        current_joints_rad,
        current_pose,
        target_pose,
        previous_pose,
        steps=steps,
        log_context=log_context,
    )

    plan_info = {
        "fallback_enabled": fallback_enabled,
        "normal_pose_ik_success": bool(ik_success) if not first_pose_is_fallback else None,
        "first_pose_was_fallback": first_pose_is_fallback,
        "fallback_attempted": False,
        "fallback_ik_success": None,
        "selected_pose_mode": "angle_fallback" if first_pose_is_fallback else "normal",
        "normal_failed_step": None if ik_success else failed_step,
        "fallback_failed_step": None,
    }

    if ik_success:
        return target_pose, previous_pose, traj_joints, True, failed_step, plan_info

    if first_pose_is_fallback:
        print(f"{prefix} 最低点备用pose逆解失败，判定当前果实采摘失败")
        plan_info["selected_pose_mode"] = "failed"
        return target_pose, previous_pose, traj_joints, False, failed_step, plan_info

    if not fallback_enabled:
        print(f"{prefix} 正常pose逆解失败，最低点备用采摘策略已关闭，判定当前果实采摘失败")
        plan_info["selected_pose_mode"] = "failed"
        return target_pose, previous_pose, traj_joints, False, failed_step, plan_info

    print(f"{prefix} 正常pose逆解失败，改用最低点备用pose重新逆解")
    fallback_target_pose, fallback_previous_pose = locator.cal_picking_pose(
        tomato_info,
        max_angle=max_angle,
        picking_offset=picking_offset,
        force_vertical_fallback=True,
    )

    fallback_traj_joints, fallback_ik_success, fallback_failed_step = cal_traj_for_fruit(
        robot,
        current_joints_rad,
        current_pose,
        fallback_target_pose,
        fallback_previous_pose,
        steps=steps,
        log_context=log_context,
    )

    plan_info.update({
        "fallback_attempted": True,
        "fallback_ik_success": bool(fallback_ik_success),
        "fallback_failed_step": None if fallback_ik_success else fallback_failed_step,
        "selected_pose_mode": "ik_failure_fallback" if fallback_ik_success else "failed",
    })

    if fallback_ik_success:
        print(f"{prefix} 最低点备用pose逆解成功，使用备用pose执行采摘")
        return fallback_target_pose, fallback_previous_pose, fallback_traj_joints, True, fallback_failed_step, plan_info

    print(f"{prefix} 正常pose和最低点备用pose均逆解失败，判定当前果实采摘失败")
    return fallback_target_pose, fallback_previous_pose, fallback_traj_joints, False, fallback_failed_step, plan_info


def pick_single_fruit(
    controller,
    traj_joints,
    ik_success,
    target_pose,
    previous_pose,
    speed,
    observed_joints_rad,
    go_original_before_basket=False,
    return_to_original_after_basket=True,
    post_basket_joints=None,
):
    post_basket_move_success = False
    controller.ee_open()

    if ik_success:
        traj_to_previous = traj_joints[:-1]
        for i, joint_angles in enumerate(traj_to_previous):
            controller.move_robot_joint(joint_angles, Config.normal_speed)
            print(f"执行插补: 已运动到路径点 {i + 1}/{len(traj_to_previous)}")
        if Config.use_previous_pose:
            print("已通过逆解轨迹到达预采摘点")
        else:
            print("已通过直线插补轨迹到达采摘点前一路径点")

        target_joint = traj_joints[-1]
        controller.move_robot_joint(target_joint, Config.slow_speed)
        print("已到达采摘点")
        reached_picking_point = True
    else:
        if not Config.use_move_p:
            print("逆解失败，不使用 move_p 备用方案，跳过当前果实")
            return

        if Config.use_previous_pose:
            previous_ret = controller.move_robot_pose(previous_pose, Config.slow_speed)
            if previous_ret != 0:
                print("逆解失败，而且机械臂自带 movej_p 到预采摘点也失败了")
                print("不执行采摘，不返回果篮")
                return

            print("逆解失败，已通过 movej_p 到达预采摘点")
        move_ret = controller.move_robot_pose(target_pose, Config.slow_speed)
        if move_ret != 0:
            if Config.use_previous_pose:
                print("逆解失败，而且机械臂自带 movej_p 从预采摘点到采摘点也失败了")
            else:
                print("逆解失败，而且机械臂自带 movej_p 直接到采摘点也失败了")
            print("不执行采摘，不返回果篮")
            return

        if Config.use_previous_pose:
            print("逆解失败，已通过 movej_p 从预采摘点到达采摘点")
        else:
            print("逆解失败，已通过 movej_p 直接到达采摘点")
        reached_picking_point = True

    if not reached_picking_point:
        print("没有到达采摘点，结束本次采摘")
        return

    print("已到达采摘点，末端进入果柄")
    time.sleep(1.5)
    controller.ee_close()
    time.sleep(1)

    if ik_success:
        pre_pick_joint = traj_joints[-2]
        ret_pre_pick = controller.move_robot_joint(pre_pick_joint, Config.slow_speed)
        if ret_pre_pick != 0:
            return
        if Config.use_previous_pose:
            print("已从采摘点退回到预采摘点")
        else:
            print("已从采摘点退回到前一路径点")

        back_path_joints = list(reversed(traj_joints[:-2]))
        for i, joint_angles in enumerate(back_path_joints):
            ret_back = controller.move_robot_joint(joint_angles, Config.normal_speed)
            if ret_back != 0:
                return
            print(f"退刀中：已退回第 {i + 1}/{len(back_path_joints)} 个插补点")

        ret_scan = controller.move_robot_joint(observed_joints_rad, Config.normal_speed)
        if ret_scan != 0:
            return
        print("已退回到当前扫描点")

    else:
        current_joints = controller.get_current_joint()
        # back_joint = ((np.array(current_joints) + np.array(observed_joints_rad)) / 2.0).tolist()  # 退一半
        back_joint = ((np.array(current_joints) + 2 * np.array(observed_joints_rad)) / 3.0).tolist()  # 退2/3
        controller.move_robot_joint(back_joint, Config.slow_speed)
        print("已从采摘点沿原轨迹安全退回2/3距离")

    time.sleep(0.3)

    if go_original_before_basket:
        original_joints_rad = np.deg2rad(Config.robot_original_joints).tolist()
        try:
            ret_original_before_basket = controller.move_robot_joint(original_joints_rad, Config.normal_speed)
            if ret_original_before_basket != 0:
                print(f"右侧扫描点去果篮前回初始位失败，错误码: {ret_original_before_basket}，继续去果篮")
        except Exception as e:
            print(f"右侧扫描点去果篮前回初始位报错，继续去果篮: {e}")
        time.sleep(0.3)

    try:
        ret_basket = controller.move_to_basket(Config.normal_speed)
        if ret_basket != 0:
            print(f"去果篮失败，错误码: {ret_basket}，继续执行后续流程")
    except Exception as e:
        print(f"去果篮报错，继续执行后续流程: {e}")

    # time.sleep(0.5)
    # controller.ee_close()
    time.sleep(0.5)
    controller.ee_open()
    time.sleep(0.5)

    if return_to_original_after_basket:
        original_joints_rad = np.deg2rad(Config.robot_original_joints).tolist()
        try:
            ret_home = controller.move_robot_joint(original_joints_rad, Config.fast_speed)
            if ret_home != 0:
                print(f"回初始位失败，错误码: {ret_home}，继续执行后续流程")
        except Exception as e:
            print(f"回初始位报错，继续执行后续流程: {e}")

    if post_basket_joints is not None:
        try:
            ret_scan = controller.move_robot_joint(post_basket_joints, Config.fast_speed)
            if ret_scan != 0:
                print(f"果篮后直达扫描点失败，错误码: {ret_scan}，继续执行后续流程")
            else:
                post_basket_move_success = True
                print("已从果篮到达后续扫描点")
        except Exception as e:
            print(f"果篮后直达扫描点报错，继续执行后续流程: {e}")

    return post_basket_move_success


def should_return_to_original_after_basket(index_scan, fruit_index, fruit_count):
    """非末目标仅第3/4点回原点；末目标第2/3点回原点，第4点末目标直达下一点。"""
    is_last_fruit = fruit_index == fruit_count - 1
    if is_last_fruit:
        return index_scan in (1, 2)
    return index_scan in (2, 3)


def should_return_to_current_scan_after_pick(index_scan, fruit_index, fruit_count):
    """非最后一颗放篮后返回当前扫描点。"""
    return fruit_index < fruit_count - 1


def get_post_basket_scan_index(index_scan, fruit_index, fruit_count, total_scans):
    """返回放篮后要去的扫描点索引；最后一个扫描点结束时返回None。"""
    if should_return_to_current_scan_after_pick(index_scan, fruit_index, fruit_count):
        return index_scan
    next_scan_index = index_scan + 1
    return next_scan_index if next_scan_index < total_scans else None


def start_realsense_camera(serial, camera_name, enable_depth):
    camera_pipeline = rs.pipeline()
    cfg = rs.config()
    if serial:
        cfg.enable_device(serial)
    if enable_depth:
        cfg.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    profile = camera_pipeline.start(cfg)
    if enable_depth:
        depth_sensor = profile.get_device().first_depth_sensor()
        Config.depth_scale = float(depth_sensor.get_depth_scale())
        print(f"{camera_name} 深度单位={Config.depth_scale:.9f} m/unit")
    align_to_color = rs.align(rs.stream.color) if enable_depth else None
    print(f"{camera_name} 初始化完成，serial={serial or 'auto'}")
    return camera_pipeline, align_to_color


def restart_realsense_camera(
    camera_pipeline,
    serial,
    camera_name,
    enable_depth=True,
    warmup_frames=10,
):
    """停止旧管线，重建RealSense管线和对齐对象，并丢弃预热帧。"""
    print("RealSense 连续采集失败，正在停止并重启相机管线")
    if camera_pipeline is not None:
        try:
            camera_pipeline.stop()
        except Exception as e:
            print(f"停止旧 RealSense 管线时出现警告：{e}")

    time.sleep(1.0)
    new_pipeline, new_align_to_color = start_realsense_camera(
        serial,
        camera_name,
        enable_depth=enable_depth,
    )

    print(f"RealSense 管线已重启，正在丢弃前 {warmup_frames} 帧进行预热")
    try:
        for _ in range(max(0, int(warmup_frames))):
            wait_aligned_images(new_pipeline, new_align_to_color)
    except Exception:
        try:
            new_pipeline.stop()
        except Exception:
            pass
        raise

    print("RealSense 管线预热完成")
    return new_pipeline, new_align_to_color


def wait_aligned_images(camera_pipeline, align_to_color, max_retries=5):
    """获取并对齐一组彩色/深度图像；单帧失败后最多重试 max_retries 次。"""
    retry_count = max(0, int(max_retries))
    total_attempts = retry_count + 1
    last_error = None

    for attempt in range(1, total_attempts + 1):
        try:
            camera_frames = camera_pipeline.wait_for_frames()
            aligned_frames = align_to_color.process(camera_frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not color_frame or not depth_frame:
                raise RuntimeError("RealSense 对齐结果缺少彩色帧或深度帧")

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            return color_image, depth_image
        except Exception as e:
            last_error = e
            if attempt <= retry_count:
                print(
                    f"RealSense 图像采集/对齐失败：{e}，"
                    f"准备进行第 {attempt}/{retry_count} 次重试"
                )
            else:
                print(f"RealSense 图像采集/对齐连续失败 {total_attempts} 次，停止重试")

    raise RealSenseFrameError(
        f"RealSense 图像采集/对齐连续失败 {total_attempts} 次：{last_error}"
    ) from last_error


def remember_depth_frame(depth_frames, depth_image, max_count):
    depth_frames.append(depth_image)
    while len(depth_frames) > max_count:
        depth_frames.pop(0)
    return depth_frames


def collect_depth_frames(camera_pipeline, align_to_color, count):
    depth_frames = []
    color_image = None
    for _ in range(max(1, int(count))):
        color_image, depth_image = wait_aligned_images(camera_pipeline, align_to_color)
        depth_frames.append(depth_image)
    return color_image, depth_frames


def show_scan_result_window(window_name, image, index_scan, total_scan, message, hold_ms=1):
    display = image.copy()

    cv2.putText(
        display,
        f"Scan {index_scan + 1}/{total_scan}: {message}",
        (30, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2
    )

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, display)

    try:
        window_handle = ctypes.windll.user32.FindWindowW(None, window_name)
        if window_handle:
            ctypes.windll.user32.ShowWindow(window_handle, 3)
    except Exception:
        pass

    # OpenCV窗口必须靠waitKey刷新，不能只用time.sleep
    cv2.waitKey(max(1, hold_ms))

def process_scan_point(
    controller,
    robot,
    locator,
    camera_pipeline,
    align_to_color,
    current_joints_deg,
    index_scan,
    total_scans,
    scan_window_name,
    already_at_scan_point=False,
):
    """处理一个扫描点；RealSense连续失败时向上层抛出恢复信号。"""
    # 1. 机械臂先运动到当前扫描点
    current_joints_rad = np.deg2rad(current_joints_deg).tolist()
    if already_at_scan_point:
        print(f"已提前到达第 {index_scan + 1} 个扫描点，直接开始稳定和拍照")
    else:
        controller.move_robot_joint(current_joints_rad, speed=Config.fast_speed)

    # 2. 到达扫描点后先稳定1秒，再拍照识别
    time.sleep(1)

    # 3. 丢掉几帧旧图，尽量使用机械臂稳定后的新图
    depth_frames = []
    depth_frame_count = max(1, int(Config.depth_multiframe_count))
    for _ in range(max(10, depth_frame_count)):
        color_image, depth_image = wait_aligned_images(camera_pipeline, align_to_color)
        remember_depth_frame(depth_frames, depth_image, depth_frame_count)

    observed_pose = controller.get_current_pose()
    observed_joints_radian = controller.get_current_joint()

    # 4. 执行检测
    tomatos_instance = locator.run_detection(color_image)

    if not tomatos_instance:
        print("未发现可采摘果实")
        show_scan_result_window(
            scan_window_name,
            color_image,
            index_scan,
            total_scans,
            "No fruit detected",
            hold_ms=1000,
        )
        cv2.destroyWindow(scan_window_name)
        return [], observed_pose, observed_joints_radian

    # 5. 执行三维定位
    tomatos_info = locator.run_localization(tomatos_instance, depth_frames, observed_pose)
    if not tomatos_info:
        for retry_index in range(max(0, int(Config.depth_retry_frame_sets))):
            print(f"深度质量不稳定，正在重采第 {retry_index + 1} 组深度帧")
            _, retry_depth_frames = collect_depth_frames(camera_pipeline, align_to_color, depth_frame_count)
            tomatos_info = locator.run_localization(tomatos_instance, retry_depth_frames, observed_pose)
            if tomatos_info:
                print("重采深度帧后定位成功")
                break

    if not tomatos_info:
        print("检测到目标但定位失败，显示检测画面1秒后换下一个扫描点")

        # 定位失败时至少显示实例分割/检测标识
        vis_image = locator.show_instances(color_image)

        show_scan_result_window(
            scan_window_name,
            vis_image,
            index_scan,
            total_scans,
            "Detected, but localization failed",
            hold_ms=1000,
        )
        cv2.destroyWindow(scan_window_name)
        return [], observed_pose, observed_joints_radian

    print(f"共发现并定位了 {len(tomatos_info)} 个果实实例")

    # 6. 生成识别标识画面：实例分割 + 拟合线 + P1/P2
    try:
        vis_image = locator.show_all_infos_on_images(color_image)
    except Exception as e:
        print(f"识别画面绘制失败，仅显示实例分割结果：{e}")
        vis_image = locator.show_instances(color_image)

    # 7. 显示识别结果，并停留1秒后再采摘
    show_scan_result_window(
        scan_window_name,
        vis_image,
        index_scan,
        total_scans,
        f"Detected {len(tomatos_info)} fruits",
        hold_ms=1000,
    )
    return tomatos_info, observed_pose, observed_joints_radian


def scan_and_pick_once(controller, robot, locator, camera_pipeline, align_to_color):
    scan_joints_list = Config.robot_grid_joint
    scan_window_name = "Arm Scan Detection"

    # 进入机械臂扫描阶段时，先关掉导航调试窗口，避免一直只看到导航画面
    try:
        cv2.destroyWindow("Line Following")
    except Exception:
        pass

    index_scan = 0
    prepositioned_scan_index = None
    while index_scan < len(scan_joints_list):
        current_joints_deg = scan_joints_list[index_scan]
        restart_count = 0
        tomatos_info = []
        observed_pose = None
        observed_joints_radian = None

        while True:
            print(f"\n正在前往第 {index_scan + 1}/{len(scan_joints_list)} 个扫描点")
            try:
                tomatos_info, observed_pose, observed_joints_radian = process_scan_point(
                    controller=controller,
                    robot=robot,
                    locator=locator,
                    camera_pipeline=camera_pipeline,
                    align_to_color=align_to_color,
                    current_joints_deg=current_joints_deg,
                    index_scan=index_scan,
                    total_scans=len(scan_joints_list),
                    scan_window_name=scan_window_name,
                    already_at_scan_point=(prepositioned_scan_index == index_scan),
                )
                prepositioned_scan_index = None
                break
            except RealSenseFrameError as e:
                if restart_count >= 1:
                    print(f"当前扫描点重启 RealSense 后仍失败，跳过该扫描点：{e}")
                    break

                restart_count += 1
                try:
                    camera_pipeline, align_to_color = restart_realsense_camera(
                        camera_pipeline,
                        Config.arm_camera_serial,
                        "机械臂扫描相机",
                        enable_depth=True,
                        warmup_frames=10,
                    )
                except Exception as restart_error:
                    raise RuntimeError(
                        f"RealSense 管线重启失败，无法继续扫描：{restart_error}"
                    ) from restart_error

                print(f"RealSense 已恢复，重新执行第 {index_scan + 1} 个扫描点")
            except Exception as e:
                print(f"当前扫描点处理失败，跳过该扫描点：{e}")
                try:
                    cv2.destroyWindow(scan_window_name)
                except Exception:
                    pass
                break

        # 8. 显示识别标识1秒后，再开始采摘
        for i, info in enumerate(tomatos_info):
            try:
                print(locator.format_depth_pose_marker(info, index_scan, i))
                print(f"规划路径并采摘果实 {i + 1}")

                target_pose, previous_pose, traj_joints, ik_success, _, plan_info = plan_traj_with_pose_retry(
                    robot,
                    locator,
                    info,
                    current_joints_rad=observed_joints_radian,
                    current_pose=observed_pose,
                    max_angle=Config.max_angle,
                    picking_offset=Config.picking_offset,
                    steps=Config.interpolation_steps,
                    log_context=(index_scan, len(scan_joints_list), i),
                )

                if not ik_success:
                    if plan_info["fallback_enabled"]:
                        print(f"当前果实可用pose均逆解失败，跳过该果实：{plan_info}")
                    else:
                        print(f"当前果实正常pose逆解失败且备用策略已关闭，跳过该果实：{plan_info}")
                    continue

                fruit_count = len(tomatos_info)
                return_to_original_after_basket = should_return_to_original_after_basket(
                    index_scan,
                    i,
                    fruit_count,
                )
                post_basket_scan_index = get_post_basket_scan_index(
                    index_scan,
                    i,
                    fruit_count,
                    len(scan_joints_list),
                )
                post_basket_joints = None
                if post_basket_scan_index == index_scan:
                    post_basket_joints = observed_joints_radian
                elif post_basket_scan_index is not None:
                    post_basket_joints = np.deg2rad(
                        scan_joints_list[post_basket_scan_index]
                    ).tolist()

                post_basket_move_success = pick_single_fruit(
                    controller=controller,
                    traj_joints=traj_joints,
                    ik_success=ik_success,
                    target_pose=target_pose,
                    previous_pose=previous_pose,
                    speed=Config.fast_speed,
                    observed_joints_rad=observed_joints_radian,
                    go_original_before_basket=(index_scan in (2, 3)),
                    return_to_original_after_basket=return_to_original_after_basket,
                    post_basket_joints=post_basket_joints,
                )

                if (
                    post_basket_move_success
                    and post_basket_scan_index == index_scan + 1
                ):
                    prepositioned_scan_index = post_basket_scan_index

            except Exception as e:
                print(f"当前果实采摘失败，跳过该果实：{e}")
                continue

        # 9. 当前扫描点结束，关闭识别画面，再切换到下一个扫描点
        try:
            cv2.destroyWindow(scan_window_name)
        except Exception:
            pass
        index_scan += 1

    print("\n所有扫描点已遍历完成，机械臂回到初始位置")
    controller.move_to_original(speed=Config.fast_speed)
    return camera_pipeline, align_to_color


if __name__ == '__main__':
    nav_camera_cap = None
    arm_camera_pipeline = None

    # 打开导航 USB 相机
    nav_camera_cap = XionganUtils.open_usb_camera(
        Config.nav_camera_name,
        Config.nav_camera_width,
        Config.nav_camera_height,
        Config.nav_camera_fps,
        Config.nav_camera_fourcc,
    )

    # 启动机械臂扫描 RealSense 相机
    arm_camera_pipeline, arm_align_to_color = start_realsense_camera(
        Config.arm_camera_serial,
        "机械臂扫描相机",
        enable_depth=True
    )

    controller = RobotController(Config, speed=30)
    robot = RobotKinematics()
    locator = Locator(Config.yolo_model_path, Config.cam_params['fx'], Config.cam_params['fy'],
                      Config.cam_params['cx'], Config.cam_params['cy'],
                      Config.cam2end_R, Config.cam2end_T, robot, Config.distortion)
    navigation = Navigation(Config)

    print("\n设备就绪，开始雄安走停走自动采摘流程")
    print(f"默认每次沿黑色胶带前进 {Config.nav_move_distance_m:.2f} 米")
    print(f"导航中识别不到黑线会直行补偿 {Config.nav_line_lost_compensation_m:.2f} 米后进入扫描采摘")
    print("扫描采摘完成后如果仍识别不到黑线，底盘会停止并结束流程")
    print("运行中按 q 退出，按空格紧急停止底盘")

    controller.move_to_original(speed=Config.fast_speed)

    def get_navigation_frame():
        return XionganUtils.read_usb_frame(nav_camera_cap)

    try:
        while True:
            nav_ok = navigation.move_forward_distance(
                distance_m=Config.nav_move_distance_m,
                get_frame=get_navigation_frame
            )
            if not nav_ok:
                print("导航停止，程序准备退出")
                break

            print("\n底盘已停止，开始遍历扫描点识别采摘")
            arm_camera_pipeline, arm_align_to_color = scan_and_pick_once(
                controller,
                robot,
                locator,
                arm_camera_pipeline,
                arm_align_to_color,
            )

            post_scan_frame = get_navigation_frame()
            if not navigation.is_black_line_visible(post_scan_frame):
                navigation.stop()
                print("扫描采摘完成后未识别到黑线，底盘停止，程序结束")
                break

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("按下 q，退出程序")
                break

    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，程序退出")
    finally:
        navigation.close()
        controller.disconnect_all_devices()
        if nav_camera_cap is not None:
            nav_camera_cap.release()
        if arm_camera_pipeline is not None:
            arm_camera_pipeline.stop()
        cv2.destroyAllWindows()

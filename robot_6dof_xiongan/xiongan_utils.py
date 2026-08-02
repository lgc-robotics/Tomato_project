import cv2
import numpy as np
from pygrabber.dshow_graph import DeviceCategories, FilterGraph, GUID, get_moniker_name


def list_usb_cameras_safely():
    graph = FilterGraph()
    filter_enumerator = graph.system_device_enum.system_device_enum.CreateClassEnumerator(
        GUID(DeviceCategories.VideoInputDevice),
        dwFlags=0,
    )
    devices = []
    try:
        moniker, count = filter_enumerator.Next(1)
    except ValueError:
        return devices

    index = 0
    while count > 0:
        try:
            name = get_moniker_name(moniker)
            devices.append((index, name))
        except Exception:
            pass
        index += 1
        moniker, count = filter_enumerator.Next(1)

    return devices


def find_usb_camera_index(camera_name):
    devices = list_usb_cameras_safely()
    for index, name in devices:
        if name == camera_name:
            print(f"已找到USB相机: index={index}, name={name}")
            return index
    raise RuntimeError(f"找不到USB相机 {camera_name}，当前设备: {devices}")


def open_usb_camera(camera_name, width, height, fps, fourcc):
    index = find_usb_camera_index(camera_name)
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    if not cap.isOpened():
        raise RuntimeError(f"无法打开USB相机 {camera_name}")

    for _ in range(10):
        cap.read()
    return cap


def read_usb_frame(cap):
    while True:
        ok, frame = cap.read()
        if ok and frame is not None:
            return frame


def format_pose(pose):
    pose = np.array(pose, dtype=float)
    return [
        round(float(pose[0]), 3),
        round(float(pose[1]), 3),
        round(float(pose[2]), 3),
        round(float(pose[3]), 4),
        round(float(pose[4]), 4),
        round(float(pose[5]), 4),
    ]


def format_joints_deg(joints_rad):
    joints_deg = np.rad2deg(np.array(joints_rad, dtype=float))
    return [round(float(v), 3) for v in joints_deg]


def fruit_log_prefix(scan_index, total_scans, fruit_index):
    return f"[扫描点 {scan_index + 1}/{total_scans} | 果实 {fruit_index + 1}]"


def path_log_prefix(log_context=None):
    if log_context is None:
        return "[路径规划]"
    scan_index, total_scans, fruit_index = log_context
    return fruit_log_prefix(scan_index, total_scans, fruit_index)

def _traj_step_name(step_index, total_steps):
    if step_index == total_steps - 1:
        role = "采摘目标点"
    elif step_index == total_steps - 2:
        role = "预采摘点"
    else:
        role = "预采摘路径点"
    return f"插补点 {step_index + 1}/{total_steps}（{role}）"


def _print_ik_failure(prefix, step_name, pose, initial_joints_rad, returned_joints_rad, ik_info):
    reason = ik_info.get("failure_reason", "unknown")
    print(f"{prefix} {step_name} 逆解失败")
    print(f"{prefix}   当前插补点pose[mm,rad]: {format_pose(pose)}")
    print(f"{prefix}   逆解初值关节角[deg]: {format_joints_deg(initial_joints_rad)}")

    if reason == "jump_limit_exceeded":
        print(f"{prefix}   逆解成功但超限的关节角[deg]: {format_joints_deg(returned_joints_rad)}")
        print(
            f"{prefix}   失败原因: 逆解成功，但相邻插补点关节1/4/6最大突变量超过 "
            f"jump_limit_deg={ik_info.get('jump_limit_deg')}"
        )
        print(f"{prefix}   相对上一个插补点关节1/4/6最大突变量[deg]: {ik_info.get('max_jump_deg')}")
        if ik_info.get("limit_joint_jump_deg") is not None:
            print(f"{prefix}   关节1/4/6突变量[deg]: {ik_info.get('limit_joint_jump_deg')}")
        if ik_info.get("joint_jump_deg") is not None:
            print(f"{prefix}   各关节突变量[deg]: {ik_info.get('joint_jump_deg')}")
    elif reason == "no_solution":
        print(f"{prefix}   逆解返回关节角[deg]: {format_joints_deg(returned_joints_rad)}")
        print(f"{prefix}   失败原因: 所有逆解都没有解出可用成功解")
    else:
        print(f"{prefix}   逆解返回关节角[deg]: {format_joints_deg(returned_joints_rad)}")
        print(f"{prefix}   失败原因: {reason}")


class XionganUtils:
    open_usb_camera = staticmethod(open_usb_camera)
    read_usb_frame = staticmethod(read_usb_frame)
    format_pose = staticmethod(format_pose)
    format_joints_deg = staticmethod(format_joints_deg)
    path_log_prefix = staticmethod(path_log_prefix)
    traj_step_name = staticmethod(_traj_step_name)
    print_ik_failure = staticmethod(_print_ik_failure)

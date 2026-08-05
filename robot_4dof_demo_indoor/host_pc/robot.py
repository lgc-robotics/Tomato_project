"""STM32 机械臂控制器串口协议辅助函数。"""

import math
import struct
import time

from config1 import END_EFFECTOR_ROTATE_LIMIT_DEG, X_MAX, X_MIN, Y_MAX, Y_MIN

MOVE_ERROR_ACK = 0xE4


def describe_move_status(status):
    """解析 STM32 返回的移动失败状态字节。"""
    if status is None:
        return "未收到状态字节"

    reached = []
    missing = []

    for bit, name in ((0, "X"), (1, "Y"), (2, "Z")):
        if status & (1 << bit):
            reached.append(name)
        else:
            missing.append(name)

    reached_text = ",".join(reached) if reached else "无"
    missing_text = ",".join(missing) if missing else "无"
    can_tx_error = "是" if (status & 0x40) else "否"
    watchdog = "是" if (status & 0x80) else "否"

    return (
        f"状态=0x{status:02X}, "
        f"已到位={reached_text}, "
        f"未到位={missing_text}, "
        f"CAN发送失败={can_tx_error}, "
        f"看门狗超时={watchdog}"
    )


def has_negative_coordinate(x, y, z):
    """判断坐标中是否存在负数。"""
    return x < 0 or y < 0 or z < 0


def is_robot_position_in_safe_range(x, y, z):
    """判断目标点是否避开配置的 X/Y 死区。"""
    return X_MIN <= x <= X_MAX and Y_MIN <= y <= Y_MAX and z >= 0


def clamp_robot_position_for_command(x, y, z):
    """发送给 STM32 前，对允许的越界指令做限幅。"""
    if x > X_MAX:
        print(
            "\n机械臂 X 指令超过最大范围，已限幅: "
            f"x={x:.2f} -> {X_MAX:.2f} 厘米"
        )
        x = X_MAX

    return x, y, z


def print_robot_position_limit_error(x, y, z):
    print(
        "\n机械臂坐标无效，已被运动死区限制拦截: "
        f"x={x:.2f}, y={y:.2f}, z={z:.2f}"
    )
    print(
        "允许范围: "
        f"X=[{X_MIN:.2f}, {X_MAX:.2f}] 厘米, "
        f"Y=[{Y_MIN:.2f}, {Y_MAX:.2f}] 厘米, "
        "Z>=0 厘米"
    )


def is_rotate_angle_safe(angle):
    """发送前检查末端角度，禁止上刀面越过安全半周。"""
    try:
        angle = float(angle)
    except (TypeError, ValueError):
        return False

    return (
        math.isfinite(angle)
        and -END_EFFECTOR_ROTATE_LIMIT_DEG
        <= angle
        <= END_EFFECTOR_ROTATE_LIMIT_DEG
    )


def print_rotate_angle_limit_error(angle):
    print(
        "\n末端旋转角被安全限制拦截，未发送05指令: "
        f"angle={angle!r} 度"
    )
    print(
        "允许范围: "
        f"[-{END_EFFECTOR_ROTATE_LIMIT_DEG:.1f}, "
        f"+{END_EFFECTOR_ROTATE_LIMIT_DEG:.1f}] 度"
    )


def wait_for_ack(ser, ack_byte, label, timeout_seconds=25.0):
    """等待一个确认信号字节，并打印意外收到的字节方便调试。"""
    deadline = time.time() + timeout_seconds
    received = bytearray()

    while time.time() < deadline:
        if ser.in_waiting:
            recv = ser.read()
            received.extend(recv)

            if recv == bytes([ack_byte]):
                print(f"{label}确认信号已收到: 0x{ack_byte:02X}")
                return True

            if recv == bytes([MOVE_ERROR_ACK]):
                status = None
                status_deadline = time.time() + 0.2

                while time.time() < status_deadline:
                    if ser.in_waiting:
                        status = ser.read()[0]
                        received.append(status)
                        break
                    time.sleep(0.001)

                print(f"{label} 收到移动异常确认信号: 0x{MOVE_ERROR_ACK:02X}")
                print("移动状态:", describe_move_status(status))
                return False

        else:
            time.sleep(0.001)

    print(
        f"{label}确认信号等待超时 {timeout_seconds:.1f} 秒，"
        f"期望 0x{ack_byte:02X}"
    )

    if received:
        print("等待期间收到的字节:", received.hex(" "))
    else:
        print("等待期间未收到任何字节")

    return False


def wait_for_acks(ser, expected_acks, timeout_seconds=25.0):
    """同时等待多个确认字节，允许确认信号以任意顺序到达。"""
    pending = dict(expected_acks)
    received = bytearray()
    deadline = time.time() + timeout_seconds

    while time.time() < deadline and pending:
        if not ser.in_waiting:
            time.sleep(0.001)
            continue

        value = ser.read()[0]
        received.append(value)

        if value == MOVE_ERROR_ACK:
            status = None
            status_deadline = time.time() + 0.2
            while time.time() < status_deadline:
                if ser.in_waiting:
                    status = ser.read()[0]
                    received.append(status)
                    break
                time.sleep(0.001)

            print(f"联合动作收到移动异常确认信号: 0x{MOVE_ERROR_ACK:02X}")
            print("移动状态:", describe_move_status(status))
            return False

        if value in pending:
            label = pending.pop(value)
            print(f"{label}确认信号已收到: 0x{value:02X}")

    if not pending:
        return True

    pending_text = ", ".join(f"0x{value:02X}" for value in pending)
    print(
        f"联合动作确认信号等待超时 {timeout_seconds:.1f} 秒，"
        f"未收到: {pending_text}"
    )
    if received:
        print("等待期间收到的字节:", received.hex(" "))
    else:
        print("等待期间未收到任何字节")
    return False


def send_robot_position(ser, x, y, z, ack_timeout_seconds=25.0):
    """发送 FF 04 坐标 FE，输入单位为厘米。"""
    if has_negative_coordinate(x, y, z):
        print(f"\n机械臂坐标无效: x={x:.2f}, y={y:.2f}, z={z:.2f}")
        return False

    x, y, z = clamp_robot_position_for_command(x, y, z)

    if not is_robot_position_in_safe_range(x, y, z):
        print_robot_position_limit_error(x, y, z)
        return False

    x_int = int(x * 1000)
    y_int = int(y * 1000)
    z_int = int(z * 1000)

    print("\n发送机械臂坐标:")
    print(f"输入(厘米): x={x:.2f}, y={y:.2f}, z={z:.2f}")
    print(f"发送整数: {x_int}, {y_int}, {z_int}")

    data = struct.pack("<iii", x_int, y_int, z_int)

    frame = bytearray()
    frame.append(0xFF)
    frame.append(0x04)
    frame.extend(data)
    frame.append(0xFE)

    ser.reset_input_buffer()
    ser.write(frame)

    print("\n坐标帧已发送:")
    print(frame.hex(" "))

    print("\n等待机械臂到位确认信号: 0x04")
    return wait_for_ack(ser, 0x04, "机械臂到位", ack_timeout_seconds)


def send_rotate_angle(ser, angle, ack_timeout_seconds=25.0):
    """发送 FF 05 角度 FE，角度单位为度。"""
    if not is_rotate_angle_safe(angle):
        print_rotate_angle_limit_error(angle)
        return False

    angle = float(angle)
    angle_int = int(angle * 100)

    print("\n发送末端旋转角:")
    print(f"输入(度): {angle}")
    print(f"发送整数: {angle_int}")

    data = struct.pack("<i", angle_int)
    data = data + b"\x00" * 8

    frame = bytearray()
    frame.append(0xFF)
    frame.append(0x05)
    frame.extend(data)
    frame.append(0xFE)

    ser.reset_input_buffer()
    ser.write(frame)

    print("\n旋转帧已发送:")
    print(frame.hex(" "))

    print("\n等待旋转确认信号: 0x05")
    return wait_for_ack(ser, 0x05, "末端旋转", ack_timeout_seconds)


def send_rotate_and_robot_position(
    ser,
    angle,
    x,
    y,
    z,
    send_interval_seconds=0.05,
    ack_timeout_seconds=25.0,
):
    """先发05启动末端旋转，短暂间隔后发04启动机械臂，再统一等待两个确认。"""
    if not is_rotate_angle_safe(angle):
        print_rotate_angle_limit_error(angle)
        return False

    angle = float(angle)

    if has_negative_coordinate(x, y, z):
        print(f"\n机械臂坐标无效: x={x:.2f}, y={y:.2f}, z={z:.2f}")
        return False

    x, y, z = clamp_robot_position_for_command(x, y, z)
    if not is_robot_position_in_safe_range(x, y, z):
        print_robot_position_limit_error(x, y, z)
        return False

    angle_int = int(angle * 100)
    rotate_data = struct.pack("<i", angle_int) + b"\x00" * 8
    rotate_frame = bytearray((0xFF, 0x05))
    rotate_frame.extend(rotate_data)
    rotate_frame.append(0xFE)

    x_int = int(x * 1000)
    y_int = int(y * 1000)
    z_int = int(z * 1000)
    position_data = struct.pack("<iii", x_int, y_int, z_int)
    position_frame = bytearray((0xFF, 0x04))
    position_frame.extend(position_data)
    position_frame.append(0xFE)

    interval = max(0.0, float(send_interval_seconds))
    ser.reset_input_buffer()

    print("\n联合动作：先启动末端旋转")
    print(f"旋转角度: {angle:.2f} 度，发送整数: {angle_int}")
    print(rotate_frame.hex(" "))
    ser.write(rotate_frame)

    time.sleep(interval)

    print(f"\n间隔 {interval * 1000:.0f} 毫秒后启动机械臂")
    print(f"目标坐标(厘米): x={x:.2f}, y={y:.2f}, z={z:.2f}")
    print(f"发送整数: {x_int}, {y_int}, {z_int}")
    print(position_frame.hex(" "))
    ser.write(position_frame)

    print("\n统一等待0x05旋转确认和0x04机械臂到位确认")
    return wait_for_acks(
        ser,
        {
            0x05: "末端旋转",
            0x04: "机械臂到位",
        },
        ack_timeout_seconds,
    )


def send_end_effector_action(ser, action, ack_timeout_seconds=25.0):
    """发送 FF 06 动作 FE，动作 0x01 表示张开，0x02 表示闭合。"""
    action = int(action)
    if action not in (0x01, 0x02):
        print(f"\n末端执行器动作无效: 0x{action:02X}")
        return False

    action_label = "张开" if action == 0x01 else "闭合"

    frame = bytearray()
    frame.append(0xFF)
    frame.append(0x06)
    frame.append(action & 0xFF)
    frame.extend(b"\x00" * 11)
    frame.append(0xFE)

    ser.reset_input_buffer()
    ser.write(frame)

    print(f"\n末端执行器动作已发送: {action_label} 0x{action:02X}")
    print(frame.hex(" "))
    print("\n等待末端执行器确认信号: 0x06")
    return wait_for_ack(ser, 0x06, f"末端执行器({action_label})", ack_timeout_seconds)

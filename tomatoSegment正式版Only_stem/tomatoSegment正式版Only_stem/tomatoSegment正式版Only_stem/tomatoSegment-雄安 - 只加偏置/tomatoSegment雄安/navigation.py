import time
from ctypes import Structure, byref, c_ubyte, c_uint, c_uint32, c_void_p, windll

import cv2
import numpy as np

from config import Config


CAN_ID_VELOCITY = 0x111
CAN_ID_LIGHT_CONTROL = 0x121
CAN_ID_ENABLE_CONTROL = 0x421
CAN_ID_CLEAR_ERROR = 0x441
MAX_LINEAR_MM_S = 1500
MAX_ANGULAR_MM_S = 523
DEBUG_WINDOW_NAME = "Line Following"
BOTTOM_TRACKING_BAND_RATIO = 0.18
LIGHT_MODE_VALUES = {
    "off": 0x00,
    "on": 0x01,
    "breath": 0x02,
    "custom": 0x03,
}


class _ControlCanInitConfig(Structure):
    """ControlCAN.dll required C memory layout for VCI_InitCAN."""

    _fields_ = [
        ("AccCode", c_uint32),
        ("AccMask", c_uint32),
        ("Reserved", c_uint32),
        ("Filter", c_ubyte),
        ("Timing0", c_ubyte),
        ("Timing1", c_ubyte),
        ("Mode", c_ubyte),
    ]


class _ControlCanFrame(Structure):
    """ControlCAN.dll required C memory layout for VCI_Transmit."""

    _fields_ = [
        ("ID", c_uint),
        ("TimeStamp", c_uint),
        ("TimeFlag", c_ubyte),
        ("SendType", c_ubyte),
        ("RemoteFlag", c_ubyte),
        ("ExternFlag", c_ubyte),
        ("DataLen", c_ubyte),
        ("Data", c_ubyte * 8),
        ("Reserved", c_ubyte * 3),
    ]


def _clamp_int(value, min_value, max_value):
    return max(min(int(value), max_value), min_value)


def _to_unsigned_int16(value):
    return (1 << 16) + value if value < 0 else value


class _LineLossCounter:

    def __init__(self, max_missing_frames):
        self.max_missing_frames = max(1, int(max_missing_frames))
        self.missing_frames = 0

    def update(self, line_detected):
        if line_detected:
            self.missing_frames = 0
            return False

        self.missing_frames += 1
        return self.missing_frames >= self.max_missing_frames


class Navigation:

    def __init__(self, config=Config):
        self.config = config
        self.can_dll = self._load_control_can_dll()
        self._set_can_dll_types()

        self.can_msg = _ControlCanFrame()
        self.can_msg.SendType = 1
        self.can_msg.RemoteFlag = 0
        self.can_msg.ExternFlag = 0

        self.pid_integral = 0.0
        self.pid_prev_error = 0.0
        self.light_command_count = 0
        self.last_light_send_time = 0.0

        self._init_chassis()
        print("导航初始化完成")

    def move_forward_distance(self, distance_m=None, get_frame=None):
        if distance_m is None:
            distance_m = self.config.nav_move_distance_m
        if get_frame is None:
            raise ValueError("请传入get_frame 函数，把相机画面交给导航使用")

        print(f"\n开始沿黑色胶带前进 {distance_m:.2f} 米")

        moved_distance = 0.0
        prev_time = time.time()
        line_loss_counter = _LineLossCounter(
            getattr(self.config, "nav_line_lost_frames", 8)
        )
        linear_vel = self.config.nav_speed
        angular_vel = getattr(self.config, "nav_angular_bias", 0.0)

        while moved_distance < distance_m:
            frame = get_frame()
            if frame is None:
                continue

            now = time.time()
            dt = now - prev_time
            prev_time = now

            line_offset, binary, line_area, line_contour = self._detect_black_line(frame)

            if line_offset is None:
                if line_loss_counter.update(line_detected=False):
                    compensation_m = getattr(self.config, "nav_line_lost_compensation_m", 0.5)
                    print(f"连续未识别到黑线，开始直行补偿{compensation_m:.2f} 米")
                    return self._compensate_after_line_lost(compensation_m, get_frame)
            else:
                line_loss_counter.update(line_detected=True)
                linear_vel, angular_vel = self._get_velocity(line_offset, dt)

            self._send_velocity(linear_vel, angular_vel)
            moved_distance += max(linear_vel, 0.0) * dt

            if self.config.nav_show_window:
                self._show_debug(frame, line_offset, binary, line_contour, moved_distance, distance_m,
                                 linear_vel, angular_vel)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("按下 q，退出导航")
                self.stop()
                return False
            if key == ord(" "):
                print("空格按下：紧急停止")
                self.stop()
                return False

        self.stop()
        print(f"本次导航完成，前进约 {moved_distance:.2f} 米")
        return True

    def is_black_line_visible(self, frame):
        if frame is None:
            return False
        line_offset, _, _, _ = self._detect_black_line(frame)
        return line_offset is not None

    def _compensate_after_line_lost(self, compensation_m, get_frame):
        moved_distance = 0.0
        linear_vel = self.config.nav_speed
        angular_vel = getattr(self.config, "nav_angular_bias", 0.0)
        loop_interval = getattr(self.config, "nav_control_loop_interval_s", 0.02)

        if compensation_m <= 0.0 or linear_vel <= 0.0:
            self.stop()
            print("丢线补偿距离或速度为0，底盘停止并进入扫描采摘")
            return True

        start_time = time.time()

        while moved_distance < compensation_m:
            now = time.time()
            moved_distance = max(linear_vel, 0.0) * (now - start_time)
            self._send_velocity(linear_vel, angular_vel)

            if self.config.nav_show_window:
                frame = get_frame()
                if frame is not None:
                    line_offset, binary, line_area, line_contour = self._detect_black_line(frame)
                    self._show_debug(frame, line_offset, binary, line_contour, moved_distance, compensation_m,
                                     linear_vel, angular_vel)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("按下 q，退出导航")
                self.stop()
                return False
            if key == ord(" "):
                print("空格按下：紧急停止")
                self.stop()
                return False

            time.sleep(loop_interval)

        self.stop()
        print(f"丢线后已直行补偿 {moved_distance:.2f} 米，停止并进入扫描采摘")
        return True

    def stop(self):
        self._send_velocity(0.0, 0.0)

    def close(self):
        self.stop()
        print("底盘已停止")

    def _load_control_can_dll(self):
        last_error = None
        for dll_path in self.config.nav_control_can_dll_paths:
            try:
                return windll.LoadLibrary(dll_path)
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"找不到ControlCAN.dll，请检查Config.nav_control_can_dll_paths: {last_error}")

    def _set_can_dll_types(self):
        self.can_dll.VCI_OpenDevice.argtypes = [c_uint32, c_uint32, c_uint32]
        self.can_dll.VCI_OpenDevice.restype = c_uint32
        self.can_dll.VCI_InitCAN.argtypes = [c_uint32, c_uint32, c_uint32, c_void_p]
        self.can_dll.VCI_InitCAN.restype = c_uint32
        self.can_dll.VCI_StartCAN.argtypes = [c_uint32, c_uint32, c_uint32]
        self.can_dll.VCI_StartCAN.restype = c_uint32
        self.can_dll.VCI_Transmit.argtypes = [c_uint32, c_uint32, c_uint32, c_void_p, c_uint32]
        self.can_dll.VCI_Transmit.restype = c_uint32

    def _init_chassis(self):
        print("初始化CAN...")
        if self.can_dll.VCI_OpenDevice(self.config.nav_dev_type, self.config.nav_dev_idx, 0) != 1:
            raise RuntimeError("打开 CAN 设备失败")

        init_config = _ControlCanInitConfig()
        init_config.AccCode = 0x00000000
        init_config.AccMask = 0xFFFFFFFF
        init_config.Filter = 1
        init_config.Timing0 = self.config.nav_can_timing0
        init_config.Timing1 = self.config.nav_can_timing1
        init_config.Mode = 0

        if self.can_dll.VCI_InitCAN(
            self.config.nav_dev_type,
            self.config.nav_dev_idx,
            self.config.nav_can_idx,
            byref(init_config),
        ) != 1:
            raise RuntimeError("CAN 初始化失败")

        if self.can_dll.VCI_StartCAN(
            self.config.nav_dev_type,
            self.config.nav_dev_idx,
            self.config.nav_can_idx,
        ) != 1:
            raise RuntimeError("CAN 启动失败")

        print("CAN 通信 OK")
        self._send_can_frame(CAN_ID_CLEAR_ERROR, [0x00])
        time.sleep(0.2)
        self._send_can_frame(CAN_ID_ENABLE_CONTROL, [0x01])
        time.sleep(0.2)
        self._send_light_control(force=True)

    def _send_velocity(self, linear_vel=0.0, angular_vel=0.0):
        v_mm = _to_unsigned_int16(_clamp_int(linear_vel * 1000.0, -MAX_LINEAR_MM_S, MAX_LINEAR_MM_S))
        w_mm = _to_unsigned_int16(_clamp_int(angular_vel * 1000.0, -MAX_ANGULAR_MM_S, MAX_ANGULAR_MM_S))
        self._send_can_frame(
            CAN_ID_VELOCITY,
            [
                (v_mm >> 8) & 0xFF,
                v_mm & 0xFF,
                (w_mm >> 8) & 0xFF,
                w_mm & 0xFF,
                0,
                0,
                0,
                0,
            ],
        )
        self._send_light_control()

    def _send_light_control(self, force=False):
        # nav_light_enable=False 时，程序完全不接管底盘灯光，保留遥控器或底盘原有灯光状态。
        if not getattr(self.config, "nav_light_enable", False):
            return

        now = time.time()
        send_period = max(float(getattr(self.config, "nav_light_send_period_s", 0.1)), 0.02)
        if not force and now - self.last_light_send_time < send_period:
            return

        # 松灵底盘 v2 协议灯光帧 0x121：
        # byte[0]=1 表示启用灯光指令控制；
        # byte[1] 是模式，off/on/breath/custom 分别对应 0/1/2/3；
        # byte[2] 只有 custom 模式生效，范围 0~100，当前推荐配置为 70。
        mode_name = str(getattr(self.config, "nav_light_mode", "custom")).lower()
        if mode_name not in LIGHT_MODE_VALUES:
            raise ValueError(
                f"Config.nav_light_mode={mode_name!r} 无效，必须是 off/on/breath/custom 之一"
            )

        mode_value = LIGHT_MODE_VALUES[mode_name]
        brightness = 0
        if mode_name == "custom":
            brightness = _clamp_int(getattr(self.config, "nav_light_brightness", 70), 0, 100)

        self._send_can_frame(
            CAN_ID_LIGHT_CONTROL,
            [
                0x01,
                mode_value,
                brightness,
                mode_value,
                brightness,
                0x00,
                0x00,
                self.light_command_count & 0xFF,
            ],
        )
        if force:
            print(
                "底盘灯光控制帧: "
                f"id=0x{CAN_ID_LIGHT_CONTROL:X}, "
                f"mode={mode_name}, "
                f"brightness={brightness}, "
                "front/rear 同步设置"
            )
        self.light_command_count = (self.light_command_count + 1) & 0xFF
        self.last_light_send_time = now

    def _send_can_frame(self, can_id, data):
        msg = _ControlCanFrame()
        msg.ID = can_id
        msg.SendType = 1
        msg.RemoteFlag = 0
        msg.ExternFlag = 0
        msg.DataLen = len(data)
        for index in range(len(data)):
            msg.Data[index] = data[index]

        self.can_dll.VCI_Transmit(
            self.config.nav_dev_type,
            self.config.nav_dev_idx,
            self.config.nav_can_idx,
            byref(msg),
            1,
        )

    def _detect_black_line(self, frame):
        h, w = frame.shape[:2]
        roi_y = int(h * (1.0 - self.config.nav_roi_bottom))
        roi = frame[roi_y:, :]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, self.config.nav_fixed_thresh, 255, cv2.THRESH_BINARY_INV)

        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        contour, area = self._find_nearest_line_contour(binary)
        if contour is None:
            return None, binary, 0.0, None

        moments = cv2.moments(contour)
        if moments["m00"] <= 0:
            return None, binary, area, None

        image_center_x = w / 2.0
        cx = moments["m10"] / moments["m00"]
        offset = (cx - image_center_x) / image_center_x
        return offset, binary, area, contour

    def _find_nearest_line_contour(self, binary):
        roi_height = binary.shape[0]
        band_y = int(roi_height * (1.0 - BOTTOM_TRACKING_BAND_RATIO))
        contour, area = self._largest_vertical_contour(binary[band_y:, :], self.config.nav_min_area * 0.3)
        if contour is not None:
            contour[:, :, 1] += band_y
            return contour, area

        return self._largest_vertical_contour(binary, self.config.nav_min_area)

    def _largest_vertical_contour(self, binary, min_area):
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            _, _, width, height = cv2.boundingRect(contour)
            if height < binary.shape[0] * 0.25 and width > height * 1.5:
                continue
            candidates.append((contour, area))

        if not candidates:
            return None, 0.0

        return max(candidates, key=lambda item: item[1])

    def _get_velocity(self, line_offset, dt):
        angular_vel = self._pid_update(line_offset, dt)
        if self.config.nav_invert_steering:
            angular_vel = -angular_vel
        angular_vel += getattr(self.config, "nav_angular_bias", 0.0)
        angular_vel = max(-self.config.nav_pid_max_output, min(self.config.nav_pid_max_output, angular_vel))
        return self.config.nav_speed, angular_vel

    def _pid_update(self, error, dt):
        dt = max(dt, 0.001)
        self.pid_integral += error * dt
        self.pid_integral = max(
            -self.config.nav_pid_max_integral,
            min(self.config.nav_pid_max_integral, self.pid_integral),
        )
        derivative = (error - self.pid_prev_error) / dt
        self.pid_prev_error = error

        raw = -(
            self.config.nav_pid_kp * error
            + self.config.nav_pid_ki * self.pid_integral
            + self.config.nav_pid_kd * derivative
        )
        return max(-self.config.nav_pid_max_output, min(self.config.nav_pid_max_output, raw))

    def _show_debug(self, frame, offset, binary, contour, moved_distance, target_distance, linear_vel, angular_vel):
        display = frame.copy()
        h, w = frame.shape[:2]
        roi_y = int(h * (1.0 - self.config.nav_roi_bottom))

        cv2.rectangle(display, (0, roi_y), (w, h), (0, 255, 0), 2)
        cv2.line(display, (w // 2, roi_y), (w // 2, h), (0, 255, 0), 1)

        if contour is not None:
            contour_display = contour.copy()
            contour_display[:, :, 1] += roi_y
            cv2.drawContours(display, [contour_display], -1, (0, 0, 255), 2)
            cx = int((offset + 1) * w / 2)
            cv2.circle(display, (cx, roi_y + 20), 8, (0, 0, 255), -1)
            cv2.line(display, (w // 2, roi_y + 20), (cx, roi_y + 20), (0, 0, 255), 2)
        else:
            cv2.putText(display, "NO LINE", (w // 2 - 50, roi_y + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        display[10:110, w - 160:w - 10] = cv2.resize(binary_bgr, (150, 100))
        cv2.rectangle(display, (w - 160, 10), (w - 10, 110), (255, 255, 255), 1)

        cv2.putText(display, f"Move: {moved_distance:.2f}/{target_distance:.2f}m", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(display, f"V={linear_vel:.2f} W={angular_vel:.3f}", (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow(DEBUG_WINDOW_NAME, display)

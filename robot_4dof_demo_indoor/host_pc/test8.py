"""Chassis line-following module.

This file can be used in two ways:  q
1. Imported by main1.py as LineFollowerChassis for whole-machine debugging.
2. Run directly with `python test8.py` for standalone chassis line-following.

Serial frames to STM32:
- control: FF 07 error(int16 LE) padding(10) FE
- debug feedback from STM32: FF 08 errorL errorH status gear IF 00 FE
- timed-move done ACK from STM32: 07
"""

import struct
import time
from collections import deque

import cv2
import numpy as np
import serial


class LineFollowerChassis:
    """Line-following chassis controller with STM32 debug feedback parsing."""

    def __init__(
        self,
        port="COM10",
        baudrate=115200,
        timeout=1,
        camera_index=3,
        frame_width=640,
        frame_height=480,
        send_interval=0.05,
        roi_top_ratio=0.4,
        search_error=400,
        history_len=15,
        min_record_error=20,
        show_debug=False,
        debug_display_interval_seconds=0.10,
        camera_warmup_frames=20,
        start_stable_frames=6,
        start_stable_spread_px=30,
        start_stable_timeout_seconds=3.0,
        error_filter_window=5,
        max_error_step_px=40,
        min_contour_area_px=80,
        max_contour_area_ratio=0.25,
        log_interval_seconds=0.25,
        ser=None,
        close_serial_on_close=True,
    ):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.camera_index = camera_index
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.send_interval = send_interval
        self.roi_top_ratio = roi_top_ratio
        self.search_error = search_error
        self.min_record_error = min_record_error
        self.show_debug = show_debug
        self.debug_display_interval_seconds = max(
            0.0, float(debug_display_interval_seconds)
        )
        self.camera_warmup_frames = max(0, int(camera_warmup_frames))
        self.start_stable_frames = max(1, int(start_stable_frames))
        self.start_stable_spread_px = max(0, int(start_stable_spread_px))
        self.start_stable_timeout_seconds = max(
            0.1, float(start_stable_timeout_seconds)
        )
        self.error_filter_window = max(1, int(error_filter_window))
        self.max_error_step_px = max(1, int(max_error_step_px))
        self.min_contour_area_px = max(0.0, float(min_contour_area_px))
        self.max_contour_area_ratio = max(
            0.01, min(1.0, float(max_contour_area_ratio))
        )
        self.log_interval_seconds = max(0.0, float(log_interval_seconds))
        self.close_serial_on_close = close_serial_on_close

        self.ser = ser
        self.cap = None
        self.last_send_time = 0
        self.error_history = deque(maxlen=history_len)
        self.error_filter = deque(maxlen=self.error_filter_window)
        self.rx_buffer = bytearray()
        self.last_control_error = 0
        self.last_control_log_time = 0.0
        self.last_debug_display_time = 0.0

        self.true_error = 0
        self.car_status = 0
        self.car_gear = 0
        self.can_if = 0
        self.timed_move_done = False

    def open(self):
        if self.ser is None:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
            )
            print(f"Chassis serial opened: {self.port}")
        else:
            print("Chassis using shared serial")

        if self.cap is None:
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not self.cap.isOpened():
                self.close()
                raise RuntimeError(f"Chassis camera open failed: {self.camera_index}")

            try:
                backend_name = self.cap.getBackendName()
            except cv2.error:
                backend_name = "unknown"

            print(
                "Chassis camera opened: "
                f"index={self.camera_index}, backend={backend_name}"
            )
            self.warm_up_camera()

    def warm_up_camera(self):
        """Discard initial frames after reopening so exposure can settle."""
        if self.cap is None or self.camera_warmup_frames <= 0:
            return

        valid_frames = 0
        for _ in range(self.camera_warmup_frames):
            ret, _ = self.cap.read()
            if ret:
                valid_frames += 1

        if valid_frames == 0:
            raise RuntimeError("Chassis camera warm-up failed: no valid frame")

        print(
            "Chassis camera warm-up complete: "
            f"{valid_frames}/{self.camera_warmup_frames} frames"
        )

    def close_camera(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        if self.show_debug:
            for window_name in ("Camera", "Mask"):
                try:
                    cv2.destroyWindow(window_name)
                except cv2.error:
                    pass

    def close(self):
        self.close_camera()

        if self.ser is not None and self.close_serial_on_close:
            self.ser.close()
            self.ser = None

    def read_debug_frame(self):
        """Parse STM32 debug frame and one-byte timed-move ACK."""
        if self.ser is None:
            return

        while self.ser.in_waiting:
            byte = self.ser.read(1)

            if len(byte) == 0:
                return

            self.rx_buffer.extend(byte)

            if len(self.rx_buffer) > 100:
                self.rx_buffer.clear()

            while len(self.rx_buffer) > 0:
                if self.rx_buffer[0] == 0x07:
                    self.timed_move_done = True
                    self.rx_buffer.pop(0)
                    print("Chassis timed move ACK received: 0x07")
                    continue

                if self.rx_buffer[0] != 0xFF:
                    self.rx_buffer.pop(0)
                    continue

                if len(self.rx_buffer) < 9:
                    return

                if self.rx_buffer[1] != 0x08:
                    self.rx_buffer.pop(0)
                    continue

                if self.rx_buffer[8] != 0xFE:
                    self.rx_buffer.pop(0)
                    continue

                self.true_error = struct.unpack("<h", self.rx_buffer[2:4])[0]
                self.car_status = self.rx_buffer[4]
                self.car_gear = self.rx_buffer[5]
                self.can_if = self.rx_buffer[6]

                del self.rx_buffer[:9]

    def build_control_frame(self, error):
        error = int(max(min(error, 32767), -32768))

        frame = bytearray()
        frame.append(0xFF)
        frame.append(0x07)
        frame.extend(struct.pack("<h", error))
        frame.extend([0x00] * 10)
        frame.append(0xFE)
        return frame

    def send_control(self, error):
        if self.ser is None:
            raise RuntimeError("Chassis serial is not opened")

        frame = self.build_control_frame(error)
        self.ser.write(frame)
        self.last_send_time = time.monotonic()

    def reset_control_filter(self):
        """Reset per-move filtering so the previous move cannot affect this one."""
        self.error_filter.clear()
        self.last_control_error = 0
        self.last_control_log_time = 0.0

    def read_line_error(self):
        """Return (line_found, error, frame, mask)."""
        if self.cap is None:
            raise RuntimeError("Chassis camera is not opened")

        ret, frame = self.cap.read()

        if not ret:
            return False, 0, None, None

        height, width, _ = frame.shape
        roi_end_y = int(height * self.roi_top_ratio)
        roi_end_y = max(1, min(height, roi_end_y))
        roi = frame[0:roi_end_y, :]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 60])
        mask = cv2.inRange(hsv, lower_black, upper_black)

        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            return False, 0, frame, mask

        roi_area = float(roi.shape[0] * roi.shape[1])
        max_contour_area = roi_area * self.max_contour_area_ratio
        valid_contours = [
            contour
            for contour in contours
            if self.min_contour_area_px
            <= cv2.contourArea(contour)
            <= max_contour_area
        ]

        if not valid_contours:
            return False, 0, frame, mask

        contour = max(valid_contours, key=cv2.contourArea)
        moments = cv2.moments(contour)

        if moments["m00"] == 0:
            return False, 0, frame, mask

        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        frame_center = width // 2
        error = cx - frame_center

        if abs(error) > self.min_record_error:
            self.error_history.append(error)

        if self.show_debug:
            cx_global = cx
            cy_global = cy
            cv2.circle(frame, (cx_global, cy_global), 6, (0, 0, 255), -1)
            cv2.line(frame, (frame_center, 0), (frame_center, height), (255, 0, 0), 2)
            cv2.rectangle(frame, (0, 0), (width - 1, roi_end_y - 1), (0, 255, 255), 2)

        return True, error, frame, mask

    def get_lost_line_search_error(self):
        if len(self.error_history) == 0:
            return 0

        score = 0

        for i, error in enumerate(self.error_history):
            score += (i + 1) * error

        if score > 0:
            return self.search_error

        if score < 0:
            return -self.search_error

        return 0

    def decide_control(self, line_found, error):
        if not line_found:
            self.error_filter.clear()
            self.last_control_error = 0
            return 0

        self.error_filter.append(int(error))
        filtered_error = int(round(float(np.median(self.error_filter))))
        lower = self.last_control_error - self.max_error_step_px
        upper = self.last_control_error + self.max_error_step_px
        send_error = max(lower, min(upper, filtered_error))
        self.last_control_error = int(send_error)
        return self.last_control_error

    def show_control_debug(self, frame, mask, error, line_found):
        if not self.show_debug or frame is None or mask is None:
            return False

        now = time.monotonic()
        if (
            now - self.last_debug_display_time
            < self.debug_display_interval_seconds
        ):
            return False

        self.last_debug_display_time = now

        self.draw_debug_overlay(frame, error, line_found)
        cv2.imshow("Camera", frame)
        cv2.imshow("Mask", mask)
        return cv2.waitKey(1) & 0xFF == ord("q")

    def wait_for_stable_line(self):
        """Wait for consecutive stable detections before sending the first command."""
        stable_errors = deque(maxlen=self.start_stable_frames)
        deadline = time.monotonic() + self.start_stable_timeout_seconds

        print(
            "Waiting for stable chassis line before movement: "
            f"{self.start_stable_frames} frames"
        )

        while time.monotonic() < deadline:
            self.read_debug_frame()
            line_found, error, frame, mask = self.read_line_error()

            if line_found:
                stable_errors.append(int(error))
            else:
                stable_errors.clear()

            if self.show_control_debug(frame, mask, error, line_found):
                return False

            if len(stable_errors) < self.start_stable_frames:
                continue

            spread = max(stable_errors) - min(stable_errors)
            if spread <= self.start_stable_spread_px:
                self.error_filter.extend(stable_errors)
                median_error = int(round(float(np.median(stable_errors))))
                print(
                    "Chassis line is stable: "
                    f"median_error={median_error}, spread={spread}px"
                )
                return True

            stable_errors.popleft()

        print("Chassis line did not stabilize; this move is cancelled")
        return False

    def log_control(self, raw_error, send_error, line_found):
        now = time.monotonic()
        if now - self.last_control_log_time < self.log_interval_seconds:
            return

        self.last_control_log_time = now
        print(
            f"Chassis control: raw={raw_error}, sent={send_error}, "
            f"line_found={line_found}"
        )

    def draw_debug_overlay(self, frame, error, line_found):
        if frame is None:
            return

        cv2.putText(
            frame,
            f"PY ERROR: {error}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"FOUND: {line_found}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"STM32 ERROR: {self.true_error}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"STATUS: {self.car_status}  GEAR: {self.car_gear}  CAN IF: {self.can_if}",
            (20, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 255),
            2,
        )

    def follow_for(self, seconds):
        self.open()
        self.reset_control_filter()

        if not self.wait_for_stable_line():
            return {
                "started": False,
                "line_found_count": 0,
                "line_lost_count": 0,
            }

        end_time = time.monotonic() + max(0, seconds)
        line_found_count = 0
        line_lost_count = 0

        while time.monotonic() < end_time:
            self.read_debug_frame()
            line_found, error, frame, mask = self.read_line_error()

            now = time.monotonic()

            if now - self.last_send_time >= self.send_interval:
                send_error = self.decide_control(line_found, error)
                self.send_control(send_error)

                if line_found:
                    line_found_count += 1
                else:
                    line_lost_count += 1

                self.log_control(error, send_error, line_found)

            if self.show_control_debug(frame, mask, error, line_found):
                break

        return {
            "started": True,
            "line_found_count": line_found_count,
            "line_lost_count": line_lost_count,
        }

    def run_timed_move(self, ack_timeout_seconds=5.0):
        """Send line-following errors until STM32 reports this timed move is done."""
        self.open()
        self.timed_move_done = False
        self.rx_buffer.clear()
        self.reset_control_filter()

        if not self.wait_for_stable_line():
            return {
                "started": False,
                "ack_received": False,
                "line_found_count": 0,
                "line_lost_count": 0,
            }

        # 丢弃相机稳定期间可能残留的上一段底盘运动确认信号。
        self.timed_move_done = False
        self.rx_buffer.clear()

        start_time = time.monotonic()
        line_found_count = 0
        line_lost_count = 0

        print(
            f"\nChassis timed move started, "
            f"waiting for ACK 0x07, timeout={ack_timeout_seconds:.1f}s"
        )

        while not self.timed_move_done:
            if time.monotonic() - start_time > ack_timeout_seconds:
                print("Chassis timed move ACK timeout")
                break

            self.read_debug_frame()
            line_found, error, frame, mask = self.read_line_error()

            now = time.monotonic()

            if now - self.last_send_time >= self.send_interval:
                send_error = self.decide_control(line_found, error)
                self.send_control(send_error)

                if line_found:
                    line_found_count += 1
                else:
                    line_lost_count += 1

                self.log_control(error, send_error, line_found)

            if self.show_control_debug(frame, mask, error, line_found):
                break

        return {
            "started": True,
            "ack_received": self.timed_move_done,
            "line_found_count": line_found_count,
            "line_lost_count": line_lost_count,
        }

    def follow_for_estimated_distance(self, distance_m, speed_mps):
        if speed_mps <= 0:
            raise ValueError("speed_mps must be greater than 0")

        seconds = distance_m / speed_mps
        print(
            f"\nChassis estimated-distance follow: distance={distance_m:.2f}m, "
            f"speed={speed_mps:.2f}m/s, time={seconds:.2f}s"
        )

        return self.follow_for(seconds)

    def follow_forever(self):
        self.open()
        self.reset_control_filter()

        if not self.wait_for_stable_line():
            return

        print("Standalone chassis line-following started. Press q to quit.")

        try:
            while True:
                self.read_debug_frame()
                line_found, error, frame, mask = self.read_line_error()

                now = time.monotonic()

                if now - self.last_send_time >= self.send_interval:
                    send_error = self.decide_control(line_found, error)
                    self.send_control(send_error)
                    self.log_control(error, send_error, line_found)

                if self.show_control_debug(frame, mask, error, line_found):
                    break

        except KeyboardInterrupt:
            print("Standalone chassis line-following stopped by user")

        finally:
            self.close()


def main():
    chassis = LineFollowerChassis(show_debug=True)
    chassis.follow_forever()


if __name__ == "__main__":
    main()


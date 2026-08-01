import numpy as np
import serial
import time
from Robotic_Arm.rm_robot_interface import *

from config import Config

# from roboticstoolbox import Robot

"""
最后维护日期：2026年3月24日
维护者：黄旭熙
"""


class EffectorDevice:
    def __init__(self, port, baudrate):
        self.port = port
        self.baudrate = baudrate
        self.ser = None

    def open(self):
        # 打开串口
        self.ser = serial.Serial(self.port, self.baudrate, timeout=1)

    def close(self):
        # 关闭串口
        if self.ser is not None and self.ser.is_open:
            self.ser.close()

    def send_command(self, command):
        # 发送信号
        try:
            if self.ser is not None and self.ser.is_open:
                self.ser.write(command)
                self.ser.flush()
                time.sleep(0.2)
                return True
        except Exception as e:
            print(f"末端执行器发送指令失败: {e}")
        return False

    def receive_data(self):
        # 接受信号
        if self.ser is not None and self.ser.is_open:
            return self.ser.readline()
        return None


class RobotController:
    """
    总控制类：负责机械臂、末端执行器和相机的初始化和动作执行
    """

    def __init__(self, config, speed):
        self.config = config
        self.speed = speed
        self.arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        self.end_effector = EffectorDevice(self.config.effector_port, self.config.effector_baudrate)

        self.connect_all_devices()

    def connect_all_devices(self):
        """
        机械臂和末端执行器、相机的连接
        """

        # 连接机械臂并设置工具坐标系
        handle = self.arm.rm_create_robot_arm(self.config.robot_ip, self.config.robot_port)
        delete = self.arm.rm_delete_tool_frame("tf")
        print(f"是否成功删除坐标系：{delete}")
        offset_m = list(self.config.tool_pose_offset)
        offset_m[0] /= 1000.0
        offset_m[1] /= 1000.0
        offset_m[2] /= 1000.0
        length_m = self.config.tool_length / 1000.0
        tool_frame = rm_frame_t("tf",
                                offset_m,
                                self.config.tool_payload,
                                0., 0., length_m)
        self.arm.rm_set_manual_tool_frame(tool_frame)
        change_tool_frame = self.arm.rm_change_tool_frame("tf")
        print(f"是否成功转换坐标系：{change_tool_frame}")

        # 设置机械臂电源输出功率给末端供电
        voltage_status = self.arm.rm_set_tool_voltage(2)  # 设置末端电源输出 0：0V， 2：12V， 3：24V
        if voltage_status == 0:
            print("末端执行器12V电源输出设置成功")

        # 连接末端
        if self.config.use_end_effector:
            self.end_effector.open()
            self.ee_open()

    def disconnect_all_devices(self):
        """
        机械臂和末端执行器的断开
        """
        if self.config.use_end_effector and self.end_effector is not None:
            self.end_effector.close()
            print("末端执行器串口已断开")
        self.arm.rm_set_tool_voltage(0)
        print("末端执行器电源已安全关闭(0V)")
        self.arm.rm_delete_robot_arm()
        print("机械臂已断开连接")

    def get_tool_voltage(self):
        """
        获取末端工具电源输出当前的电压值
        :return: 具体的电压整数值(0, 12, 24),如果读取失败返回-1
        """
        result = self.arm.rm_get_tool_voltage()  # 官方返回的是一个元组 (状态码,电压类型)
        status = result[0]
        voltage_type = result[1]

        if status == 0:
            if voltage_type == 0:
                voltage_value = 0
            elif voltage_type == 2:
                voltage_value = 12
            elif voltage_type == 3:
                voltage_value = 24
            else:
                voltage_value = voltage_type
            print(f"当前末端电源输出电压为:{voltage_value}V")
            return voltage_value
        else:
            print(f"获取末端电源输出失败，错误码：{status}")
            return -1

    def get_current_joint(self):
        """
        获取机械臂当前六个关节角
        :return: [joint1,...,joint6] 单位：rad (弧度)
        """
        # 机械臂获取的是角度(degree)，在此转为弧度返回
        current_joints_deg = self.arm.rm_get_current_arm_state()[1]['joint']
        current_joints_rad = np.deg2rad(current_joints_deg).tolist()
        return current_joints_rad

    def get_current_pose(self):
        """
        获取机械臂当前位姿
        :return: [x,y,z,r,p,y] 单位：mm,rad
        """

        current_pose = self.arm.rm_get_current_arm_state()[1]['pose']
        current_pose[0] *= 1000.0
        current_pose[1] *= 1000.0
        current_pose[2] *= 1000.0
        return current_pose

    def move_robot_joint(self, joints_rad, speed=15):
        """
        机械臂关节空间运动
        :param joints_rad: [joint1,...,joint6] 单位：rad
        :param speed: 运动速度
        """
        # 接收到弧度后，转回角度
        joints_deg = np.rad2deg(joints_rad).tolist()
        ret = self.arm.rm_movej(joints_deg, v=speed, r=0, connect=0, block=1)
        return ret

    def move_robot_pose(self, pose, speed=15):
        """
        机械臂关节空间运动到目标位姿
        :param pose:[x,y,z,r,p,y] 单位：mm,rad
        :param speed:
        """
        pose_copy = list(pose)  # 复制一个全新的pose，不影响外部原始的pose
        pose_copy[0] /= 1000.0
        pose_copy[1] /= 1000.0
        pose_copy[2] /= 1000.0
        ret = self.arm.rm_movej_p(pose_copy, v=speed, r=0, connect=0, block=1)
        return ret

    def move_robot_line_pose(self, pose, speed=10):
        """
        机械臂笛卡尔空间直线运动
        :param pose:[x,y,z,r,p,y] 单位：mm,rad
        :param speed:
        """
        pose_copy = list(pose)  # 复制一个全新的pose，不影响外部原始的pose
        pose_copy[0] /= 1000.0
        pose_copy[1] /= 1000.0
        pose_copy[2] /= 1000.0
        ret = self.arm.rm_movel(pose_copy, v=speed, r=0, connect=0, block=1)
        return ret

    def move_to_original(self, speed=50):
        """
        机械臂回到初始位姿
        """
        joints_rad = np.deg2rad(Config.robot_original_joints).tolist()
        ret = self.move_robot_joint(joints_rad, speed=speed)
        return ret

    def move_to_basket(self, speed=40):
        """
        机械臂回到果篮
        """
        joints_rad = np.deg2rad(Config.robot_basket_joints).tolist()
        ret = self.move_robot_joint(joints_rad, speed=speed)
        return ret

    def ee_open(self):
        if self.config.use_end_effector and self.end_effector is not None:
            # print("发送末端执行器打开指令")
            ok = self.end_effector.send_command(b'\x01')
            print("末端执行器打开")
            return ok
        else:
            print("模拟末端执行器打开")
            return True

    def ee_close(self):
        if self.config.use_end_effector and self.end_effector is not None:
            # print("发送末端执行器闭合指令")
            ok = self.end_effector.send_command(b'\x02')
            print("末端执行器闭合")
            return ok
        else:
            print("模拟末端执行器闭合")
            return True


if __name__ == '__main__':
    import time
    controller = RobotController(Config, speed=30)

    # current_joint = controller.get_current_joint()
    # # print(current_joint)
    # controller.move_to_original()
    # controller.move_robot_joint(current_joint, 80)
    # controller.move_to_basket()
    # controller.move_to_original()

    # 循环测试末端是否能够稳定连续开合
    while 1:
        time.sleep(1)
        controller.ee_open()
        time.sleep(1)
        controller.ee_close()

    # controller.ee_open()
    # controller.ee_close()

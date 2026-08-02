import os

import numpy as np

class Config:
    """
    全局配置类：集中管理所有的文件路径、检测定位算法参数以及机器人数据
    """

    # 项目路径
    project_root = os.path.dirname(os.path.abspath(__file__))

    result_dir = os.path.join(project_root, "output", "result")  # 数据、可视化结果保存目录
    os.makedirs(result_dir, exist_ok=True)

    calibration_path = os.path.join(project_root, "output", "calibration_result_220mm")
    intrinsic_file = os.path.join(calibration_path, "intrinsicParams.txt")
    distortion_file = os.path.join(calibration_path, "distortion.txt")
    cam2end_R_file = os.path.join(calibration_path, "cam2end_R.txt")
    cam2end_T_file = os.path.join(calibration_path, "cam2end_T.txt")

    # 使用单类别 tomato 实例分割模型；类别 0 按采摘目标处理。
    # 该模型不包含 main_stem，因此不会启用基于主茎检测的动态避让。
    yolo_model_path = os.path.join(project_root, "output", "model", "tomato_pedicel_seg", "weights", "best.pt")
    yolo_expected_classes = ("stem",)

    # 底盘导航相机（MF-100 普通USB相机）
    nav_camera_name = "HD 720P Webcam"
    nav_camera_width = 1280
    nav_camera_height = 720
    nav_camera_fps = 30
    nav_camera_fourcc = "MJPG"

    # 机械臂扫描相机（RealSense）
    # 用于果实识别、深度定位和采摘
    arm_camera_serial = "139522075092"

    # ========================== 导航配置（黑色胶带寻迹） ==========================
    # 走停走模式：每次沿黑色胶带前进的距离，单位 m
    nav_move_distance_m = 0.8

    # 底盘前进速度，单位 m/s
    nav_speed = 0.25

    # 识别不到黑线后继续直行补偿的距离，单位 m；补偿完成后停下进入扫描采摘
    nav_line_lost_compensation_m = 0.0

    # 连续多少帧未检测到黑线后，才确认真正丢线
    nav_line_lost_frames = 8

    # 是否反转转向方向；如果发现车越纠越偏，就改成 True
    nav_invert_steering = False

    # 是否显示黑线识别调试窗口
    nav_show_window = True

    # 底盘前侧光源控制开关：
    # True 表示程序接管底盘光源，会通过 CAN 周期发送灯光控制帧；
    # False 表示程序不控制光源，底盘保持遥控器、默认逻辑或上一次指令的灯光状态。
    nav_light_enable = True

    # 底盘前侧光源模式：
    # "off"    = 程序接管后强制关灯
    # "on"     = 程序接管后常亮，但不按亮度值细调
    # "breath" = 呼吸灯模式，亮度周期变化，不建议用于黑线循迹
    # "custom" = 自定义亮度模式，使用下面的 nav_light_brightness，推荐用于循迹补光
    nav_light_mode = "off"

    # 自定义亮度，只有 nav_light_mode = "custom" 时生效。
    # 取值范围 0~100：0 表示不亮，100 表示最亮；当前推荐值 70，避免过曝影响黑线识别。
    nav_light_brightness = 100

    # 松灵底盘灯光控制帧建议 100ms 发送一次；协议接收超时约 500ms，周期发送可保持灯光状态。
    nav_light_send_period_s = 0.1

    # 黑线识别参数
    nav_roi_bottom = 0.35
    nav_min_area = 550
    nav_fixed_thresh = 60

    # PID 纠偏参数
    nav_pid_kp = 0.18
    nav_pid_ki = 0.0
    nav_pid_kd = 0.04
    nav_pid_max_integral = 0.3
    nav_pid_max_output = 0.18

    # CAN 设备参数
    nav_control_can_dll_paths = [
        os.path.join(project_root, "support_files", "ControlCAN.dll"),
        os.path.join(project_root, "ControlCAN.dll"),
        os.path.join(project_root, "navigation_guide", "ControlCAN.dll"),
    ]
    nav_dev_type = 4
    nav_dev_idx = 0
    nav_can_idx = 0
    nav_can_timing0 = 0x00
    nav_can_timing1 = 0x1C

    # 加载标定数据
    intrinsicParams = np.loadtxt(intrinsic_file)
    distortion = np.loadtxt(distortion_file)
    cam2end_R = np.loadtxt(cam2end_R_file)
    cam2end_T = np.loadtxt(cam2end_T_file)  # mm
    # 从内参矩阵中提取相机参数
    cam_params = {
        'fx': intrinsicParams[0, 0],
        'fy': intrinsicParams[1, 1],
        'cx': intrinsicParams[0, 2],
        'cy': intrinsicParams[1, 2]
    }

    # 机械臂和末端控制相关参数
    robot_ip = "192.168.1.18"
    robot_port = 8080
    tool_length = 220.0  # 末端长度(mm)
    tool_pose_offset = [0, 0, tool_length, 0, 0, 0]  # 工具位姿偏移量[x,y,z,rx,ry,rz](mm,rad)
    tool_payload = 0.8  # 末端负载质量(kg)
    effector_port = 'COM9'
    effector_baudrate = 9600
    use_end_effector = True  # 选择是否连接末端来使用
    # 用于设置move_robot_joint或者其他move的速度(0-100)
    slow_speed = 30
    normal_speed = 60
    fast_speed = 85
    interpolation_steps = 3  # 机械臂直线插补的步数
    use_previous_pose = True  # True：经过预采摘点；False：从起点直线插补到目标点
    previous_pose_distance_mm = 100.0  # 预采摘点位于目标点沿末端坐标系-Z方向的距离
    use_move_p = False  # 逆解失败后是否允许使用 move_p 继续从预采摘点到采摘点
    enable_fallback_picking_strategy = False  # 是否启用最低点备用采摘策略；关闭后不再触发或重试备用pose
    enable_ik_pose_visualization = False  # 是否显示直线插补逆解成功pose的三维可视化
    ik_pose_axis_length_mm = 45.0  # 三维可视化中每个pose坐标轴的长度(mm)

    # 机器人相关参数
    robot_original_joints = [5, 70, -130, 0, -40, 0]  # 机械臂初始位置的关节角度
    # robot_basket_joints = [98.1, 17.8, -140.3, -3.6, -37.3, -5.0]  # 采摘结束后放回果篮的关节角度(第一代机器)
    robot_basket_joints = [86.1, 22.2, -135.8, 2.9, -23.0, 0]  # 采摘结束后放回果篮的关节角度(第二代机器)

    robot_sin_joint = [[73.7, 29., -127.95, -92.6, -73.91, 99.32],  # 第一代机器
                       [67.8, 40.33, -143.33, -81.27, -65.82, 98.54],
                       [48.76, 62.86, -153.98, -66.55, -50.91, 75.59],
                       [19.9, 71.39, -145.06, -42.13, -30.26, 40.4],
                       [-19.9, 64.65, -133.42, 55.76, -24.14, -50.88],
                       # [-48.76, 50.24, -121.16, 97.96, -45.97, -81.26],
                       [-67.8, 37.64, -122.03, 101.89, -67.14, -89.39],
                       [-73.7, 29., -127.95, 92.6, -73.91, -99.32]]

    # 圣女果采摘网格法扫描(joint)
    """
       x=-360,          # 固定 X 坐标 (mm)
        z1=350,          # 扫描下边界
        z2=650,          # 扫描上边界
        y_period=680,    # Y方向扫描范围 (mm)
        rows=2,          # 行数
        cols=3,          # 列数
        pitch_deg=-30.0,   # 相机俯仰角，默认为 0 度
    """
    robot_grid_joint = [
        [80.83, 5.60, -61.25, -60.15, -99.68, 55.78],
        [-0.00, 37.00, -75.42, 0.00, -81.58, -0.00],
        [-80.83, 5.60, -61.25, 60.15, -99.68, -55.78],
        [-80.83, 23.60, -121.12, 60.81, -78.33, -92.00],
        [0.00, 71.79, -134.87, -0.00, -56.92, 0.00],
        [80.83, 23.60, -121.12, -60.81, -78.33, 92.00],
    ]
    # 检测与定位相关参数
    conf = 0.30  # Formal robot pedicel threshold.
    iou = 0.50  # Single-class pedicel NMS threshold.
    inference_imgsz = 1280  # 与高精度训练/验证分辨率一致，减少细果梗下采样损失
    inference_max_det = 100
    inference_retina_masks = True  # 使用原图分辨率掩码，提高边缘深度取样精度
    mask_threshold = 0.5
    depth_scale = 0.001  # RealSense深度单位，m/unit；启动相机后会读取实际传感器值
    depth_threshold = 1500  # 深度阈值，mm
    stem_duplicate_center_distance_mm = 25.0  # 缩小去重半径，避免把相邻果梗错误合并
    stem_duplicate_line_distance_mm = 12.0
    mask_erode_kernel_size = 3  # 定位点云取样前先收缩mask，减少果梗边缘混入背景深度
    mask_erode_iterations = 1

    depth_cluster_mad_scale = 2.5  # 用深度中位数主簇过滤背景点，数值越小越严格
    depth_cluster_min_window_mm = 15.0  # 收紧背景过滤，同时给RealSense噪声保留余量
    depth_cluster_min_keep_ratio = 0.3  # 主簇过小时不盲目接受偶发深度点
    depth_multiframe_count = 7  # 每次定位使用的深度帧数量，用中位数一致性剔除偶发坏帧
    depth_retry_frame_sets = 1  # 深度质量不合格导致定位失败时，额外重采的深度帧组数
    depth_quality_min_points = 20  # 过少点无法稳定估计果梗三维轴线
    depth_quality_min_valid_ratio = 0.25
    depth_quality_min_cluster_ratio = 0.5
    depth_quality_max_mad_mm = 15.0
    require_endpoint_exact_depth = True
    endpoint_depth_patch_radius_px = 8
    endpoint_min_depth_samples = 3
    endpoint_max_depth_std_mm = 15.0
    endpoint_max_z_gap_mm = 80.0
    endpoint_max_distance_mm = 200.0
    endpoint_min_depth_mm = 100.0
    depth_quality_max_span_mm = 100.0  # 兼顾斜向果梗的真实深度跨度
    depth_multiframe_outlier_mm = 30.0  # 静止扫描时80mm过宽，收紧以排除跳变帧
    depth_multiframe_fuse = True  # 对一致帧的P1/P2逐坐标取中位数，降低深度抖动
    depth_multiframe_min_consistent = 3  # 至少3帧一致才输出高精度定位
    allow_degraded_depth_pose = False  # 精度优先：不合格时重采/跳过，不直接按坏深度采摘

    # 果梗点云拟合参数（来自 pcl.py 的优化方法，单位均为 mm）
    point_filter_neighbors = 16  # 细长点云使用较小邻域，减少端点被误删
    point_filter_std_ratio = 1.5  # 粗滤保留真实果梗点，精确离群由后续RANSAC完成
    line_ransac_max_iterations = 2000  # RANSAC 最大随机采样次数
    line_ransac_threshold_mm = 8.0  # 15mm对细果梗过宽，容易把背景点纳入轴线
    line_ransac_min_inlier_ratio = 0.55
    line_endpoint_percentiles = (5.0, 95.0)  # 用稳健分位端点抑制残余极端点

    max_angle = np.deg2rad(60.0)  # 计算的果梗角度限制，即垂直地面的圆锥约束最大允许角度

    vertical_x_fallback_trigger_angle_deg = 30.0  # TCP Z相对基座-X目标侧水平轴的俯仰角阈值；上仰和下俯都按绝对值判断
    fallback_pose_down_pitch_deg = 20.0  # 备用姿态中 TCP X 轴从竖直向上朝基座-X目标侧下俯的角度

    picking_offset = [0.0, 0.0, 0.0]  # 正常采摘点XYZ偏置 (单位: mm)
    fallback_picking_offset = [0.0, 0.0, 0.0]  # 备用采摘策略XYZ偏置 (单位: mm)

    # 末端执行器坐标系下的偏置调整
    picking_offset_end_effector = [0.0, 0.0, 25.0]  # 正常采摘点末端执行器坐标系XYZ偏置
    fallback_picking_offset_end_effector = [10.0, 0.0, 25.0]  # 备用采摘策略末端执行器坐标系XYZ偏置

    enable_main_stem_dynamic_offset = False  # 是否启用基于果梗和主茎距离的动态偏置；False关闭，True打开
    scissor_max_opening_mm = 60.0  # 末端剪刀最大开口，单位 mm
    main_stem_safe_clearance_mm = 30.0  # 主茎到TCP采摘点的期望安全间距，单位 mm
    main_stem_offset_activation_distance_mm = 30.0  # 果梗和主茎小于该距离才沿远离主茎方向动态外偏
    main_stem_max_offset_mm = 15.0  # 单次最大外偏，避免采摘点偏离果梗过远

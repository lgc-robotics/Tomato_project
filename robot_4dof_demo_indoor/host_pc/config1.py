"""整机联调配置。"""

# 机械臂安全工作空间，单位：厘米。
# X 轴死区：0-1.5 厘米 和 58-60 厘米。
# Y 轴死区：0-4 厘米。
X_MIN = 1.5
X_MAX = 58.0
Y_MIN = 4.0
Y_MAX = 90.0

# 自动扫描位，单位：厘米。
# 固定在 Y=60 扫描三层，三个扫描点依次为 Z=1、21、41。
SCAN_X = 7.0
SCAN_Y_START = 45.0
SCAN_Y_MAX = 45.0
SCAN_Y_STEP = 1.0
SCAN_Z_START = 1.0
SCAN_Z_MAX = 41.0
SCAN_Z_STEP = 20.0
SCAN_SETTLE_SECONDS = 1

# 采摘后的果梗释放区域，单位：厘米。
# Y 轴只允许落在 26～55 或 72～80；55～72 之间是禁止释放的死区。
# 当原始 Y 落入死区时，程序会自动吸附到距离更近的 55 或 72。
RETREAT_Y_LOWER_MIN = 26.0
RETREAT_Y_LOWER_MAX = 55.0
RETREAT_Y_UPPER_MIN = 72.0
RETREAT_Y_UPPER_MAX = 80.0
# 释放点 Z 轴只能位于 0～10 厘米，X 轴继续沿用原回退逻辑。
RETREAT_Z_MIN = 0.0
RETREAT_Z_MAX = 10.0

# 每个扫描位连续检测多少帧。
# mask 点云会在单帧内同时使用整条果梗的全部有效深度点，不再依赖大量帧等待
# 某一个中心像素碰巧读到深度；2 帧用于目标合并和坐标平均。
NUM_FRAMES = 2

# 主动靠近后二次扫描最多检测多少帧。
# 二次扫描只负责寻找果梗自身深度；4 帧仍失败就立即沿用第一次结果，避免流程过慢。
REFINE_FRAMES = 4

# 主动二次扫描时，机械臂相对首次扫描点沿 X 轴前进的距离，单位：厘米。
# 例如首次扫描 X=7，设置 30 后，二次扫描候选 X 为 37。
ACTIVE_RESCAN_X_ADVANCE_CM = 30.0

# 二次扫描点与粗目标 X 坐标之间至少保留的距离，单位：厘米。
# 最终 X 会取“首次扫描 X + 前进量”和“粗目标 X - 此安全距离”中的较小值。
ACTIVE_RESCAN_TARGET_STANDOFF_CM = 20.0

# 相机安装在末端上方时，二次扫描点相对粗目标向下补偿的 Z 距离，单位：厘米。
# 若粗目标 Z 小于该值，二次扫描 Z 自动限制为 0。
ACTIVE_RESCAN_CAMERA_Z_OFFSET_CM = 20.0

# 机械臂 X 每前进 1 厘米时，相机参考深度预计减少多少厘米。
# 1.0 表示 X 前进 30 厘米，人工参考深度从 85 厘米动态变为约 55 厘米。
# 如果现场实测只减少 28 厘米，可改成 28/30，即约 0.93。
ACTIVE_RESCAN_REFERENCE_X_SCALE = 1.0

# 复拍深度相对“当前视角动态参考深度”允许的最大误差，单位：米。
# 默认 0.05 表示候选搜索会排除相差超过 5 厘米的点，且只有 CUT_MASK_FG 才算真实深度。
# 该值过大会把果梗后方背景当成果梗，过小则可能增加复拍失败率。
ACTIVE_RESCAN_REFERENCE_MAX_ERROR_M = 0.05

# 复拍时果梗 mask 外侧背景环的宽度，单位：像素。
# 程序会比较 mask 内候选深度和这一圈背景的中位深度，判断 mask 内是否仍然取到了背景。
ACTIVE_RESCAN_BACKGROUND_RING_WIDTH_PX = 6

# 背景环至少需要多少个有效深度像素才执行前景/背景对比。
# 背景本身没有深度时不会强行否决果梗候选，以免空旷场景漏采。
ACTIVE_RESCAN_BACKGROUND_RING_MIN_POINTS = 10

# 果梗候选必须比 mask 外圈背景至少近多少米，才证明它形成了独立前景深度。
# 0.01 表示至少近 1 厘米；内外深度几乎相同时，会判定为背景深度泄漏。
ACTIVE_RESCAN_MIN_MASK_DEPTH_CONTRAST_M = 0.01

# 到达主动二次扫描点后，等待机械臂和相机稳定的时间，单位：秒。
ACTIVE_RESCAN_SETTLE_SECONDS = 1.0

# 主动二次扫描中，哪些定位来源才算“果梗自身真实三维坐标”。
# 当前只接受整条果梗 mask 点云经过过滤和 PCA 后得到的结果。
ACTIVE_RESCAN_ACCEPT_DEPTH_MODES = ("MASK_PCA",)

# 至少需要多少帧相近的果梗自身深度，才结束复拍并采用该坐标。
# 深度波动范围沿用 MULTI_FRAME_DEPTH_STABLE_BAND_M。
ACTIVE_RESCAN_MIN_STABLE_FRAMES = 2

# 二次画面有多个果梗时，候选目标与首次粗坐标允许的最大 Y/Z 平面距离，单位：厘米。
# 匹配时不比较第一次可能错误的 X，只比较类别、Y/Z 和果梗角度。
ACTIVE_RESCAN_MATCH_MAX_YZ_DISTANCE_CM = 25.0

# 主动复拍时，同一果梗跨帧中心点允许移动的最大距离，单位：像素。
# 靠近后风吹摆动会表现为更大的像素位移；设置略大于普通扫描的 DIST_THRESHOLD。
ACTIVE_RESCAN_FRAME_MATCH_DISTANCE_PX = 45

# 第一个近距离视角仍没有真实果梗深度时，沿 Y 轴换视角的距离，单位：厘米。
# 当前设置为 0，关闭第三次侧向复拍；二次4帧失败后直接沿用第一次结果以节省时间。
ACTIVE_RESCAN_SIDE_RETRY_CM = 0.0

# 所有近距离复拍仍失败时，是否继续使用质量最高的主茎/参考深度尝试采摘。
# True 符合“最大程度采摘”；False 表示没有真实果梗深度就跳过该目标。
ACTIVE_RESCAN_USE_FALLBACK_AFTER_FAILURE = True

# 采摘预瞄参数，单位：厘米 / 秒。
ENABLE_PICK_PREAIM = True
PICK_PREAIM_X_OFFSET = 17
PICK_PREAIM_SETTLE_SECONDS = 0.25

# 视觉目标 Y 轴标定偏置，单位：厘米。
# 该偏置只加到识别出的采摘点及其预瞄位，不改扫描点坐标。
# 当程序计算的 Y 比实测值大时使用负数。
# 当前已换用12点手眼标定，默认不再叠加旧标定的 Y 轴临时补偿。
PICK_TARGET_Y_CALIBRATION_OFFSET_CM = 0.0

# 视觉目标 Z 轴标定偏置，单位：厘米。
# 该偏置只加到识别出的采摘点及其预瞄位，不改扫描层高。
# 当程序计算的 Z 比实测值低时使用正数。
# 当前已换用12点手眼标定，默认不再叠加旧标定的 Z 轴临时补偿。
PICK_TARGET_Z_CALIBRATION_OFFSET_CM = 0.0

# 末端旋转 05 与机械臂移动 04 联合发送的间隔，单位：秒。
# 必须先发 05，让下位机启动旋转；再发 04，才能让两个动作重叠执行。
ROTATE_MOVE_SEND_INTERVAL_SECONDS = 0.05

# 末端旋转参数：果梗倾角在该阈值内时不旋转。
STEM_NO_ROTATE_THRESHOLD_DEG = 15

# 末端旋转关节的绝对安全角度，单位：度。
# 上刀面不允许翻转朝向果实端，因此任何采摘动作都必须位于 [-90, 90] 度。
# 这是机械结构硬限制，不建议为了匹配果梗姿态而调大。
END_EFFECTOR_ROTATE_LIMIT_DEG = 90.0

# 新采摘策略参数，单位：厘米。
# 类别名需要和 YOLO 模型的 names 对上；如果你的模型类别名不同，改这里即可。
# 果梗类别名列表。YOLO 检测结果的类别名只要命中其中一个，就会当成果梗处理。
FRUIT_STEM_CLASS_NAMES = ("fruit_stem", "stem", "peduncle", "果梗")

# 主茎类别名列表。YOLO 检测结果的类别名只要命中其中一个，就会当成主茎处理。
MAIN_STEM_CLASS_NAMES = ("main_stem", "trunk", "branch", "主茎")

# 末端旋转中心到静刀片刀尖的距离。
# 这个值由机械结构决定，实验前按实际刀头位置复测。
STATIC_BLADE_TIP_OFFSET_CM = 1.42

# 末端旋转中心到动刀片刀尖的距离。
# 动刀片、齿轮或安装位置变化后，需要重新测量这个值。
MOVING_BLADE_TIP_OFFSET_CM = 1.00

# 近主茎判断安全距离。
# 当前等于静刀尖距离 + 动刀尖距离，即 1.42 + 1.00 = 2.42 厘米。
# 剪切点到主茎最近距离小于该值时，进入近主茎导入剪切策略。
CUT_SAFE_DISTANCE_CM = STATIC_BLADE_TIP_OFFSET_CM + MOVING_BLADE_TIP_OFFSET_CM

# 高风险提示距离。
# 小于该距离时仍然尝试导入剪切，但会在终端打印提醒，方便现场重点观察。
CUT_HIGH_RISK_DISTANCE_CM = 2.0

# 是否启用“静刀/动刀贴近果梗”的横向坐标偏置。
# False：标定测试模式，机械臂旋转中心直接对准果梗剪切点；刀片偏置计算保留但不生效。
# True：恢复原策略，近主茎时根据所选刀片把旋转中心横向偏移到导入位置。
ENABLE_BLADE_CONTACT_OFFSET = True

# 导入时让果梗落在“旋转中心到所选刀尖”的中间位置。
# 0.5 表示半程；如果想更靠近刀尖可调大，想更靠近中心可调小。
GUIDE_BLADE_CONTACT_RATIO = 0.5

# 近主茎导入剪切时，先沿 X 方向插入刀口的距离。
# 过小可能果梗没进刀口；过大可能碰到果实或主茎。
GUIDE_INSERT_DEPTH_CM = 1.5

# 近主茎导入剪切时，沿远离主茎方向推开果梗的距离。
# 过小可能分离不够；过大可能把果梗或果实推偏。
GUIDE_PUSH_AWAY_CM = 1.0

# 近主茎导入剪切时，推离主茎的同时 X 轴继续前送的距离。
# 目的是让果梗落入剪切受力点。
GUIDE_FORWARD_X_CM = 3.0

# 果梗 mask 上参与选择剪切点的起始比例。
# 0.05 表示去掉果梗起始端 5%，避免剪到端点噪声。
STEM_CUT_REGION_START_RATIO = 0.05

# 果梗 mask 上参与选择剪切点的结束比例。
# 0.95 表示去掉果梗结束端 5%，只在中间 90% 区域里取果梗中心剪切点。
STEM_CUT_REGION_END_RATIO = 0.95

# 末端旋转电机方向修正。
# 0 度时静刀在右侧；如果现场发现旋转方向和期望相反，把 -1 改成 1。
END_EFFECTOR_CLOCKWISE_SIGN = -1

# 图像中果梗剪切点在主茎左侧时使用静刀贴近；右侧时使用动刀贴近。
# 如果现场发现相机画面左右镜像导致判断反了，把这个值改成 False。
FRUIT_LEFT_USES_STATIC_BLADE = True

# YOLO 参数。
IMG_SIZE = 640
YOLO_CONF = 0.38
# YOLO 推理阶段保留候选的最低置信度。最终采摘仍由 YOLO_CONF 筛选。
# 设置得比 YOLO_CONF 低，便于结果图和终端区分“模型没看见”和“置信度不足”。
YOLO_INFERENCE_CONF = 0.10
# YOLO 非极大值抑制的 IoU 阈值。
YOLO_IOU = 0.40
# True 让 Ultralytics 尽量返回原图尺寸 mask，避免细果梗 mask 缩放后错位。
YOLO_RETINA_MASKS = True
DIST_THRESHOLD = 35
MIN_VOTE = 1

# mask 三维点云定位参数。
# 二值化 YOLO 概率 mask 的阈值；只有大于该值的像素进入点云。
MASK_THRESHOLD = 0.50
# 一个果梗实例至少需要多少个 mask 像素。
MIN_MASK_PIXELS = 10
# 经过深度和离群点过滤后，至少保留多少个三维点才允许 PCA 拟合。
MIN_PCA_POINTS = 30
# PCA 第一主轴与第二主轴奇异值的最小比值；越大越要求点云呈细长果梗形状。
MIN_PCA_LINEARITY = 2.0
# PCA 拟合线段允许的最短/最长长度，单位：厘米，用于拒绝退化点云和异常长背景。
MIN_PCA_LINE_LENGTH_CM = 1.0
MAX_PCA_LINE_LENGTH_CM = 15.0
# 先删除点云深度最前和最后各多少百分比，降低极端深度噪声影响。
POINT_CLOUD_DEPTH_TRIM_LOW_PERCENT = 5.0
POINT_CLOUD_DEPTH_TRIM_HIGH_PERCENT = 95.0
# 单个深度簇允许围绕其中位深度波动的范围，单位：米。
# 程序会拆出多个候选深度簇，不再固定选择最近的一簇。
POINT_CLOUD_NEAR_DEPTH_PERCENT = 30.0  # 仅供旧版兼容代码使用。
POINT_CLOUD_DEPTH_BAND_M = 0.03
# 同一画面不同果梗/主茎深度相差不超过该值时，认为属于同一植物平面。
POINT_CLOUD_SCENE_CONSENSUS_BAND_M = 0.08
# 果梗候选深度簇与画面共识深度最多允许相差多少米。
# 超过该范围说明更可能是 RealSense 近距离伪点或后方背景，不用于采摘。
POINT_CLOUD_SCENE_MAX_ERROR_M = 0.15
# 一个实例中的候选深度簇至少需要多少个点，过小的孤立簇直接忽略。
POINT_CLOUD_CLUSTER_MIN_POINTS = 20
# 最后按三维质心距离保留该百分比以内的点，去掉空间上孤立的离群点。
POINT_CLOUD_DISTANCE_KEEP_PERCENT = 95.0

# RealSense 深度过滤参数，单位：米。
MIN_DEPTH = 0.15
MAX_DEPTH = 1.20
DEPTH_CONSISTENCY_THRESHOLD = 0.20

# 人工测得的果梗平面深度，单位：米。
# 这是“最大程度采摘”策略的兜底深度：RealSense 打到背景时，可用这个深度反推坐标。
# 每次现场相机位置变化后都要重新手动测量并修改。
TARGET_DEPTH_REFERENCE_M = 0.85

# 参考深度软范围，单位：米。
# 新剪切点深度策略中，候选深度落在“人工参考深度±该范围”内时不额外扣分；
# 旧检测框兜底策略中，会优先选这个范围内的深度点。
TARGET_DEPTH_BAND_M = 0.10

# 围绕最终剪切点搜索深度的初始半径，单位：像素。
# 细果梗建议 6-10；太小可能找不到点，太大可能混入背景。
CUT_DEPTH_SEARCH_RADIUS_PX = 8

# 初始半径找不到可靠深度时，允许逐步扩大到的最大半径，单位：像素。
# 值越大越不容易漏采，但越容易引入背景深度。
CUT_DEPTH_MAX_RADIUS_PX = 25

# 深度搜索半径每次扩大的步长，单位：像素。
CUT_DEPTH_RADIUS_STEP_PX = 4

# 剪切点附近至少需要多少个有效深度点才采用当前搜索半径。
# 3 表示不再让单个偶然深度像素直接成为果梗深度候选。
CUT_DEPTH_MIN_POINTS = 3

# 果梗 mask 内同一深度连通簇至少需要多少个像素。
# 只有空间相邻、深度也相近的像素达到该数量，才允许标记为 mask 内候选。
CUT_DEPTH_CONNECTED_MIN_POINTS = 3

# 构造果梗深度连通簇时，相邻像素允许的最大深度差，单位：米。
# 0.02 表示相邻像素深度相差不超过 2 厘米时才归入同一簇。
CUT_DEPTH_CLUSTER_BAND_M = 0.02

# 是否对足够粗的果梗 mask 做一次轻度自适应腐蚀后再搜索深度。
# 细果梗不会腐蚀，避免整个 mask 被抹掉；粗果梗会去掉最容易混入背景的边缘。
ENABLE_CUT_MASK_ADAPTIVE_EROSION = True

# 只有 mask 内部最大距离达到该值时才执行腐蚀，单位：像素。
# 2.5 大致表示果梗局部宽度至少约 5 像素。
CUT_MASK_EROSION_MIN_HALF_WIDTH_PX = 2.5

# 自适应腐蚀使用的椭圆核尺寸和迭代次数。
# 当前 3x3、1 次属于轻度腐蚀，不建议直接增大到 5x5。
CUT_MASK_EROSION_KERNEL_SIZE = 3
CUT_MASK_EROSION_ITERATIONS = 1

# 腐蚀后至少保留多少个 mask 像素，否则自动退回原始 mask。
CUT_MASK_EROSION_MIN_REMAINING_POINTS = 5

# 首次扫描时果梗 mask 外侧背景环宽度，单位：像素。
# 程序比较 mask 内候选和外圈中位深度，判断候选是否形成独立前景。
CUT_MASK_BACKGROUND_RING_WIDTH_PX = 4

# 背景环至少需要多少个有效深度点才执行内外深度对比。
# 空旷背景没有足够深度时不会强行否决，只会把候选标记为“未验证前景”。
CUT_MASK_BACKGROUND_RING_MIN_POINTS = 10

# 果梗候选必须比 mask 外圈至少近多少米，才标记为“真实前景深度”。
# 0.01 表示至少近 1 厘米；内外深度相同会被判为背景泄漏。
CUT_MASK_MIN_BACKGROUND_DEPTH_CONTRAST_M = 0.01

# 是否为每个果梗、每一帧保存 mask 内原始深度热力图和压缩数值文件。
# 文件保存在当次“运行记录/运行_时间/depth_diagnostics”目录中。
SAVE_CUT_DEPTH_DIAGNOSTICS = True

# 深度点离剪切点像素距离的评分权重。
# 值越大，越偏向选择离剪切点最近的深度。
CUT_DEPTH_DISTANCE_WEIGHT = 1.0

# 深度点偏离人工参考深度的评分权重。
# 值越大，越不容易选到背景墙深度。
CUT_DEPTH_REFERENCE_WEIGHT = 80.0

# 候选点不在果梗 mask 内时的惩罚分。
# 值越大，越优先选择果梗 mask 内的深度；为最大程度采摘，不直接丢弃 mask 外近邻。
CUT_DEPTH_OUTSIDE_MASK_PENALTY = 30.0

# 背景墙深度，单位：米。设为 0 表示关闭背景墙惩罚。
# 如果现场测得相机到墙约 101 厘米，就填 1.01。
BACKGROUND_DEPTH_M = 1.50

# 判断“接近背景墙”的深度范围，单位：米。
# 当前 0.04 表示 BACKGROUND_DEPTH_M±4 厘米内都会被认为像背景。
BACKGROUND_DEPTH_BAND_M = 0.04

# 候选深度接近背景墙时的惩罚分。
# 值越大，越不容易选墙面；但不会硬跳过，仍保留最大采摘倾向。
CUT_DEPTH_BACKGROUND_PENALTY = 1000.0

# 是否允许没有可靠 RealSense 深度时，使用 TARGET_DEPTH_REFERENCE_M 兜底采摘。
# True 表示只要识别到果梗且坐标合法，就尽量给出采摘坐标。
ALLOW_REFERENCE_DEPTH_FALLBACK = True

# 如果最佳 RealSense 深度偏离人工参考深度超过该值，改用人工参考深度兜底。
# 例如参考 0.90、候选 1.11，误差 0.21 大于 0.12，就认为它大概率打到背景。
CUT_DEPTH_REFERENCE_FALLBACK_TRIGGER_M = 0.12

# 果梗剪切点深度不可靠时，是否优先借用最近主茎的局部深度。
# True 适合细果梗：果梗本身深度容易丢，主茎更粗，深度通常更稳定。
USE_MAIN_STEM_DEPTH_FALLBACK = True

# 主茎深度兜底的初始搜索半径，单位：像素。
# 以“离剪切点最近的主茎点”为中心，在主茎 mask 内找深度。
MAIN_STEM_DEPTH_SEARCH_RADIUS_PX = 8

# 主茎初始半径找不到足够深度时，允许扩大到的最大半径，单位：像素。
# 值越大越容易找到深度，但也越可能混入主茎边缘或背景噪声。
MAIN_STEM_DEPTH_MAX_RADIUS_PX = 30

# 主茎深度搜索半径每次扩大的步长，单位：像素。
MAIN_STEM_DEPTH_RADIUS_STEP_PX = 4

# 主茎局部区域至少需要多少个有效深度点才采用主茎深度。
# 主茎较粗，默认可以比果梗严格一点；如果现场漏得多，可调小到 2 或 3。
MAIN_STEM_DEPTH_MIN_POINTS = 3

# 主茎局部深度稳定范围，单位：米。
# 会优先使用接近局部中位数的深度点，减少边缘混入背景造成的跳变。
MAIN_STEM_DEPTH_STABLE_BAND_M = 0.04

# 主茎深度相对人工参考深度允许偏差的最大值，单位：米。
# 0 表示不按人工参考深度过滤；建议保留 0.20 左右，防止误采到远处背景。
MAIN_STEM_DEPTH_REFERENCE_MAX_ERROR_M = 0.20

# 多帧融合时，MASK_PCA 是当前唯一正式定位来源。
# 旧 CUT_MASK、MAIN_STEM、REF 等模式仅供历史兼容，新 detect_targets 不再生成。
GOOD_DEPTH_MODES = ("MASK_PCA",)

# 数值可能稳定、但没有足够前景证据的果梗候选模式。
UNVERIFIED_FRUIT_DEPTH_MODES = ("CUT_MASK", "CUT_NEAR")

# 多帧深度融合时，认为是主茎兜底深度的模式。
# 主茎深度比人工参考更贴近现场，但仍不如果梗自身深度。
MAIN_STEM_DEPTH_MODES = ("MAIN_STEM",)

# 至少拍到多少帧稳定果梗深度，才把它当成最高优先级结果。
# 2 表示两帧果梗深度相近就认为可信；想更稳可调成 3。
MIN_GOOD_DEPTH_FRAMES = 2

# 如果没拍到稳定果梗深度，至少多少帧稳定主茎深度才采用主茎兜底。
# 主茎较粗，通常 2 帧就能明显压住单帧跳变。
MIN_MAIN_STEM_DEPTH_FRAMES = 2

# 多帧稳定深度允许的最大波动，单位：米。
# 0.03 表示同一目标多帧深度差在 3 厘米内就算稳定。
MULTI_FRAME_DEPTH_STABLE_BAND_M = 0.03

# 是否在已经拍到稳定果梗深度后提前结束当前扫描位检测。
# True 可以减少等待；如果你想每个扫描位固定拍满 NUM_FRAMES，就改成 False。
ALLOW_EARLY_STOP_ON_STABLE_DEPTH = True

# 至少拍到多少帧后才允许提前停止。
# 防止第一两帧偶然稳定就过早结束，建议 3-4。
EARLY_STOP_MIN_FRAMES = 4

# 当视觉换算出来的 X 轴坐标超过机械臂行程时，是否把这次深度判为可疑。
# True 表示先用 TARGET_DEPTH_REFERENCE_M 重新计算整组 X/Y/Z，再决定是否限幅。
USE_REFERENCE_DEPTH_WHEN_X_OUT_OF_RANGE = True

# X 轴触发参考深度重算的边界余量，单位：厘米。
# 0.0 表示只要超过 X_MIN/X_MAX 就重算；如果只想明显越界才重算，可改成 0.5 或 1.0。
X_RANGE_REFERENCE_RECHECK_MARGIN_CM = 0.0

# 旧检测框深度兜底时，参考深度范围内至少需要多少个点才认为这片深度可用。
# 现在主流程优先走剪切点附近深度；这个参数主要用于没有 mask 或策略失败时。
REFERENCE_DEPTH_MIN_POINTS = 5

# 旧检测框深度兜底时，用来判断一批深度点是否稳定的范围，单位：米。
# 值越小越严格；值越大越容易接受散乱深度。
DEPTH_STABLE_BAND_M = 0.03

# 是否显示并保存最终检测结果图。
# True 方便现场看剪切点、深度采样点和坐标；False 可减少显示窗口干扰。
SHOW_FINAL_RESULT = True

# 最终结果图窗口等待时间，单位：毫秒。
# 1 表示快速刷新；如果想停住观察图片，可以临时改成 500 或 1000。
FINAL_RESULT_WAIT_MS = 1

# 是否显示 YOLO 果梗置信度阈值图。
# 该图与最终三维定位图分开保存，会显示推理阈值以上的所有果梗及其置信度。
# 绿色表示达到 YOLO_CONF 采摘阈值，橙色表示已识别但尚未达到采摘阈值。
SHOW_DETECTION_THRESHOLD_RESULT = True

# RealSense 视频流配置。
# 细果梗在高分辨率下能覆盖更多 mask/深度像素，是点云定位成功率的重要条件。
COLOR_WIDTH = 1280
COLOR_HEIGHT = 720
DEPTH_WIDTH = 1280
DEPTH_HEIGHT = 720
FPS = 30
MODEL_PATH = "best2.pt"

# 机械臂 / STM32 串口配置。
SERIAL_PORT = "COM10"
BAUDRATE = 115200
SERIAL_TIMEOUT = 1
ROBOT_ACK_TIMEOUT_SECONDS = 25.0

# 整机联调参数。
TRACK_TOTAL_METERS = 50.0
STATION_SPACING_METERS = 1.0
CHASSIS_MOVE_ACK_TIMEOUT_SECONDS = 5.0
PICK_BEFORE_FIRST_MOVE = False
STATION_SETTLE_SECONDS = 1.0

# 底盘循迹串口 / 摄像头参数。
CHASSIS_SERIAL_PORT = "COM10"
CHASSIS_BAUDRATE = 115200
CHASSIS_SERIAL_TIMEOUT = 1
CHASSIS_CAMERA_INDEX = 2
CHASSIS_CAMERA_WIDTH = 640
CHASSIS_CAMERA_HEIGHT = 480
CHASSIS_SEND_INTERVAL = 0.05
# 是否显示底盘循迹原图和黑线阈值图。
CHASSIS_SHOW_DEBUG = True
# 调试窗口刷新间隔；0.10 秒约为 10 帧/秒，只限制显示，不降低循迹计算频率。
CHASSIS_DEBUG_DISPLAY_INTERVAL_SECONDS = 0.10
CHASSIS_OPEN_CAMERA_AT_START = False
CHASSIS_RELEASE_CAMERA_DURING_PICK = True

# 每次重新打开底盘相机后，先丢弃这些帧，让曝光和白平衡恢复稳定。
CHASSIS_CAMERA_WARMUP_FRAMES = 20
# 正式发送底盘控制帧前，要求连续检测到黑线的帧数。
CHASSIS_START_STABLE_FRAMES = 6
# 上述连续帧中，黑线横向误差最大值与最小值允许相差的像素数。
CHASSIS_START_STABLE_SPREAD_PX = 30
# 等待黑线稳定的最长时间；超时后取消本次底盘移动，不盲目转向。
CHASSIS_START_STABLE_TIMEOUT_SECONDS = 3.0
# 对最近若干帧误差取中位数，抑制单帧曝光异常和错误轮廓。
CHASSIS_ERROR_FILTER_WINDOW = 5
# 相邻两次下发的转向误差最多变化多少像素，避免突然大幅转向。
CHASSIS_MAX_ERROR_STEP_PX = 40
# 黑线轮廓的最小面积，小于该值按噪点处理。
CHASSIS_MIN_CONTOUR_AREA_PX = 80
# 黑线轮廓占循迹区域的最大比例，过大的暗块按曝光异常处理。
CHASSIS_MAX_CONTOUR_AREA_RATIO = 0.25
# 控制台循迹信息的最小打印间隔，避免电脑低电量时频繁输出拖慢循环。
CHASSIS_LOG_INTERVAL_SECONDS = 0.25

# 是否启用“深度不可靠时主动靠近复拍”。
# True：稳定果梗自身深度直接采摘，其余目标先移动到近距离视角重新测深度。
# False：完全关闭主动复拍，直接沿用首次扫描结果。
USE_REFINE_PICK = False

# 是否保存每次运行的完整控制台日志、所有结果图和图文 HTML 报告。
# True 时会在项目目录下自动建立一个按启动时间编号的独立文件夹。
ENABLE_RUN_RECORDING = True

# 运行记录总目录名称；每次启动会在其中新建“运行_年月日_时分秒”文件夹。
RUN_RECORD_FOLDER_NAME = "运行记录"

# HTML 中每张结果图最多附带多少个字符的对应控制台信息。
# 完整控制台日志不受该值影响，始终全部保存在“控制台日志.txt”中。
RUN_REPORT_MAX_LOG_CHARS_PER_IMAGE = 30000

# RealSense 等帧保护参数。
REALSENSE_FRAME_TIMEOUT_MS = 5000
REALSENSE_FRAME_RETRY_COUNT = 3
REALSENSE_FRAME_RETRY_DELAY_SECONDS = 0.2

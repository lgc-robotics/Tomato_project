import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')

# from runtime_bootstrap import ensure_runtime
#
# ensure_runtime()  # 搜索一下是干嘛的

import matplotlib

matplotlib.use('TkAgg')
import numpy as np
import time
import roboticstoolbox as rtb
from spatialmath import SE3
from config import Config
from controller import RobotController
"""
最后维护日期：2026年4月7日
维护者：刘嘉梁
"""


class RobotKinematics:
    def __init__(self, jacobian_method='differential', tool_length=Config.tool_length):
        """
        添加各关节角度约束
        初始化机器人参数
        jacobian_method: 选择雅可比计算方法：
                                'differential' (微分/解析法),
                                'vector' (矢量积法),
                                'numerical' (数值法)
        tool_length: 工具长度 (mm)
        """
        # 关节角度限制配置
        # self.q_min = np.radians([-178, -178, -178, -178, -178, -360])
        # self.q_max = np.radians([178, 178, 145, 178, 178, 360])
        self.q_min = np.radians([-178, -105, -178, -178, -178, -360])
        self.q_max = np.radians([178, 100, 145, 178, 178, 360])
        q_safe = np.pi/180*3
        self.q_min = self.q_min + q_safe
        self.q_max = self.q_max - q_safe

        self.tool_length = tool_length
        self.jacobian_method = jacobian_method

        # 定义雅可比计算方法字典
        methods = {
            'differential': self._differential_jacobian,
            'vector': self._vector_jacobian,
        }

        # 绑定函数 (实现自动选择配对)
        self.input_jacobian = methods[jacobian_method]

    # 获取雅可比
    def get_jacobian(self, q):
        # 调用刚才绑定好的函数
        return self.input_jacobian(q)

    @staticmethod
    def _DHTrans(alpha, a, d, theta):
        """
        根据改进 D-H 参数计算单个关节的齐次变换矩阵
        输入:
        alpha : float连杆扭角
        a : float连杆长度
        d : float连杆偏距
        theta : float关节转角
        输出：4x4 齐次变换矩阵.
        """
        T = np.array([
            [np.cos(theta), -np.sin(theta), 0, a],
            [np.sin(theta) * np.cos(alpha), np.cos(theta) * np.cos(alpha), -np.sin(alpha), -np.sin(alpha) * d],
            [np.sin(theta) * np.sin(alpha), np.cos(theta) * np.sin(alpha), np.cos(alpha), np.cos(alpha) * d],
            [0, 0, 0, 1]
        ])
        return T

    def fkine(self, theta):
        """
        正运动学求解
        根据给定的关节角度，计算各关节的变换矩阵及末端最终位姿。

        输入:
        theta : 6个关节的角度列表或数组 (单位: 弧度).

        输出:
        tuple : (matrices, T_final)
            matrices (list): 包含6个关节相对于上一关节的变换矩阵列表 [A1, A2, ..., A6].

            T_final : 4x4 的末端相对于基座的最终绝对齐次变换矩阵.

        """
        # 输入：D-H参数
        the = np.array(theta)
        initial_offset = np.array([0, - np.pi / 2, + np.pi / 2, 0, + np.pi, + np.pi])  # 初始偏移
        th = the + initial_offset  # 偏移后的theta
        d = np.array([162.5, 0, 0, 405, 0, 132.3 + self.tool_length])
        a = np.array([0, -86, 380, 69, 0, 0])
        alp = np.array([0, -np.pi / 2, 0, np.pi / 2, -np.pi / 2, -np.pi / 2])

        # 计算各变换矩阵 (A1 - A6)填入列表
        matrices = []
        T_final = np.eye(4)  # 用于累积结果
        for i in range(6):
            # 计算当前关节变换矩阵
            Ai = self._DHTrans(alp[i], a[i], d[i], th[i])
            matrices.append(Ai)

            # 顺便乘到总结果里 (利用 @ 矩阵乘法)
            T_final = T_final @ Ai

            # 同时返回列表[A1, A2, A3, A4, A5, A6] 和 最终末端4×4矩阵
        return matrices, T_final  # 返回

    def _differential_jacobian(self, q):
        """
        【微分法】求解雅可比矩阵。
        通过矩阵的显式微分推导公式计算。

        输入:
        q : 6个关节的角度数组.

        输出:
        6x6 雅可比矩阵.

        """
        # 获取所有单关节矩阵 [A1...A6]
        A, _ = self.fkine(q)

        # 计算累积矩阵 (从后往前乘)
        T56 = A[5]
        T46 = A[4] @ T56
        T36 = A[3] @ T46
        T26 = A[2] @ T36
        T16 = A[1] @ T26
        T06 = A[0] @ T16

        # 3. 构建各列 (显式展开计算)

        # 第1列 (基于 T16)
        j11 = np.array([
            -T16[0, 0] * T16[1, 3] + T16[1, 0] * T16[0, 3],
            -T16[0, 1] * T16[1, 3] + T16[1, 1] * T16[0, 3],
            -T16[0, 2] * T16[1, 3] + T16[1, 2] * T16[0, 3],
            T16[2, 0], T16[2, 1], T16[2, 2]
        ])

        # 第2列 (基于 T26)
        j22 = np.array([
            -T26[0, 0] * T26[1, 3] + T26[1, 0] * T26[0, 3],
            -T26[0, 1] * T26[1, 3] + T26[1, 1] * T26[0, 3],
            -T26[0, 2] * T26[1, 3] + T26[1, 2] * T26[0, 3],
            T26[2, 0], T26[2, 1], T26[2, 2]
        ])

        # 第3列 (基于 T36)
        j33 = np.array([
            -T36[0, 0] * T36[1, 3] + T36[1, 0] * T36[0, 3],
            -T36[0, 1] * T36[1, 3] + T36[1, 1] * T36[0, 3],
            -T36[0, 2] * T36[1, 3] + T36[1, 2] * T36[0, 3],
            T36[2, 0], T36[2, 1], T36[2, 2]
        ])

        # 第4列 (基于 T46)
        j44 = np.array([
            -T46[0, 0] * T46[1, 3] + T46[1, 0] * T46[0, 3],
            -T46[0, 1] * T46[1, 3] + T46[1, 1] * T46[0, 3],
            -T46[0, 2] * T46[1, 3] + T46[1, 2] * T46[0, 3],
            T46[2, 0], T46[2, 1], T46[2, 2]
        ])

        # 第5列 (基于 T56)
        j55 = np.array([
            -T56[0, 0] * T56[1, 3] + T56[1, 0] * T56[0, 3],
            -T56[0, 1] * T56[1, 3] + T56[1, 1] * T56[0, 3],
            -T56[0, 2] * T56[1, 3] + T56[1, 2] * T56[0, 3],
            T56[2, 0], T56[2, 1], T56[2, 2]
        ])

        j66 = np.array([0, 0, 0, 0, 0, 1])

        # 构建变换矩阵 T_cam2base 从工具坐标系转成世界坐标系
        T = np.array([
            [T06[0, 0], T06[0, 1], T06[0, 2], 0, 0, 0],
            [T06[1, 0], T06[1, 1], T06[1, 2], 0, 0, 0],
            [T06[2, 0], T06[2, 1], T06[2, 2], 0, 0, 0],
            [0, 0, 0, T06[0, 0], T06[0, 1], T06[0, 2]],
            [0, 0, 0, T06[1, 0], T06[1, 1], T06[1, 2]],
            [0, 0, 0, T06[2, 0], T06[2, 1], T06[2, 2]]
        ])

        jacobian = T @ np.column_stack([j11, j22, j33, j44, j55, j66])
        return jacobian

    def _vector_jacobian(self, q):
        """
        【矢量积法】求解雅可比矩阵

        公式:
        Col_i = [ z_i x (p_end - p_i) ]  <- 线速度部分
                [ z_i                 ]  <- 角速度部分

        输入:
        q :
        6个关节的角度数组 (单位: 弧度).

        输出:
        6x6 雅可比矩阵.
                """
        # 获取所有单关节变换矩阵
        matrices, _ = self.fkine(q)
        A1, A2, A3, A4, A5, A6 = matrices

        # 计算累积变换矩阵
        T01 = A1
        T02 = T01 @ A2
        T03 = T02 @ A3
        T04 = T03 @ A4
        T05 = T04 @ A5
        T06 = T05 @ A6  # 末端

        P_end = T06[:3, 3]

        # 提取各连杆末端的 Z 轴 (z1-z6) 和 原点位置 (P1-P6)
        z1 = T01[:3, 2]
        P1 = T01[:3, 3]
        z2 = T02[:3, 2]
        P2 = T02[:3, 3]
        z3 = T03[:3, 2]
        P3 = T03[:3, 3]
        z4 = T04[:3, 2]
        P4 = T04[:3, 3]
        z5 = T05[:3, 2]
        P5 = T05[:3, 3]
        z6 = T06[:3, 2]
        P6 = T06[:3, 3]

        jacobian = np.zeros((6, 6))
        # noinspection PyUnreachableCode
        # 第 1 列 (使用 z1, P1)
        jacobian[:3, 0] = np.cross(z1, P_end - P1)
        jacobian[3:, 0] = z1
        # noinspection PyUnreachableCode
        # 第 2 列 (使用 z2, P2)
        jacobian[:3, 1] = np.cross(z2, P_end - P2)
        jacobian[3:, 1] = z2
        # noinspection PyUnreachableCode
        # 第 3 列 (使用 z3, P3)
        jacobian[:3, 2] = np.cross(z3, P_end - P3)
        jacobian[3:, 2] = z3
        # noinspection PyUnreachableCode
        # 第 4 列 (使用 z4, P4)
        jacobian[:3, 3] = np.cross(z4, P_end - P4)
        jacobian[3:, 3] = z4
        # noinspection PyUnreachableCode
        # 第 5 列 (使用 z5, P5)
        jacobian[:3, 4] = np.cross(z5, P_end - P5)
        jacobian[3:, 4] = z5
        # noinspection PyUnreachableCode
        # 第 6 列 (使用 z6, P6)
        # 注意: P_end - P6 = 0向量, 叉积为0, 只保留姿态部分
        jacobian[:3, 5] = np.cross(z6, P_end - P6)
        jacobian[3:, 5] = z6

        return jacobian


    @staticmethod
    def _pose_error(t_curr, t_target):
        """
        计算当前位姿与目标位姿之间的误差向量 (6维)。
        - 位置误差：线性欧氏距离 (dx, dy, dz)
        - 姿态误差：基于轴角法的旋转向量 (rx, ry, rz)

        输入:
        t_curr :
            当前末端的 4x4 齐次变换矩阵.
        t_target :
            目标位姿的 4x4 齐次变换矩阵.

       输出:
            形状为 (6,) 的误差向量 [dx, dy, dz, rx, ry, rz].
        """
        # 位置误差 (当前 - 目标)
        p_err = t_curr[:3, 3] - t_target[:3, 3]

        # 姿态误差 (当前 * 目标的转置)
        R_curr = t_curr[:3, :3]
        R_target = t_target[:3, :3]

        R_err = R_curr @ R_target.T

        """
        轴角法：
        提取旋转矩阵并转换为旋转向量
        cos = (np.trace(R_diff) - 1) / 2
        罗德里格斯公式:R = I + sin(θ)K + (1-cos(θ)) * K**2  (I = np.eye(3))
        R - R**T_cam2base = 2sin(θ)K  (K是旋转轴u = [ux, uy, uz]的反对称矩阵)
            [ 0   -uz   uy ]
        K = [ uz   0   -ux ]
            [-uy   ux   0  ]
        """

        trace_val = np.trace(R_err)
        cos_theta = (trace_val - 1) / 2.0
        cos_theta = np.clip(cos_theta, -1.0, 1.0)

        theta = np.arccos(cos_theta)

        if np.abs(theta) < 1e-7:
            w_err = np.zeros(3)
        else:
            sin_theta = np.sin(theta)
            axis = (1 / (2 * sin_theta)) * np.array([
                R_err[2, 1] - R_err[1, 2],
                R_err[0, 2] - R_err[2, 0],
                R_err[1, 0] - R_err[0, 1]
            ])
            w_err = theta * axis

        return np.hstack((p_err, w_err))

    def vex(self, S):
        """
        功能：从反对称矩阵中取出旋转轴
        输入：旋转矩阵np.dot(R0, R1.T)，从当前姿态(R1)到目标姿态(R0)的旋转微分。

        输出：np.array(3,1)，旋转轴xyz各轴分量

        """
        W = np.zeros((3, 1), dtype=np.float64)
        W[0, 0] = 0.5 * (S[2, 1] - S[1, 2])
        W[1, 0] = 0.5 * (S[0, 2] - S[2, 0])
        W[2, 0] = 0.5 * (S[1, 0] - S[0, 1])

        return W

    def tr2delta(self, T0, T1):
        """
        功能：计算T0-T1(位姿矩阵)的平移和旋转微分运动，基于基系
        输入：T0：目标位姿
             T1：当前位姿

        输出:np.array(6,1) (X、Y、Z、wx、wy、wz)

        """
        delta = np.zeros((6, 1))
        R0 = T0[:3, :3]
        R1 = T1[:3, :3]

        dR = self.vex(np.matmul(R0, R1.T))
        t = np.array([T0[0, 3] - T1[0, 3], T0[1, 3] - T1[1, 3], T0[2, 3] - T1[2, 3]])
        t = t.reshape(3, 1)

        for r in range(3):
            delta[r, 0] = t[r, 0]
            delta[r + 3, 0] = dR[r, 0]

        return delta

    def q_limits(self, q, q_ref):
        q = q.copy()
        ok = True

        for j in range(6):
            qn = (q[j] + np.pi) % (2 * np.pi) - np.pi
            vals = [
                qn + 2 * np.pi * k
                for k in range(-2, 3)
                if self.q_min[j] <= qn + 2 * np.pi * k <= self.q_max[j]
            ]

            if vals:
                q[j] = min(vals, key=lambda a: abs(a - q_ref[j]))
            else:
                q[j] = np.clip(q[j], self.q_min[j], self.q_max[j])
                ok = False

        return q, ok

    def lm_ik(self, target_pose, initial_q=None, max_iter=400, lam=0.1, rlimit=100, tol=None, pos_tol=0.5, rot_tol=np.deg2rad(4.0)):
        """
        lm 法
        """
        # 初始化变量
        success = False
        iterations = 0
        rejcount = 0

        # 初始化关节角
        if initial_q is None:
            q = np.random.uniform(self.q_min, self.q_max)
        else:
            q = np.array(initial_q, dtype=float).copy()

        t_target = self.pose2homography(target_pose)

        # 迭代求解
        for iters in range(max_iter):
            iterations = iters + 1

            _, t_start = self.fkine(q)
            # err = self.tr2delta(t_start, t_target)
            err = self._pose_error(t_start, t_target)
            err = err[:, np.newaxis]
            # 检查收敛
            if np.linalg.norm(err[:3]) < pos_tol and np.linalg.norm(err[3:]) < rot_tol:
                success = True
                break

            # 计算雅可比和更新步长
            J = self._vector_jacobian(q)
            JtJ = J.T @ J
            dq = -np.linalg.inv(JtJ + (lam + 1e-8) * np.eye(JtJ.shape[0])) @ J.T @ err
            q_new = q + dq.squeeze()

            # 计算新误差
            _, t_curr = self.fkine(q_new)
            err_new = self._pose_error(t_curr, t_target)
            err_new_norm = np.linalg.norm(err_new)
            err_norm = np.linalg.norm(err)

            # 自适应调整阻尼
            if err_new_norm < err_norm:
                q = q_new
                lam = lam / 2
                rejcount = 0
            else:
                lam = lam * 2
                rejcount += 1
                if rejcount > rlimit:
                    break
            # 角度约束到 [-pi, pi] 并限制范围
            # q = (q + np.pi) % (2 * np.pi) - np.pi
            # for i in range((len(q))):
            #     while q[i] > np.pi:
            #         q[i] -= 2 * np.pi
            #     while q[i] < -np.pi:
            #         q[i] += 2 * np.pi

            q = np.clip(q, self.q_min, self.q_max)
        # 最终验证
        _, t_final = self.fkine(q)
        final_error = self._pose_error(t_final, t_target)

        # 检查最终误差和角度限制
        if np.linalg.norm(final_error[:3]) < pos_tol and np.linalg.norm(final_error[3:]) < rot_tol:
            success = True

        # 返回结果
        return q, success

    def Gn_ik(
            self,
            target_pose,
            initial_q=None,
            use_weighted=True,
            max_iter=800,
            tol=None,
            pos_tol=0.5,
            rot_tol=np.deg2rad(4.0),
            mu0=0.03,
            rho_bound=1000000.0,
            bound_scale_deg=1.2,
            jump_schedule=((20.0, 0.00000010), (40.0, 0.00000050), (60.0, 0.00000200)),
    ):
        """
        方法: 罚函数
        成功率: 77.8%。
        """
        t_des = self.pose2homography(target_pose)
        q_min = self.q_min.copy()
        q_max = self.q_max.copy()

        if initial_q is not None:
            q_ref = np.array(initial_q, dtype=float).copy()
        else:
            q_ref = np.random.uniform(q_min, q_max)

        q_ref = np.clip(q_ref, q_min, q_max)
        I6 = np.eye(6, dtype=float)
        bound_scale_rad = np.deg2rad(bound_scale_deg)
        bound_scale = np.full(6, bound_scale_rad, dtype=float)

        def is_safe(q_value):
            return bool(np.all(q_value >= q_min) and np.all(q_value <= q_max))

        def fix_angle(q_value):
            q_new = q_value.copy()
            for j in range(6):
                q_base = (q_new[j] + np.pi) % (2.0 * np.pi) - np.pi
                candidates = []
                for k in range(-2, 3):
                    q_try = q_base + 2.0 * np.pi * k
                    if q_min[j] <= q_try <= q_max[j]:
                        candidates.append(q_try)
                if candidates:
                    q_new[j] = min(candidates, key=lambda item: abs(item - q_ref[j]))
                else:
                    q_new[j] = np.clip(q_new[j], q_min[j], q_max[j])
            return q_new

        def get_jump_diff(q_value):
            d = q_value - q_ref
            for j in range(6):
                while d[j] > np.pi:
                    d[j] -= 2.0 * np.pi
                while d[j] < -np.pi:
                    d[j] += 2.0 * np.pi
            return d

        def residual(q_value):
            _, t_curr = self.fkine(q_value)
            e = self._pose_error(t_curr, t_des)

            if use_weighted:
                values = [e[0] / 20.0, e[1] / 20.0, e[2] / 20.0, e[3], e[4], e[5]]
            else:
                values = [e[0], e[1], e[2], e[3], e[4], e[5]]

            lower = np.maximum(q_min - q_value, 0.0) / bound_scale
            upper = np.maximum(q_value - q_max, 0.0) / bound_scale
            bound_w = np.sqrt(rho_bound) * 1e-6
            values.extend(bound_w * lower)
            values.extend(bound_w * upper)

            d = get_jump_diff(q_value)
            for threshold_deg, rho_jump in jump_schedule:
                jump_over = np.maximum(np.abs(d) - np.deg2rad(threshold_deg), 0.0)
                values.extend(np.sqrt(rho_jump) * jump_over)

            return np.array(values, dtype=float)

        def residual_jacobian(q_value):
            J_pose = self._vector_jacobian(q_value)

            if use_weighted:
                pose_weight = np.array([1.0 / 20.0, 1.0 / 20.0, 1.0 / 20.0, 1.0, 1.0, 1.0])
            else:
                pose_weight = np.ones(6, dtype=float)

            rows = list((pose_weight[:, None] * J_pose).tolist())
            bound_w = np.sqrt(rho_bound) * 1e-6

            for j in range(6):
                row = np.zeros(6, dtype=float)
                if q_value[j] < q_min[j]:
                    row[j] = -bound_w / bound_scale[j]
                rows.append(row)

            for j in range(6):
                row = np.zeros(6, dtype=float)
                if q_value[j] > q_max[j]:
                    row[j] = bound_w / bound_scale[j]
                rows.append(row)

            d = get_jump_diff(q_value)
            d_abs = np.abs(d)

            for threshold_deg, rho_jump in jump_schedule:
                threshold_rad = np.deg2rad(threshold_deg)
                for j in range(6):
                    row = np.zeros(6, dtype=float)
                    if d_abs[j] > threshold_rad:
                        sign_value = 1.0 if d[j] >= 0.0 else -1.0
                        row[j] = np.sqrt(rho_jump) * sign_value
                    rows.append(row)

            return np.array(rows, dtype=float)

        q = q_ref.copy()
        mu = mu0

        for _ in range(max_iter):
            r_now = residual(q)
            _, t_curr = self.fkine(q)
            error = self._pose_error(t_curr, t_des)

            if np.linalg.norm(error[:3]) < pos_tol and np.linalg.norm(error[3:]) < rot_tol and is_safe(q):
                return q, True

            J = residual_jacobian(q)
            A = J.T @ J + mu * I6
            b = J.T @ r_now

            try:
                dq = -np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                dq = -np.linalg.lstsq(A, b, rcond=None)[0]

            if not np.all(np.isfinite(dq)):
                break

            max_dq = np.max(np.abs(dq))
            if max_dq > np.deg2rad(90.0):
                dq *= np.deg2rad(90.0) / max_dq

            current_cost = 0.5 * float(r_now @ r_now)
            accepted = False
            alpha = 1.0

            while alpha >= 1e-4:
                dq_try = alpha * dq
                q_trial = fix_angle(q + dq_try)
                r_trial = residual(q_trial)
                trial_cost = 0.5 * float(r_trial @ r_trial)

                if trial_cost < current_cost:
                    q = q_trial
                    mu = max(mu * 0.3, 1e-12)
                    accepted = True
                    break

                alpha *= 0.5

            if not accepted:
                mu = min(mu * 10.0, 1e12)

            if accepted and np.linalg.norm(alpha * dq) < 1e-10:
                break

        q = np.clip(q, q_min, q_max)
        return q, False

    def horizontal_line_sampling(self, x, z_height, y_period, steps, pattern_deg):
        """
        直线扫描
        x: 固定X(mm)
        z_height: 固定Z(mm)
        y_period: Y方向总长度(mm)，范围[-y_period/2, y_period/2]
        steps: 点数
        pattern_deg: 姿态俯仰角循环列表(度)，例如[-30, 0, 30, 0]
        return: 关节角(度) np.ndarray shape=(steps,6)
        """
        theta_star = np.deg2rad(np.array(Config.robot_original_joints, dtype=float))

        y_samples = np.linspace(-y_period / 2, y_period / 2, steps)
        x_samples = np.full_like(y_samples, x, dtype=float)
        z_samples = np.full_like(y_samples, z_height, dtype=float)

        thetas = []
        q = theta_star.copy()

        for i in range(steps):
            view_angle = np.deg2rad(pattern_deg[i % len(pattern_deg)])

            y_axis = np.array([0, 1, 0], dtype=np.float32)
            z_axis = np.array([-np.cos(view_angle), 0, np.sin(view_angle)], dtype=np.float32)
            z_axis /= np.linalg.norm(z_axis)
            x_axis = np.cross(y_axis, z_axis)
            x_axis /= np.linalg.norm(x_axis)

            R = np.vstack([x_axis, y_axis, z_axis]).T
            rpy = self._rotation2rpy(R)

            pose = np.array([x_samples[i], y_samples[i], z_samples[i], rpy[0], rpy[1], rpy[2]], dtype=float)

            q, success = self.LM_ik(pose, initial_q=q)
            if not success:
                raise RuntimeError(f"IK failed at step {i + 1}, pose={np.round(pose, 3).tolist()}")

            thetas.append(q.copy())

        thetas = np.rad2deg(np.array(thetas))
        thetas = np.round(thetas, 2)

        print("robot_line_joint = [")
        for row in thetas:
            print(f"    {[round(float(v), 2) for v in row]},")
        print("]")
        return thetas

    # —————————————————————————————————————————————————————————————————————————————————————————————
    # 以上是刘嘉梁代码，以下是曾文勇+黄旭熙代码
    # 最后维护日期：2026年4月7日
    # 维护者：黄旭熙
    # —————————————————————————————————————————————————————————————————————————————————————————————

    def sin_arc_length_sampling(self, x, z1, z2, y_period, steps, view_angle_range=np.pi / 4, draw_2d=False,
                                draw_3d=False):
        """
        功能：生成沿正弦曲线的机械臂扫描轨迹（等弧长采样）
             文勇：x=-350,z1=300,z2=700,steps=8
        输入：
            x: 轨迹固定的X坐标 (mm)
            z1: 正弦曲线Z方向起点 (mm)
            z2: 正弦曲线Z方向终点 (mm)
            y_period: 正弦曲线在Y方向的一个周期长度 (mm)
            steps: 轨迹点数量
            view_angle_range: 末端视角最大俯仰角 (rad，默认±45°)
            draw_2d: 是否绘制2D可视化图（默认False）
            draw_3d: 是否绘制3D交互式可视化图（默认False）
        输出：
            thetas: 每个轨迹点对应的机械臂关节角度 (deg)
        """
        # theta_STAR = np.array([6.3, 64.8, -120, -3.5, -37, -24.44])  # 文勇初始关节角，用于逆解迭代初值
        theta_STAR = np.array(Config.robot_original_joints)  # 旭熙初始关节角，用于逆解迭代初值
        theta_STAR = np.deg2rad(theta_STAR)  # 转为弧度制

        z_amplitude = (z2 - z1) / 2  # 正弦曲线振幅：Z方向总高度差的一半
        z_mid = (z1 + z2) / 2  # 正弦曲线Z方向中点（平均高度）
        y_start = -y_period / 2  # Y方向起点：周期中心点向左半周期
        y_end = y_period / 2  # Y方向终点：周期中心点向右半周期

        # 高密度采样
        y_dense = np.linspace(y_start, y_end, 10000)  # 高密度生成Y坐标（10000个点，保证弧长计算精度）
        # 正弦曲线公式：Z = 振幅*sin(2π*y/周期) + 偏移
        z_dense = z_amplitude * np.sin(2 * y_dense / y_period * np.pi) + z1 + z_amplitude

        # 计算累积弧长
        dy = y_dense[1] - y_dense[0]  # 相邻两个高密度点的Y步长（固定值）
        dz = np.diff(z_dense)  # 相邻点Z坐标差值
        arc_lengths = np.sqrt(dy ** 2 + dz ** 2)  # 每一小段的弧长（勾股定理）
        cumulative_length = np.concatenate(([0], np.cumsum(arc_lengths)))  # 累积弧长（从起点到当前点的总长度）
        total_length = cumulative_length[-1]  # 整条正弦曲线的总长度

        # 均匀弧长间隔
        target_lengths = np.linspace(0, total_length, steps)  # 生成均匀的目标弧长间隔（保证机械臂匀速运动）

        # 找到对应坐标
        y_samples = np.interp(target_lengths, cumulative_length, y_dense)  # 线性插值：从累积弧长映射回Y坐标
        z_samples = z_amplitude * np.sin(2 * y_samples / y_period * np.pi) + z1 + z_amplitude  # 根据采样后的Y计算Z坐标
        x_samples = x * np.ones_like(y_samples)  # X坐标固定，全部设为输入的x值
        positions = np.vstack([x_samples, y_samples, z_samples])  # 组合成 N×3 的位置矩阵 [x, y, z]
        positions = np.transpose(positions)  # 转置后形状：(steps, 3)

        # 计算观测视点的RPY角度
        orientations = np.zeros((steps, 3), dtype=np.float32)  # 存储rpy
        rotation_matrices_list = []  # 收集所有的旋转矩阵供3D可视化使用
        for i in range(steps):
            x = positions[i, 0]
            y = positions[i, 1]
            z = positions[i, 2]

            # y_axis = np.array([-0.06736549, 0.84714799, 0.52706943], dtype=np.float32)
            y_axis = np.array([0, 1, 0], dtype=np.float32)  # 末端固定Y轴：世界坐标系Y轴方向 (0,1,0)
            # 计算视点俯仰角：
            # z_offset 归一化到 (-1, 1) → 乘以最大视角 → 得到当前点的视角
            z_offset = (z - z_mid) / z_amplitude  # (-1, 1)
            view_angle = z_offset * view_angle_range
            # 相机Z轴方向：带俯仰角的观察方向
            # 负号表示朝向X负方向观察，符合相机安装朝向
            z_axis = np.array([-np.cos(view_angle), 0, np.sin(view_angle)], dtype=np.float32)
            z_axis /= np.linalg.norm(z_axis)  # 单位化Z轴
            # noinspection PyUnreachableCode
            x_axis = np.cross(y_axis, z_axis)  # 右手坐标系：X轴 = Y轴 × Z轴
            x_axis = x_axis / np.linalg.norm(x_axis)  # 单位化X轴

            R = np.vstack([x_axis, y_axis, z_axis])  # 构造旋转矩阵 R = [x_axis, y_axis, z_axis]
            R = np.transpose(R)

            rotation_matrices_list.append(R)  # 将算好的旋转矩阵存入列表
            rpy = self._rotation2rpy(R)
            orientations[i, 0] = rpy[0]
            orientations[i, 1] = rpy[1]
            orientations[i, 2] = rpy[2]

        pose = np.hstack([positions, orientations])  # 把拼接成完整位姿 (steps, 6)

        thetas = []

        for i in range(len(pose)):
            # sin_H = self.pose2homography(pose[i])  # 位姿 → 齐次变换矩阵
            if i == 0:
                theta, success = self.lm_ik(pose[i], theta_STAR)  # 第一点用初始theta_STAR迭代，后续用上一角度迭代
            else:
                theta, success = self.lm_ik(pose[i], theta)
            thetas.append(theta)
        thetas = np.rad2deg(thetas)  # 转角度
        thetas = np.round(thetas, 2)  # 保留2位小数

        # 自动绘图2D/3D
        if draw_2d:
            self.visualize_sin_traj_2d(x, y_dense, z_dense, y_samples, z_samples, z1, z2, z_mid, y_period, steps)
        if draw_3d:
            self.visualize_sin_traj_3d(x, y_dense, z_dense, x_samples, y_samples, z_samples, z1, z2, steps,
                                       rotation_matrices_list)
        return thetas

    @staticmethod
    def visualize_sin_traj_2d(x, y_dense, z_dense, y_samp, z_samp, z1, z2, z_mean, y_period, steps):
        """
       功能：2D可视化正弦曲线扫描轨迹（等弧长采样）
            绘制内容：完整正弦曲线 + 均匀采样点 + 关键参数标注
       输入：
           x: 固定X坐标
           y_dense: 高密度Y坐标
           z_dense: 高密度Z坐标
           y_samp: 采样点Y坐标
           z_samp: 采样点Z坐标
           z1: 轨迹最低高度
           z2: 轨迹最高高度
           z_mean: 轨迹中线高度
           y_period: Y方向周期长度
           steps: 采样点数量
       """

        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
        plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

        plt.figure(figsize=(10, 6))
        plt.plot(y_dense, z_dense, 'b-', linewidth=2, label="正弦轨迹")
        plt.scatter(y_samp, z_samp, color='red', s=50, zorder=5, label=f"等弧长采样点 ({steps}个)")

        for i, (y, z) in enumerate(zip(y_samp, z_samp)):
            plt.annotate(f"{i + 1}", (y, z), xytext=(0, -20), textcoords="offset points", fontsize=12, weight='bold')

        plt.axhline(z_mean, c='gray', ls='--', label=f"z_mean {z_mean}")
        plt.axhline(z1, c='green', ls=':', label=f"波谷z1 {z1}")
        plt.axhline(z2, c='orange', ls=':', label=f"波峰z2 {z2}")

        plt.text(y_dense[0], z_mean + 30, f"振幅={(z2 - z1) / 2}", fontsize=11)
        plt.text(0, z1 - 80, f"周期y_period {y_period}", ha='center', fontsize=11)
        plt.xlabel("Y/mm")
        plt.ylabel("Z/mm")
        plt.title(f"SIN曲线扫描轨迹 —— 固定 X={x} mm")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # 输出采样点坐标
        print("\n正弦轨迹采样点坐标")
        for i, (y, z) in enumerate(zip(y_samp, z_samp)):
            print(f"{i + 1:2d} -> Y = {y:6.2f}   Z = {z:6.2f}")

    @staticmethod
    def visualize_sin_traj_3d(x, y_dense, z_dense, x_samples, y_samples, z_samples, z1, z2, steps, rotation_matrices):
        """
        功能：3D交互式正弦扫描轨迹可视化
             绘制内容：3D轨迹曲线、采样点、末端坐标系三轴
        输入：
            x: 固定X坐标
            y_dense: 高密度轨迹Y坐标
            z_dense: 高密度轨迹Z坐标
            x_samples: 采样点X坐标
            y_samples: 采样点Y坐标
            z_samples: 采样点Z坐标
            z1: 轨迹最低点Z
            z2: 轨迹最高点Z
            steps: 采样点数量
            rotation_matrices: 每个点的旋转矩阵列表
        """

        # 解决中文显示
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        x_dense = np.full_like(y_dense, x)

        # 创建画布
        fig = plt.figure(figsize=(12, 9), num=f'SIN曲线扫描轨迹 —— 固定 X={x} mm')
        ax = fig.add_subplot(111, projection='3d')
        ax.set_box_aspect([1, 1, 1])
        ax.set_proj_type('ortho')  # 正交投影，真实显示角度

        # 画轨迹
        ax.plot(x_dense, y_dense, z_dense, color="#888888", linewidth=1.5, label="正弦轨迹")
        ax.scatter(x_samples, y_samples, z_samples, color="#FF4136", s=50, zorder=5, label=f"采样点 ({steps}个)")

        # 标注点编号
        for idx in range(steps):
            ax.text(x_samples[idx] + 2, y_samples[idx], z_samples[idx], f"{idx + 1}", fontsize=10, color='black',
                    weight='bold')

        # 画每个点的工具三轴
        axis_len = 5  # 坐标系箭头长度
        linewidth = 1.2  # 轴线宽度
        arrow_head = 0.3  # 箭头大小（3D下数值很小才合适）
        for i in range(steps):
            px, py, pz = x_samples[i], y_samples[i], z_samples[i]  # 获取当前采样点坐标
            R = rotation_matrices[i]  # 获取当前姿态旋转矩阵
            print(R)
            #  X轴（红色）
            ax.quiver(px, py, pz, R[0, 0], R[1, 0], R[2, 0],
                      length=axis_len,  # 箭头长度
                      color='red',
                      linewidth=linewidth,  # 线条粗细
                      arrow_length_ratio=arrow_head,  # 箭头大小比例
                      normalize=True,
                      label='X' if i == 0 else "")

            # Y轴（绿色）
            ax.quiver(px, py, pz, R[0, 1], R[1, 1], R[2, 1],
                      length=axis_len,
                      color='green',
                      linewidth=linewidth,
                      arrow_length_ratio=arrow_head,
                      normalize=True,
                      label='Y' if i == 0 else "")

            # Z轴（蓝色）
            ax.quiver(px, py, pz, R[0, 2], R[1, 2], R[2, 2],
                      length=axis_len,
                      color='blue',
                      linewidth=linewidth,
                      arrow_length_ratio=arrow_head,
                      normalize=True,
                      label='Z' if i == 0 else "")

        # 图注设置
        ax.set_xlabel('X/mm')
        ax.set_ylabel('Y/mm')
        ax.set_zlabel('Z/mm')
        ax.set_title(f'SIN曲线扫描轨迹 —— 固定 X={x} mm  波高={z1}-{z2}mm')
        ax.legend()
        print("\n采样点坐标")
        for i in range(steps):
            print(f"{i + 1:2d} | X={x_samples[i]:7.2f} Y={y_samples[i]:7.2f} Z={z_samples[i]:7.2f}")
        plt.ion()  # 开启交互模式
        plt.show(block=True)  # block=True 会阻塞，直到关闭窗口

    # 旋转矩阵转换为欧拉角
    @staticmethod
    def _rotation2rpy(R):
        """
        功能：旋转矩阵转换为RPY欧拉角（Roll-X, Pitch-Y, Yaw-Z）
        输入：
            R： np.array, (4,4)或(3,3) - 旋转矩阵/齐次变换矩阵
        输出：
            np.array([roll, pitch, yaw]), 弧度,(3,)
        """
        if R.shape == (4, 4):
            R = R[0:3, 0:3]

        pitch = np.arctan2(-R[2][0], np.sqrt(R[0][0] ** 2 + R[1][0] ** 2))

        cos_eps = 1e-10
        if np.abs(np.cos(pitch)) < cos_eps:  # 奇异位置：pitch = ±90°
            yaw = 0.0
            pitch_sign = np.sign(pitch)  # pitch > 0 时为 +1，< 0 时为 -1
            roll = np.arctan2(pitch_sign * R[0][1], R[1][1])
        else:  # 正常情况
            roll = np.arctan2(R[2][1], R[2][2])
            yaw = np.arctan2(R[1][0], R[0][0])

        return np.array([roll, pitch, yaw])

    # 欧拉角转换为旋转矩阵 (ZYX顺序)
    @staticmethod
    def _rpy2rotation(euler):
        """
        功能：欧拉角转换为旋转矩阵 (ZYX顺序)
        输入：
            列表， [rx, ry, rz], 弧度
        输出：
            旋转矩阵： np.array, (3,3)
        """
        # 计算绕X轴的旋转矩阵
        rx = np.array([[1, 0, 0],
                       [0, np.cos(euler[0]), -np.sin(euler[0])],
                       [0, np.sin(euler[0]), np.cos(euler[0])]])

        # 计算绕Y轴的旋转矩阵
        ry = np.array([[np.cos(euler[1]), 0, np.sin(euler[1])],
                       [0, 1, 0],
                       [-np.sin(euler[1]), 0, np.cos(euler[1])]])

        # 计算绕Z轴的旋转矩阵
        rz = np.array([[np.cos(euler[2]), -np.sin(euler[2]), 0],
                       [np.sin(euler[2]), np.cos(euler[2]), 0],
                       [0, 0, 1]])

        # 旋转矩阵相乘：Z-Y-X顺序（先rx，再ry，最后rz）
        return rz @ ry @ rx  # 等价于rz*ry*rx（矩阵乘法顺序不能乱）

    # 旋转矩阵R与位置矩阵P合成4*4齐次矩阵
    @staticmethod
    def _get_homography_from_R_P(pp_R, P):
        """旋转矩阵R与位置矩阵P合成4*4齐次矩阵"""
        pp_H = np.eye(4)  # 初始化4x4单位矩阵（对角线为1，其余为0）
        pp_H[:3, :3] = pp_R  # 填充旋转矩阵部分 (左上角3x3)
        # pp_H[:3, 3] = P[:, 0] # 填充平移向量部分 (右侧3x1列)
        pp_H[:3, 3] = np.array(P).flatten()

        return pp_H  # 返回齐次矩阵

    # 从齐次矩阵分解旋转和平移
    @staticmethod
    def _get_R_P_from_homography(T):
        """从 4×4 齐次位姿矩阵（T_cam2base）中分解出旋转矩阵（R）和平移向量（t）"""
        # 提取旋转矩阵 (左上角 3x3)
        R = T[:3, :3]
        # 提取平移向量 (右侧 3x1 列)
        P = T[:3, 3]
        # 返回旋转矩阵和平移向量
        return R, P

    # 输入4*4位姿矩阵，得到位姿pose
    def homography2pose(self, H):
        """
        功能：输入4*4位姿矩阵，得到位姿pose
        输入：
            H： np.array, 4*4
        输出：
            pose: np.array [x,y,z,rx,ry,rz]  (6,)
        """
        rx, ry, rz = self._rotation2rpy(H)

        return np.array([H[0, 3], H[1, 3], H[2, 3], rx, ry, rz])

    # 将位姿（x,y,z,r,p,y）转换为4×4齐次位姿矩阵（H）
    def pose2homography(self, pose):
        """
            功能：将位姿信息（x,y,z,rx,ry,rz）直接转换为4×4齐次位姿矩阵（H）
            当检测到目标的位姿（x,y,z,rx,ry,rz）时，用该函数转换为齐次矩阵，再通过矩阵运算转换到机器人基坐标系，规划抓取路径
                pose:(x,y,z,rx,ry,rz)转换为4*4位姿矩阵
            输入：
                pose:(x,y,z,rx,ry,rz)

            输出：
                T_cam2base: 4*4位姿矩阵，np.array
        """
        x, y, z, rx, ry, rz = pose  # 解包位姿信息
        euler = np.array([rx, ry, rz])  # 构造欧拉角数组
        R = self._rpy2rotation(euler)  # 调将欧拉角转为旋转矩阵
        t = np.array([x, y, z]).reshape(3, 1)  # 构造平移向量（3×1列向量）
        H = np.eye(4)  # 初始化4×4单位矩阵
        H[:3, :3] = R  # 填充旋转矩阵部分
        H[:3, 3] = t[:, 0]  # 填充平移向量部分（取t的第一列）

        return H  # 返回4×4齐次矩阵

    @staticmethod
    def _rotation_to_axis_angle(R):
        """
        计算旋转矩阵的旋转轴和角度
        旋转矩阵的迹与旋转角度的关系：tr(R) = 1 + 2cosθ
        输入：
            R: 3x3 旋转矩阵
        输出：
            axis: 旋转轴（单位向量）[x,y,z]
            theta_f: 旋转角度（弧度）
        """
        # 计算旋转角度：θ = arccos[(tr(R)-1)/2]，tr(R)是旋转矩阵的迹（对角线元素和）
        theta_f = np.arccos((np.trace(R) - 1) / 2)
        if theta_f < 1e-6:  # 角度接近0（无旋转），返回默认旋转轴[1,0,0]和角度0
            return np.array([1, 0, 0]), 0
        else:
            # 计算旋转轴：利用旋转矩阵反对称部分的归一化结果
            # 公式推导：R - R^T_cam2base = 2sinθ * [axis]_×（反对称矩阵），提取轴向量
            axis = np.array([
                R[2, 1] - R[1, 2],  # 反对称矩阵(0,1)元素
                R[0, 2] - R[2, 0],  # 反对称矩阵(1,2)元素
                R[1, 0] - R[0, 1]]) / (2 * np.sin(theta_f))  # 除以2sinθ归一化
            axis = axis / np.linalg.norm(axis)  # 确保旋转轴是单位向量
            return axis, theta_f

    @staticmethod
    def _axis_angle_to_rotation(axis, theta):
        """
        轴角转旋转矩阵（罗德里格斯公式）
        核心原理：将轴角表示转为旋转矩阵
        输入：
            axis: 旋转轴（单位向量）[x,y,z]
            theta: 旋转角度（弧度）
        输出：
            R: 3x3 旋转矩阵
        """
        if theta < 1e-6:  # 角度接近0（无旋转），返回单位矩阵
            return np.eye(3)
        # a, b, c = axis[0], axis[1], axis[2]  # 提取旋转轴的分量
        a, b, c = axis  # 提取旋转轴的三个分量
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        # 罗德里格斯公式展开：R = cosθ*I + sinθ*[axis]_× + (1-cosθ)*axis*axis^T_cam2base
        return np.array([
            [a * a * (1 - cos_theta) + cos_theta, a * b * (1 - cos_theta) - c * sin_theta,
             a * c * (1 - cos_theta) + b * sin_theta],
            [a * b * (1 - cos_theta) + c * sin_theta, b * b * (1 - cos_theta) + cos_theta,
             b * c * (1 - cos_theta) - a * sin_theta],
            [a * c * (1 - cos_theta) - b * sin_theta, b * c * (1 - cos_theta) + a * sin_theta,
             c * c * (1 - cos_theta) + cos_theta]
        ])

    def linear_interpolation(self, start_pose, target_pose, steps=4):
        """
        功能：规划一段直线路径,用于直线插补
        输入：
            start_pose: 初始位姿[x,y,z,r,p,y](单位: mm,rad)
            target_pose: 目标位姿[x,y,z,r,p,y](单位: mm,rad)
            steps: 插补步数(步数越多越平滑，但计算越慢)
        输出：
            tra_poses: 末端位姿度列表
        """
        # 计算起点和终点的齐次矩阵
        T_start = self.pose2homography(start_pose)
        T_end = self.pose2homography(target_pose)  # 目标位姿转齐次矩阵：[x,y,z,r,p,y] → 4x4齐次矩阵

        # 分离旋转矩阵和位置向量,准备插补
        R_start = T_start[:3, :3]  # 起点旋转矩阵（3x3）
        R_end = T_end[:3, :3]  # 终点旋转矩阵（3x3）
        P_start = T_start[:3, 3]  # 起点位置向量 [x,y,z] (mm)
        P_end = np.array(target_pose[:3])  # 终点位置向量（mm）

        # 计算旋转差异：R_diff = R_start^T_cam2base * R_end（从起点旋转到终点旋转的增量）
        R_diff = R_start.T @ R_end
        # 将旋转差异转为轴角表示（旋转轴+旋转角度），用于姿态插补
        axis, angle = self._rotation_to_axis_angle(R_diff)

        traj_poses = []  # 初始化关节轨迹列表（存储每一步的关节角度）

        # 生成中间插补点并逐点逆解为关节角
        # 遍历插补步数（1~steps），生成0.1,0.2,...,1.0的进度值
        for i in range(1, steps + 1):
            s = i / steps  # 进度 0.1, 0.2 ... 1.0

            # 位置线性插补：笛卡尔空间直线，保证末端走直线
            P_curr = (1 - s) * P_start + s * P_end

            # 将起点旋转矩阵，沿旋转轴旋转 angle*s 角度，得到当前姿态
            R_curr = R_start @ self._axis_angle_to_rotation(axis, angle * s)

            pp_H = self._get_homography_from_R_P(R_curr, P_curr)
            pose_curr = self.homography2pose(pp_H)
            traj_poses.append(pose_curr)

        return traj_poses

    # ===============================================================================================
    # 调用逆解生成角度
    def grid_joint(self, x, z1, z2, y_period, rows=2, cols=3, pitch_deg=0.0, draw_2d=False, draw_3d=False):
        """
        功能：生成网格法机械臂扫描轨迹（可配置行列数、边界、俯仰角），采用蛇形路线
        输入：
            x: 轨迹固定的X坐标 (mm)
            z1: 扫描下边界 (mm)
            z2: 扫描上边界 (mm)
            y_period: Y方向总跨度 (mm)，范围为 [-y_period/2, y_period/2]
            rows: 网格行数（默认为2）
            cols: 网格列数（默认为3）
            pitch_deg: 末端固定俯仰角 (deg，默认0°)
            draw_2d: 是否绘制2D可视化图（默认False）
            draw_3d: 是否绘制3D交互式可视化图（默认False）
        输出：
            thetas: 每个网格点对应的机械臂关节角度 (deg)，格式为 (N, 6) 的 numpy 数组
        """
        # 读取初始关节角用于逆解迭代初值
        theta_STAR = np.array(Config.robot_original_joints)
        theta_STAR = np.deg2rad(theta_STAR)  # 转为弧度

        # 1. 生成网格坐标点
        # Z轴方向（从上边界 z2 向下边界 z1 递减）
        z_vals = np.linspace(z2, z1, rows) if rows > 1 else np.array([z2])
        # Y轴方向（从左边界 -y_period/2 向右边界 y_period/2 递增）
        y_vals = np.linspace(-y_period / 2, y_period / 2, cols) if cols > 1 else np.array([0.0])

        positions = []
        for r in range(rows):
            z_curr = z_vals[r]
            # 奇数行和偶数行交替改变Y轴遍历方向（蛇形路线）
            if r % 2 == 0:
                y_row = y_vals  # 从左往右
            else:
                y_row = y_vals[::-1]  # 从右往左

            for y_curr in y_row:
                positions.append([x, y_curr, z_curr])

        positions = np.array(positions)
        steps = len(positions)

        # 2. 计算观测视点的RPY角与旋转矩阵
        orientations = np.zeros((steps, 3), dtype=np.float64)
        rotation_matrices_list = []

        view_angle = np.deg2rad(pitch_deg)
        y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)  # 末端固定Y轴方向 (0,1,0)

        # 相机Z轴方向：带俯仰角的观察方向
        z_axis = np.array([-np.cos(view_angle), 0.0, np.sin(view_angle)], dtype=np.float64)
        z_axis /= np.linalg.norm(z_axis)  # 单位化Z轴

        x_axis = np.cross(y_axis, z_axis)  # 右手坐标系：X轴 = Y轴 × Z轴
        x_axis = x_axis / np.linalg.norm(x_axis)  # 单位化X轴

        R = np.column_stack((x_axis, y_axis, z_axis))
        rpy = self._rotation2rpy(R)

        for i in range(steps):
            rotation_matrices_list.append(R)
            orientations[i] = rpy

        # 拼接成完整的 Cartesian 位姿 (steps, 6)
        pose = np.hstack([positions, orientations])

        # 3. 逐点求解逆运动学 (IK)
        thetas = []
        theta = theta_STAR.copy()
        grid_jump_limit_deg = 125.0
        main_q_min = self.q_min.copy()
        main_q_max = self.q_max.copy()
        self.q_min = np.radians([-178, -105, -178, -178, -178, -360])
        self.q_max = np.radians([178, 100, 145, 178, 178, 360])

        try:
            from scipy.optimize import least_squares

            for i in range(steps):
                theta_ref = theta_STAR if i == 0 else theta
                if i == 0:
                    theta, success = self.Gn_ik(pose[i], initial_q=theta_STAR)  # 第一点用 theta_STAR
                else:
                    theta, success = self.Gn_ik(pose[i], initial_q=theta)  # 后续点用上一时刻的逆解作为初值

                # Gn_ik 的通用姿态容差较宽，可能选中腕部顺/逆时针偏转的近似解。
                # 网格扫描点在本函数内再以完整旋转矩阵做有界精修，不依赖腕部分支的符号。
                target_transform = np.eye(4, dtype=np.float64)
                target_transform[:3, :3] = R
                target_transform[:3, 3] = positions[i]
                jump_limit_rad = np.deg2rad(grid_jump_limit_deg)
                lower = np.maximum(self.q_min, theta_ref - jump_limit_rad)
                upper = np.minimum(self.q_max, theta_ref + jump_limit_rad)
                theta_start = np.clip(np.asarray(theta, dtype=np.float64), lower, upper)

                def grid_pose_residual(q_value):
                    _, current_transform = self.fkine(q_value)
                    position_error = (
                        np.asarray(current_transform[:3, 3], dtype=np.float64)
                        - target_transform[:3, 3]
                    ) / 100.0
                    rotation_error = (
                        np.asarray(current_transform[:3, :3], dtype=np.float64)
                        - target_transform[:3, :3]
                    ).reshape(-1)
                    return np.concatenate((position_error, rotation_error))

                refined = least_squares(
                    grid_pose_residual,
                    theta_start,
                    bounds=(lower, upper),
                    method="trf",
                    max_nfev=1000,
                    xtol=1e-14,
                    ftol=1e-14,
                    gtol=1e-14,
                )
                theta = np.asarray(refined.x, dtype=np.float64)

                _, solved_transform = self.fkine(theta)
                solved_position = np.asarray(solved_transform[:3, 3], dtype=np.float64)
                solved_rotation = np.asarray(solved_transform[:3, :3], dtype=np.float64)
                position_error_mm = float(np.linalg.norm(solved_position - positions[i]))
                y_axis_error = float(np.linalg.norm(solved_rotation[:, 1] - y_axis))
                rotation_error = float(np.linalg.norm(solved_rotation - R, ord="fro"))
                within_limits = bool(
                    np.all(theta >= self.q_min - 1e-12)
                    and np.all(theta <= self.q_max + 1e-12)
                )

                if (
                    not refined.success
                    or position_error_mm > 0.5
                    or y_axis_error > 1e-9
                    or rotation_error > 1e-8
                    or not within_limits
                ):
                    raise RuntimeError(
                        f"IK 求解失败：第 {i + 1} 个网格点不满足严格姿态约束，"
                        f"位置误差={position_error_mm:.9f}mm，"
                        f"Y轴误差={y_axis_error:.3e}，旋转矩阵误差={rotation_error:.3e}"
                    )

                thetas.append(theta.copy())
        finally:
            # 网格生成的宽限位只在本函数内部生效，主流程仍使用实例原有限位。
            self.q_min = main_q_min
            self.q_max = main_q_max

        # 保留 IK 的完整精度。关节角舍入到两位小数会破坏末端 Y 轴的平行约束。
        thetas_rad = np.asarray(thetas, dtype=np.float64)

        # 用实际将返回的关节角复核末端姿态。
        # 要求末端 +Y 轴与基座 +Y 轴同向，不合格的解不会被返回。
        max_y_axis_error = 1e-9
        for i, theta_rad in enumerate(thetas_rad):
            _, solved_transform = self.fkine(theta_rad)
            solved_y_axis = np.asarray(solved_transform[:3, 1], dtype=np.float64)
            solved_y_axis_norm = float(np.linalg.norm(solved_y_axis))
            if not np.isfinite(solved_y_axis_norm) or solved_y_axis_norm <= 0.0:
                raise RuntimeError(f"姿态校验失败：第 {i + 1} 个网格点的末端Y轴无效")
            solved_y_axis /= solved_y_axis_norm
            y_axis_error = float(np.linalg.norm(solved_y_axis - y_axis))
            if y_axis_error > max_y_axis_error:
                y_axis_angle_deg = np.rad2deg(2.0 * np.arcsin(np.clip(
                    y_axis_error / 2.0, 0.0, 1.0
                )))
                raise RuntimeError(
                    f"姿态校验失败：第 {i + 1} 个网格点的末端Y轴方向"
                    f"与基座+Y轴夹角为 {y_axis_angle_deg:.12f}°，"
                    f"超过误差限制 {max_y_axis_error:.1e}"
                )

        thetas = np.rad2deg(thetas_rad)

        # 打印生成的角度列表（方便复制粘贴到配置文件）
        print(f"\nrobot_grid_joint ({rows}x{cols}) = [")
        for row in thetas:
            values = ", ".join(f"{float(v):.2f}" for v in row)
            print(f"    [{values}],")
        print("]")

        # 4. 可视化
        if draw_2d:
            self.visualize_grid_traj_2d(x, positions, z1, z2, rows, cols)
        if draw_3d:
            self.visualize_grid_traj_3d(x, positions, steps, rotation_matrices_list)

        return thetas

    # 可视化直线插补
    def visualize_ik_pose_path_3d(self, pose_records, target_pose, window_title):
        if not Config.enable_ik_pose_visualization or not pose_records:
            return

        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans SC", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        axis_length = float(Config.ik_pose_axis_length_mm)
        fig = plt.figure(window_title)
        fig.clf()
        ax = fig.add_subplot(111, projection="3d")

        positions = np.array([record["pose"][:3] for record in pose_records], dtype=float)
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], color="black", linewidth=1.5, label="逆解成功路径")

        axis_colors = ("r", "g", "b")
        axis_labels = ("X", "Y", "Z")
        for index, record in enumerate(pose_records):
            pose = np.array(record["pose"], dtype=float)
            position = pose[:3]
            rotation = self.pose2homography(pose)[:3, :3]
            role = record["role"]

            if role == "采摘目标点":
                marker = "*"
                color = "red"
                size = 120
            elif role == "预采摘点":
                marker = "s"
                color = "orange"
                size = 55
            else:
                marker = "o"
                color = "black"
                size = 35

            ax.scatter(position[0], position[1], position[2], marker=marker, color=color, s=size)
            ax.text(position[0], position[1], position[2], f"{index + 1}:{role}", fontsize=8)

            for axis_index, axis_color in enumerate(axis_colors):
                direction = rotation[:, axis_index] * axis_length
                end = position + direction
                ax.quiver(
                    position[0], position[1], position[2],
                    direction[0], direction[1], direction[2],
                    color=axis_color,
                    length=1.0,
                    normalize=False,
                    linewidth=1.2,
                )
                ax.text(end[0], end[1], end[2], axis_labels[axis_index], color=axis_color, fontsize=8)

        target_position = np.array(target_pose[:3], dtype=float)
        ax.scatter(
            target_position[0],
            target_position[1],
            target_position[2],
            marker="*",
            color="magenta",
            edgecolors="black",
            s=220,
            label="locator目标pose",
        )
        ax.text(
            target_position[0],
            target_position[1],
            target_position[2],
            "locator目标pose",
            color="magenta",
            fontsize=10,
        )

        mins = positions.min(axis=0)
        maxs = positions.max(axis=0)
        mins = np.minimum(mins, target_position)
        maxs = np.maximum(maxs, target_position)
        center = (mins + maxs) / 2.0
        span = max(float(np.max(maxs - mins)), axis_length * 3.0, 1.0)
        half = span / 2.0 + axis_length
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
        ax.set_zlim(center[2] - half, center[2] + half)
        ax.set_xlabel("X / mm")
        ax.set_ylabel("Y / mm")
        ax.set_zlabel("Z / mm")
        ax.set_title(window_title)
        ax.legend(loc="best")
        ax.view_init(elev=25, azim=-55)
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.001)
    # ========================================================================================


if __name__ == "__main__":
    robot = RobotKinematics()
    #
    """测试逆解成功率"""
    # joints = np.random.rand(1000, 6) * (robot.q_max - robot.q_min) + robot.q_min
    # success = 0
    # success_ = 0
    # for idx, joint in enumerate(joints):
    #     _, target_t = robot.fkine(list(joint))
    #     target_pose = robot.homography2pose(target_t)
    #     print("循环次数", idx)
    #     _, successs = robot.Gn_ik(target_pose, initial_q=None)
    #     if successs:
    #         success += 1
    # print("success rate:", success / joints.shape[0])

    """计算sin曲线+2D/3D可视化"""
    # thetas = robot.sin_arc_length_sampling(x_fixed=-350,  # 固定X坐标
    #                                        z_low=300,  # 波谷Z
    #                                        z_high=700,  # 波峰Z
    #                                        y_period=100,  # Y方向周期
    #                                        point_num=8,  # 采样点数
    #                                        draw_2d=True,
    #                                        draw_3d=True)

    """直线扫描角度生成"""
    # thetas = robot.horizontal_line_sampling(
    #     x=-430,  # X进来不要过低
    #     z_height=410,
    #     y_period=700,
    #     steps=7,
    #     pattern_deg=[-2, 2, -2, 2]
    # )

    """网格法扫描角度生成"""
    # controller = RobotController(Config, speed=Config.slow_speed)
    # thetas = robot.grid_joint(
    #     x=-360,          # 固定 X 坐标 (mm)
    #     z1=350,          # 扫描下边界
    #     z2=650,          # 扫描上边界
    #     y_period=680,    # Y方向扫描范围 (mm)
    #     rows=2,          # 行数
    #     cols=3,          # 列数
    #     pitch_deg=-30.0,   # 相机俯仰角，默认为 0 度
    # )
    # try:
    #     controller.move_to_original(speed=Config.fast_speed)
    #     time.sleep(1)
    #
    #     for index_scan, current_joints_deg in enumerate(thetas):
    #         current_joints_rad = np.deg2rad(current_joints_deg).tolist()
    #
    #         ret = controller.move_robot_joint(
    #             current_joints_rad,
    #             speed=Config.fast_speed
    #         )
    #
    #         if ret != 0:
    #             print(f"第 {index_scan + 1} 个扫描点运动失败，错误码: {ret}")
    #             break
    #
    #         time.sleep(1)
    #
    # finally:
    #     controller.disconnect_all_devices()

    """测试末端执行器稳定性"""
    controller = RobotController(Config, speed=Config.normal_speed)

    # controller.move_to_original(speed=Config.fast_speed)
    #
    # time.sleep(1.5)
    #
    # controller.move_to_basket(speed=Config.normal_speed)
    #
    # time.sleep(1.5)

    while True:
        controller.ee_open()
        time.sleep(3)

        controller.ee_close()
        time.sleep(10)
        controller.ee_open()

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import Config

# 尝试导入 ultralytics 的 YOLO
try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None


PEDICEL_ALIASES = {
    "stem",
    "pedicel",
    "fruit_pedicel",
    "fruit_peduncle",
    "peduncle",
    "fruit_stem",
    "tomato_stem",
    "tomato",
    "item",
    "class_0",
}

MAIN_STEM_ALIASES = {
    "main_stem",
    "mainstem",
    "main-stem",
    "main stem",
}


def _canonical_class_name(name: str) -> str:
    raw = str(name).strip().lower()
    key = raw.replace(" ", "_").replace("-", "_")
    if raw in PEDICEL_ALIASES or key in PEDICEL_ALIASES:
        return "stem"
    if raw in MAIN_STEM_ALIASES or key in MAIN_STEM_ALIASES:
        return "main_stem"
    return key


class Locator:
    """
    Locator：目标实例分割 + 深度点云定位 + 三维直线拟合。
    当前支持两类目标：
        1. stem：圣女果果梗，输出 P1 / P2
        2. main_stem：主干分段，按段输出 M1 / M2、M3 / M4 ...
    """

    def __init__(self, yolo_dir, fx, fy, cx, cy, cam2end_R, cam2end_T, robot, distortion=None):
        # 相机内参
        self.yolo_dir = yolo_dir
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.cam2end_R = cam2end_R
        self.cam2end_T = cam2end_T
        self.robot = robot
        self.ref_vec = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if distortion is None:
            distortion = getattr(Config, "distortion", np.zeros(5, dtype=np.float64))
        self.distortion = np.asarray(distortion, dtype=np.float64).reshape(-1)
        self.camera_matrix = np.array([
            [float(self.fx), 0.0, float(self.cx)],
            [0.0, float(self.fy), float(self.cy)],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        # YOLO 模型与检测参数
        self.model = None
        self.conf = Config.conf
        self.iou = Config.iou
        self.inference_imgsz = int(getattr(Config, "inference_imgsz", 1280))
        self.inference_max_det = int(getattr(Config, "inference_max_det", 100))
        self.inference_retina_masks = bool(getattr(Config, "inference_retina_masks", True))
        self.mask_threshold = float(getattr(Config, "mask_threshold", 0.5))
        self.depth_scale = Config.depth_scale
        self.mask_erode_kernel_size = int(Config.mask_erode_kernel_size)
        self.mask_erode_iterations = int(Config.mask_erode_iterations)
        self.depth_cluster_mad_scale = float(Config.depth_cluster_mad_scale)
        self.depth_cluster_min_window_mm = float(Config.depth_cluster_min_window_mm)
        self.depth_cluster_min_keep_ratio = float(Config.depth_cluster_min_keep_ratio)
        self.depth_quality_min_points = int(Config.depth_quality_min_points)
        self.depth_quality_min_valid_ratio = float(Config.depth_quality_min_valid_ratio)
        self.depth_quality_min_cluster_ratio = float(Config.depth_quality_min_cluster_ratio)
        self.depth_quality_max_mad_mm = float(Config.depth_quality_max_mad_mm)
        self.depth_quality_max_span_mm = float(Config.depth_quality_max_span_mm)
        self.depth_multiframe_outlier_mm = float(Config.depth_multiframe_outlier_mm)
        self.depth_multiframe_fuse = bool(getattr(Config, "depth_multiframe_fuse", True))
        self.depth_multiframe_min_consistent = max(
            2, int(getattr(Config, "depth_multiframe_min_consistent", 3))
        )
        self.allow_degraded_depth_pose = bool(getattr(Config, "allow_degraded_depth_pose", False))
        self.require_endpoint_exact_depth = bool(getattr(Config, "require_endpoint_exact_depth", True))
        self.endpoint_depth_patch_radius_px = int(getattr(Config, "endpoint_depth_patch_radius_px", 8))
        self.endpoint_min_depth_samples = int(getattr(Config, "endpoint_min_depth_samples", 3))
        self.endpoint_max_depth_std_mm = float(getattr(Config, "endpoint_max_depth_std_mm", 15.0))
        self.endpoint_max_z_gap_mm = float(getattr(Config, "endpoint_max_z_gap_mm", 80.0))
        self.endpoint_max_distance_mm = float(getattr(Config, "endpoint_max_distance_mm", 200.0))
        self.endpoint_min_depth_mm = float(getattr(Config, "endpoint_min_depth_mm", 100.0))
        self.scissor_max_opening_mm = float(Config.scissor_max_opening_mm)
        self.main_stem_safe_clearance_mm = float(Config.main_stem_safe_clearance_mm)
        self.main_stem_offset_activation_distance_mm = float(Config.main_stem_offset_activation_distance_mm)
        self.main_stem_max_offset_mm = float(Config.main_stem_max_offset_mm)
        self.enable_main_stem_dynamic_offset = bool(Config.enable_main_stem_dynamic_offset)
        self.stem_duplicate_center_distance_mm = float(Config.stem_duplicate_center_distance_mm)
        self.stem_duplicate_line_distance_mm = float(Config.stem_duplicate_line_distance_mm)

        # pcl.py 优化方法的参数。只借用点云过滤和直线拟合思路，
        # 相机内参、深度尺度和手眼标定仍完全使用 locator/Config 中的数据。
        self.point_filter_neighbors = max(1, int(getattr(Config, "point_filter_neighbors", 21)))
        self.point_filter_std_ratio = max(0.0, float(getattr(Config, "point_filter_std_ratio", 1.0)))
        self.line_ransac_max_iterations = max(1, int(getattr(Config, "line_ransac_max_iterations", 2000)))
        self.line_ransac_threshold_mm = max(1e-6, float(getattr(Config, "line_ransac_threshold_mm", 15.0)))
        self.line_ransac_min_inlier_ratio = float(np.clip(
            getattr(Config, "line_ransac_min_inlier_ratio", 0.5), 0.0, 1.0
        ))
        endpoint_percentiles = getattr(Config, "line_endpoint_percentiles", (5.0, 95.0))
        self.line_endpoint_percentiles = tuple(float(v) for v in endpoint_percentiles)
        if (
            len(self.line_endpoint_percentiles) != 2
            or not 0.0 <= self.line_endpoint_percentiles[0] < self.line_endpoint_percentiles[1] <= 100.0
        ):
            raise ValueError("line_endpoint_percentiles 必须是0到100之间递增的两个数")

        # 类别名约定：训练 dataset.yaml 中应为 {0: stem, 1: main_stem}
        self.class_names = {0: "stem"}
        self.target_class_ids = {0}

        # 缓存上一次结果，便于可视化
        self._last_instances: List[Dict[str, Any]] = []
        self._last_depth: Optional[np.ndarray] = None
        self._last_localizations: List[Dict[str, Any]] = []
        self._last_pc: Optional[np.ndarray] = None
        self._last_line: Optional[np.ndarray] = None
        self._last_T_base2cam = None
        self._last_pick_targets: List[Dict[str, Any]] = []
        self._last_point_filter_info: Optional[Dict[str, Any]] = None
        self._last_rejected_localizations: List[Dict[str, Any]] = []

        self._load_model()

    def _load_model(self):
        """在指定路径或目录下寻找并加载 YOLO 权重。"""
        if YOLO is None:
            raise ImportError("未安装 ultralytics，请先执行: pip install ultralytics")

        model_path = Path(str(self.yolo_dir))
        if model_path.is_file():
            candidates = [model_path]
        else:
            candidates = [
                model_path / "best.pt",
                model_path / "weights" / "best.pt",
                model_path / "train" / "weights" / "best.pt",
            ]

        for p in candidates:
            if p.exists():
                self.model = YOLO(str(p))
                # 尽量读取模型内保存的类别名
                names = getattr(self.model, "names", None)
                if isinstance(names, dict) and names:
                    self.class_names = {int(k): str(v) for k, v in names.items()}
                elif isinstance(names, list) and names:
                    self.class_names = {i: str(v) for i, v in enumerate(names)}
                self._refresh_target_class_ids()
                expected_classes = {
                    _canonical_class_name(name)
                    for name in getattr(Config, "yolo_expected_classes", ("stem",))
                }
                actual_classes = {
                    _canonical_class_name(name)
                    for name in self.class_names.values()
                }
                missing_classes = expected_classes - actual_classes
                if missing_classes:
                    self.model = None
                    missing_text = ", ".join(sorted(missing_classes))
                    actual_text = ", ".join(sorted(actual_classes)) or "无"
                    raise ValueError(
                        f"YOLO模型类别不兼容，缺少: {missing_text}；模型实际类别: {actual_text}。"
                        "正式定位必须使用stem/main_stem两类实例分割权重。"
                    )
                return

        raise FileNotFoundError(
            f"未找到 YOLO 权重，请把 best.pt 放到 {self.yolo_dir}，或直接把 yolo_dir 传成 best.pt 路径"
        )

    def _refresh_target_class_ids(self):
        target_ids = {
            int(cid)
            for cid, name in self.class_names.items()
            if _canonical_class_name(name) == "stem"
        }
        if not target_ids and len(self.class_names) == 1:
            only_id = int(next(iter(self.class_names.keys())))
            target_ids = {only_id}
            self.class_names[only_id] = "stem"
        else:
            for cid in list(target_ids):
                self.class_names[int(cid)] = "stem"
        self.target_class_ids = target_ids or {0}

    def _get_class_name(self, cls_id: int) -> str:
        """根据类别 id 获取类别名。"""
        return str(self.class_names.get(int(cls_id), f"class_{int(cls_id)}"))

    def _line_color(self, class_name: str) -> Tuple[int, int, int]:
        """不同类别使用不同颜色，OpenCV 使用 BGR。"""
        if class_name == "stem":
            return (0, 255, 255)      # 黄色：果梗
        if class_name == "main_stem":
            return (255, 0, 255)      # 紫色：主干
        return (255, 255, 255)

    @staticmethod
    def _is_main_stem_class(localization_or_instance: Dict[str, Any]) -> bool:
        cls_id = localization_or_instance.get("cls_id", None)
        name = str(localization_or_instance.get("class_name", "")).strip().lower()
        name_norm = _canonical_class_name(name)
        if cls_id is not None and int(cls_id) == 1:
            return True
        return name_norm == "main_stem"

    @classmethod
    def _is_stem_class(cls, localization_or_instance: Dict[str, Any]) -> bool:
        if cls._is_main_stem_class(localization_or_instance):
            return False
        cls_id = localization_or_instance.get("cls_id", None)
        name = str(localization_or_instance.get("class_name", "")).strip().lower()
        name_norm = _canonical_class_name(name)
        if cls_id is not None and int(cls_id) == 0:
            return True
        return name_norm == "stem" or name_norm == ""

    def run_detection(self, bgr):
        """
        运行 YOLO 实例分割。
        返回实例列表，每个实例包含 mask、bbox、contour、score、cls_id、class_name。
        """
        if bgr is None:
            raise ValueError("输入图像为空")

        results = self.model.predict(
            source=bgr,
            verbose=False,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.inference_imgsz,
            max_det=self.inference_max_det,
            retina_masks=self.inference_retina_masks,
        )
        if not results:
            self._last_instances = []
            return []

        result = results[0]
        h, w = bgr.shape[:2]
        instances: List[Dict[str, Any]] = []

        if result.masks is None or result.boxes is None:
            self._last_instances = []
            return []

        mask_data = result.masks.data.detach().cpu().numpy()
        boxes_xyxy = result.boxes.xyxy.detach().cpu().numpy()
        scores = result.boxes.conf.detach().cpu().numpy()
        cls_ids = result.boxes.cls.detach().cpu().numpy().astype(np.int32)

        for i in range(len(mask_data)):
            mask = mask_data[i]
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            mask = mask > self.mask_threshold
            if int(mask.sum()) < 10:
                continue

            cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            contour = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.int32)

            cls_id = int(cls_ids[i])
            class_name = self._get_class_name(cls_id)
            if cls_id not in self.target_class_ids and _canonical_class_name(class_name) != "stem":
                continue
            class_name = "stem"
            instances.append({
                "mask": mask,
                "contour": contour,
                "bbox": boxes_xyxy[i].astype(np.float32),
                "score": float(scores[i]),
                "cls_id": cls_id,
                "class_name": class_name,
            })

        self._last_instances = instances
        return instances

    def run_localization(self, instances, depth, ee_pose):
        """
        对每个检测实例执行三维定位。
        stem 输出 P1/P2；main_stem 按分段顺序输出 M1/M2、M3/M4 ...。
        """
        if depth is None:
            raise ValueError("depth 不能为空")

        depth_frames = self._normalize_depth_frames(depth)
        self._last_depth_frames = depth_frames
        self._last_depth = depth_frames[-1]
        pps: List[Dict[str, Any]] = []
        self._last_rejected_localizations = []
        last_T_base2cam = None

        for instance in instances:
            pp, T_base2cam = self._localize_instance_from_depth_frames(instance, depth_frames, ee_pose)
            if T_base2cam is not None:
                last_T_base2cam = T_base2cam
            if not self._is_valid_localization(pp):
                continue

            # 保留该直线对应的实例类别信息
            pp["cls_id"] = int(instance.get("cls_id", -1))
            pp["class_name"] = str(instance.get("class_name", self._get_class_name(pp["cls_id"])))
            pp["bbox"] = instance.get("bbox")
            pp["score"] = float(instance.get("score", 0.0))
            pp["instance"] = instance
            pps.append(pp)

        pps = self._deduplicate_stem_localizations(pps)

        # 统一给端点命名：stem=P1/P2，main_stem=M1/M2、M3/M4...
        self._assign_endpoint_names(pps)

        self._last_localizations = pps
        self._last_T_base2cam = last_T_base2cam
        self._last_pick_targets = self._build_pick_targets(pps)
        return self._last_pick_targets

    def _assign_endpoint_names(self, localizations: List[Dict[str, Any]]):
        """按照类别给拟合线段端点命名。"""
        # 果梗：每个 stem 都保留 P1/P2 字段
        stem_locs = [loc for loc in localizations if self._is_stem_class(loc)]
        for idx, loc in enumerate(stem_locs, start=1):
            loc["segment_index"] = idx
            loc["point1_name"] = "P1"
            loc["point2_name"] = "P2"
            loc["P1"] = loc["p1"]
            loc["P2"] = loc["p2"]
            loc[loc["point1_name"]] = loc["p1"]
            loc[loc["point2_name"]] = loc["p2"]

        # 主干：先按图像中 bbox 的 y、x 顺序排序，再依次命名 M1/M2、M3/M4...
        main_locs = [loc for loc in localizations if self._is_main_stem_class(loc)]

        def _bbox_key(loc):
            bbox = loc.get("bbox")
            if bbox is None:
                c = loc.get("center", np.zeros(3, dtype=np.float32))
                return float(c[1]), float(c[0])
            x1, y1, x2, y2 = np.asarray(bbox, dtype=np.float32).tolist()
            return float((y1 + y2) * 0.5), float((x1 + x2) * 0.5)

        main_locs.sort(key=_bbox_key)
        for idx, loc in enumerate(main_locs, start=1):
            loc["segment_index"] = idx
            loc["point1_name"] = f"M{2 * idx - 1}"
            loc["point2_name"] = f"M{2 * idx}"
            loc[loc["point1_name"]] = loc["p1"]
            loc[loc["point2_name"]] = loc["p2"]

        # 未知类别保留 Cx 命名，避免可视化或打印时报错
        other_locs = [
            loc for loc in localizations
            if not self._is_stem_class(loc) and not self._is_main_stem_class(loc)
        ]
        for idx, loc in enumerate(other_locs, start=1):
            loc["segment_index"] = idx
            loc["point1_name"] = f"C{idx}_1"
            loc["point2_name"] = f"C{idx}_2"
            loc[loc["point1_name"]] = loc["p1"]
            loc[loc["point2_name"]] = loc["p2"]

    @staticmethod
    def _is_valid_localization(localization):
        if not localization:
            return False

        line = np.asarray(localization.get("line"), dtype=np.float32)
        center = np.asarray(localization.get("center"), dtype=np.float32)
        main_axis = np.asarray(localization.get("main_axis"), dtype=np.float32)

        if line.shape != (2, 3) or center.shape != (3,) or main_axis.shape != (3,):
            return False
        if not np.all(np.isfinite(line)) or not np.all(np.isfinite(center)) or not np.all(np.isfinite(main_axis)):
            return False
        if np.linalg.norm(line[1] - line[0]) < 1e-6:
            return False
        if np.linalg.norm(main_axis) < 1e-6:
            return False
        return True

    @staticmethod
    def _normalize(vec, eps=1e-9):
        vec = np.asarray(vec, dtype=np.float64).reshape(3)
        norm = np.linalg.norm(vec)
        if not np.isfinite(norm) or norm < eps:
            return None
        return vec / norm

    @classmethod
    def _line_direction(cls, localization):
        line = np.asarray(localization.get("line"), dtype=np.float64)
        if line.shape == (2, 3):
            direction = cls._normalize(line[1] - line[0])
            if direction is not None:
                return direction
        p1 = np.asarray(localization.get("p1", np.zeros(3)), dtype=np.float64)
        p2 = np.asarray(localization.get("p2", np.zeros(3)), dtype=np.float64)
        direction = cls._normalize(p2 - p1)
        if direction is not None:
            return direction
        return cls._normalize(localization.get("main_axis", np.array([1.0, 0.0, 0.0])))

    # 原采摘点算法保留如下，需要恢复中点采摘时可取消注释，并注释下方同名函数。
    @staticmethod
    def _line_midpoint(localization):
        """返回 P1/P2 的中点，作为统一的采摘基准点。"""
        p1 = np.asarray(localization.get("p1"), dtype=np.float64)
        p2 = np.asarray(localization.get("p2"), dtype=np.float64)
        if p1.shape != (3,) or p2.shape != (3,):
            line = np.asarray(localization.get("line"), dtype=np.float64)
            if line.shape != (2, 3):
                return None
            p1, p2 = line[0], line[1]
        if not np.all(np.isfinite(p1)) or not np.all(np.isfinite(p2)):
            return None
        return (p1 + p2) * 0.5



    #
    #
    # @staticmethod
    # def _line_midpoint(localization):
    #     """
    #     返回 P1/P2 中相对地面更低的端点，作为采摘基准点。
    #
    #     当前机器人基坐标系 Z 轴竖直向上，因此 Z 值较小的端点离地面更近。
    #     函数名保持不变，以兼容正式版原有调用链。
    #     """
    #     p1 = np.asarray(localization.get("p1"), dtype=np.float64)
    #     p2 = np.asarray(localization.get("p2"), dtype=np.float64)
    #     if p1.shape != (3,) or p2.shape != (3,):
    #         line = np.asarray(localization.get("line"), dtype=np.float64)
    #         if line.shape != (2, 3):
    #             return None
    #         p1, p2 = line[0], line[1]
    #     if not np.all(np.isfinite(p1)) or not np.all(np.isfinite(p2)):
    #         return None
    #     return (p1 if p1[2] <= p2[2] else p2).copy()

    @staticmethod
    def _nearest_point_on_segment(point, seg_a, seg_b):
        point = np.asarray(point, dtype=np.float64).reshape(3)
        seg_a = np.asarray(seg_a, dtype=np.float64).reshape(3)
        seg_b = np.asarray(seg_b, dtype=np.float64).reshape(3)
        ab = seg_b - seg_a
        denom = float(np.dot(ab, ab))
        if denom < 1e-9:
            return seg_a.copy()
        t = float(np.dot(point - seg_a, ab) / denom)
        t = np.clip(t, 0.0, 1.0)
        return seg_a + t * ab

    def _nearest_main_stem(self, stem_info, main_stems, center=None):
        if center is None:
            center = stem_info.get("center")
        center = np.asarray(center, dtype=np.float64).reshape(3)

        nearest_info = None
        nearest_point = None
        nearest_distance = np.inf
        for main_info in main_stems:
            if not self._is_valid_localization(main_info):
                continue
            line = np.asarray(main_info.get("line"), dtype=np.float64)
            point = self._nearest_point_on_segment(center, line[0], line[1])
            distance = float(np.linalg.norm(center - point))
            if distance < nearest_distance:
                nearest_info = main_info
                nearest_point = point
                nearest_distance = distance

        return nearest_info, nearest_point, nearest_distance

    def _deduplicate_stem_localizations(self, localizations: List[Dict[str, Any]]):
        deduped: List[Dict[str, Any]] = []

        for loc in localizations:
            if not self._is_stem_class(loc):
                deduped.append(loc)
                continue

            duplicate_index = None
            for idx, kept in enumerate(deduped):
                if not self._is_stem_class(kept):
                    continue
                if self._is_duplicate_stem_by_distance(loc, kept):
                    duplicate_index = idx
                    break

            if duplicate_index is None:
                deduped.append(loc)
                continue

            if self._stem_duplicate_keep_key(loc) > self._stem_duplicate_keep_key(deduped[duplicate_index]):
                deduped[duplicate_index] = loc

        return deduped

    def _is_duplicate_stem_by_distance(self, stem_a, stem_b):
        center_distance = self._stem_center_distance(stem_a, stem_b)
        if center_distance is not None and center_distance <= self.stem_duplicate_center_distance_mm:
            return True

        line_distance = self._stem_line_distance(stem_a, stem_b)
        if line_distance is None:
            return False

        # 线段距离很近但中心相隔过远时，通常不是同一个采摘点，避免误合并相邻果梗。
        center_limit = self.stem_duplicate_center_distance_mm * 2.0
        center_is_near = center_distance is None or center_distance <= center_limit
        return center_is_near and line_distance <= self.stem_duplicate_line_distance_mm

    @staticmethod
    def _stem_duplicate_keep_key(stem_info):
        score = float(stem_info.get("score", 0.0))
        pc = stem_info.get("pc")
        pc_count = len(pc) if pc is not None else 0
        length = Locator._line_length(stem_info) or 0.0
        return score, pc_count, length

    @staticmethod
    def _stem_center_distance(stem_a, stem_b):
        center_a = np.asarray(stem_a.get("center"), dtype=np.float64)
        center_b = np.asarray(stem_b.get("center"), dtype=np.float64)
        if center_a.shape != (3,) or center_b.shape != (3,):
            return None
        if not np.all(np.isfinite(center_a)) or not np.all(np.isfinite(center_b)):
            return None
        return float(np.linalg.norm(center_a - center_b))

    @staticmethod
    def _stem_line_distance(stem_a, stem_b):
        line_a = np.asarray(stem_a.get("line"), dtype=np.float64)
        line_b = np.asarray(stem_b.get("line"), dtype=np.float64)
        if line_a.shape != (2, 3) or line_b.shape != (2, 3):
            return None
        if not np.all(np.isfinite(line_a)) or not np.all(np.isfinite(line_b)):
            return None

        distances = [
            Locator._distance_point_to_segment(line_a[0], line_b[0], line_b[1]),
            Locator._distance_point_to_segment(line_a[1], line_b[0], line_b[1]),
            Locator._distance_point_to_segment(line_b[0], line_a[0], line_a[1]),
            Locator._distance_point_to_segment(line_b[1], line_a[0], line_a[1]),
        ]
        return float(min(distances))

    def _build_pick_targets(self, localizations: List[Dict[str, Any]]):
        main_stems = [loc for loc in localizations if self._is_main_stem_class(loc)]
        pick_targets = []

        for loc in localizations:
            if not self._is_stem_class(loc):
                continue

            # 在对外输出前同步采摘基准：选择基坐标系Z更小、相对地面更低的端点。
            pick_point = self._line_midpoint(loc)
            if pick_point is None:
                continue
            p1 = np.asarray(loc.get("p1"), dtype=np.float64)
            picked_p1 = p1.shape == (3,) and np.allclose(pick_point, p1)
            loc["center"] = pick_point.astype(np.float32)
            loc["pick_point"] = pick_point.astype(np.float32)
            loc["pick_point_name"] = loc.get("point1_name" if picked_p1 else "point2_name", "P1" if picked_p1 else "P2")
            loc["pick_point_source"] = "p1p2_midpoint"
            loc["pick_point_base_z_mm"] = float(pick_point[2])

            main_info, nearest_point, distance = self._nearest_main_stem(
                loc,
                main_stems,
                center=pick_point,
            )
            loc["main_stem_info"] = main_info
            loc["main_stem_nearest_point"] = nearest_point
            loc["main_stem_distance_mm"] = distance if np.isfinite(distance) else None
            pick_targets.append(loc)

        return pick_targets

    def _normalize_depth_frames(self, depth):
        if isinstance(depth, (list, tuple)):
            depth_frames = [np.asarray(item) for item in depth if item is not None]
        else:
            depth_array = np.asarray(depth)
            if depth_array.ndim == 3 and depth_array.shape[-1] not in (1, 3):
                depth_frames = [depth_array[i] for i in range(depth_array.shape[0])]
            else:
                depth_frames = [depth_array]

        depth_frames = [frame for frame in depth_frames if frame is not None and frame.size > 0]
        if not depth_frames:
            raise ValueError("depth 不能没有有效帧")
        return depth_frames

    def _localize_instance_from_depth_frames(self, instance, depth_frames, ee_pose):
        high_quality_candidates: List[Dict[str, Any]] = []
        degraded_candidates: List[Dict[str, Any]] = []
        last_T_base2cam = None

        for frame_index, depth_frame in enumerate(depth_frames):
            self._last_depth = depth_frame
            pc_cam, depth_quality = self._instance_to_pc(
                instance,
                depth_frame,
                return_quality=True,
                enforce_quality=False,
            )
            if pc_cam is None:
                pc_cam = np.zeros((0, 3), dtype=np.float32)

            pc_base, T_base2cam = self._cam_to_robot(ee_pose, pc_cam)
            last_T_base2cam = T_base2cam

            # 先用 pcl.py 的 K 近邻统计方法剔除孤立噪点，再做 RANSAC 直线拟合。
            pc_filtered = self._pc_filter(pc_base) if len(pc_base) else np.zeros((0, 3), dtype=np.float32)
            pp = self._bbox_diagonal_localization(instance, depth_frame, T_base2cam, pc_filtered)
            if not self._is_valid_localization(pp):
                if pp:
                    self._last_rejected_localizations.append(pp)
                continue

            pp["depth_quality"] = depth_quality
            pp["depth_frame_index"] = frame_index
            pp["_T_base2cam"] = T_base2cam
            if bool(depth_quality.get("accepted", False)) and bool(pp.get("depth_reliable", False)):
                high_quality_candidates.append(pp)
            else:
                if "reject_reason" not in pp:
                    pp["reject_reason"] = (
                        "mask_depth_quality_failed"
                        if not bool(depth_quality.get("accepted", False))
                        else self._depth_reject_reason(pp)
                    )
                self._last_rejected_localizations.append(pp)
                degraded_candidates.append(pp)

        selected = self._select_stable_localization_candidate(high_quality_candidates)
        if selected is not None:
            selected["depth_pose_degraded"] = False
            selected["depth_pose_source"] = "high_quality"
        elif self.allow_degraded_depth_pose:
            selected = self._select_stable_localization_candidate(
                high_quality_candidates + degraded_candidates,
                allow_degraded=True,
            )
            if selected is not None:
                selected["depth_pose_degraded"] = True
                selected["depth_pose_source"] = "degraded_depth"

        if selected is None:
            return None, last_T_base2cam

        selected_T_base2cam = selected.pop("_T_base2cam", last_T_base2cam)
        return selected, selected_T_base2cam

    def _select_stable_localization_candidate(self, candidates, allow_degraded=False):
        if not candidates:
            return None

        if len(candidates) == 1:
            if not allow_degraded and self.depth_multiframe_min_consistent > 1:
                return None
            selected = dict(candidates[0])
            selected["depth_multiframe_info"] = {
                "candidate_count": 1,
                "consistent_count": 1,
                "selected_index": int(selected.get("depth_frame_index", 0)),
                "center_deviation_mm": 0.0,
                "fused": False,
            }
            return selected

        centers = np.asarray([item["center"] for item in candidates], dtype=np.float64)
        finite = np.all(np.isfinite(centers), axis=1)
        if not np.any(finite):
            return None

        finite_indices = np.where(finite)[0]
        finite_centers = centers[finite]
        median_center = np.median(finite_centers, axis=0)
        deviations = np.linalg.norm(finite_centers - median_center[None, :], axis=1)
        keep = deviations <= self.depth_multiframe_outlier_mm

        degraded_selection = False
        if int(np.count_nonzero(keep)) < self.depth_multiframe_min_consistent:
            if not allow_degraded:
                return None
            degraded_selection = True
            quality_scores = []
            for local_index, candidate_index in enumerate(finite_indices):
                quality = candidates[int(candidate_index)].get("depth_quality", {})
                z_mad = quality.get("z_mad_mm")
                z_span = quality.get("z_span_mm")
                cluster_ratio = float(quality.get("cluster_ratio", 0.0))
                cluster_count = float(quality.get("cluster_count", 0.0))
                score = float(deviations[local_index])
                if z_mad is not None:
                    score += 0.5 * float(z_mad)
                if z_span is not None:
                    score += 0.1 * float(z_span)
                score -= 30.0 * cluster_ratio
                score -= 0.01 * cluster_count
                quality_scores.append(score)
            best_local_index = int(np.argmin(np.asarray(quality_scores, dtype=np.float64)))
        else:
            kept_local_indices = np.where(keep)[0]
            best_local_index = int(kept_local_indices[np.argmin(deviations[kept_local_indices])])

        best_index = int(finite_indices[best_local_index])
        selected = dict(candidates[best_index])
        consistent_candidate_indices = [
            int(finite_indices[local_index])
            for local_index in np.where(keep)[0]
        ]

        fused = False
        endpoint_median_deviation_mm = None
        if (
            self.depth_multiframe_fuse
            and not degraded_selection
            and len(consistent_candidate_indices) >= self.depth_multiframe_min_consistent
        ):
            fused_candidate, endpoint_median_deviation_mm = self._fuse_localization_candidates(
                candidates,
                consistent_candidate_indices,
                selected,
            )
            if fused_candidate is not None:
                selected = fused_candidate
                fused = True

        selected["depth_multiframe_info"] = {
            "candidate_count": int(len(candidates)),
            "consistent_count": int(np.count_nonzero(keep)),
            "selected_index": int(selected.get("depth_frame_index", best_index)),
            "center_deviation_mm": float(deviations[best_local_index]),
            "median_center": median_center.astype(np.float32),
            "degraded_selection": bool(degraded_selection),
            "fused": bool(fused),
            "fused_frame_indices": [
                int(candidates[index].get("depth_frame_index", index))
                for index in consistent_candidate_indices
            ] if fused else [],
            "endpoint_median_deviation_mm": endpoint_median_deviation_mm,
        }
        return selected

    def _fuse_localization_candidates(self, candidates, candidate_indices, reference):
        """
        将稳定帧的端点方向对齐后逐坐标取中位数。

        这样既保留RANSAC/PCA得到的线方向，又能抑制RealSense单帧深度跳动；
        pc、质量信息和相机变换沿用最接近中心中位数的参考帧。
        """
        reference_direction = self._line_direction(reference)
        if reference_direction is None:
            return None, None

        aligned_p1 = []
        aligned_p2 = []
        for index in candidate_indices:
            candidate = candidates[int(index)]
            p1 = np.asarray(candidate.get("p1"), dtype=np.float64)
            p2 = np.asarray(candidate.get("p2"), dtype=np.float64)
            if p1.shape != (3,) or p2.shape != (3,):
                continue
            if not np.all(np.isfinite(p1)) or not np.all(np.isfinite(p2)):
                continue

            direction = self._normalize(p2 - p1)
            if direction is None:
                continue
            if float(np.dot(direction, reference_direction)) < 0.0:
                p1, p2 = p2, p1
            aligned_p1.append(p1)
            aligned_p2.append(p2)

        if len(aligned_p1) < self.depth_multiframe_min_consistent:
            return None, None

        p1_array = np.asarray(aligned_p1, dtype=np.float64)
        p2_array = np.asarray(aligned_p2, dtype=np.float64)
        fused_p1 = np.median(p1_array, axis=0)
        fused_p2 = np.median(p2_array, axis=0)
        fused_direction = self._normalize(fused_p2 - fused_p1)
        if fused_direction is None:
            return None, None
        if float(np.dot(fused_direction, reference_direction)) < 0.0:
            fused_p1, fused_p2 = fused_p2, fused_p1
            fused_direction = -fused_direction

        fused = dict(reference)
        fused_line = np.stack([fused_p1, fused_p2], axis=0).astype(np.float32)
        fused["line"] = fused_line
        fused["p1"] = fused_line[0].copy()
        fused["p2"] = fused_line[1].copy()
        fused["center"] = ((fused_p1 + fused_p2) * 0.5).astype(np.float32)
        fused["main_axis"] = fused_direction.astype(np.float32)
        fused["line_fit_method"] = f"{reference.get('line_fit_method', 'line')}_multiframe_median"
        fused["multiframe_fused"] = True

        endpoint_deviations = 0.5 * (
            np.linalg.norm(p1_array - fused_p1[None, :], axis=1)
            + np.linalg.norm(p2_array - fused_p2[None, :], axis=1)
        )
        median_deviation = float(np.median(endpoint_deviations))
        return fused, median_deviation

    def format_depth_pose_marker(self, tomato_info, scan_index=None, target_index=None):
        quality = tomato_info.get("depth_quality", {}) or {}
        degraded = bool(tomato_info.get("depth_pose_degraded", False))
        accepted = bool(quality.get("accepted", False))
        center = np.asarray(tomato_info.get("center", np.zeros(3)), dtype=np.float64).reshape(3)
        scan_text = "NA" if scan_index is None else str(int(scan_index) + 1)
        target_text = "NA" if target_index is None else str(int(target_index) + 1)

        def _fmt(value, digits=3):
            if value is None:
                return "NA"
            try:
                value = float(value)
            except (TypeError, ValueError):
                return "NA"
            if not np.isfinite(value):
                return "NA"
            return f"{value:.{digits}f}"

        pose_text = "降级" if degraded else "正常"
        quality_text = "通过" if accepted else "降级"
        multiframe_info = tomato_info.get("depth_multiframe_info", {}) or {}
        fusion_text = "融合" if bool(multiframe_info.get("fused", False)) else "单帧择优"
        pick_point_name = str(tomato_info.get("pick_point_name", "P?"))
        valid_percent = (
            "NA" if quality.get("valid_ratio") is None
            else f"{float(quality.get('valid_ratio')) * 100.0:.0f}%"
        )
        endpoint_std_mm = tomato_info.get("endpoint_depth_std_mm")
        endpoint_z_gap_mm = tomato_info.get("endpoint_z_gap_mm")
        endpoint_length_mm = tomato_info.get("endpoint_distance_mm")
        cluster_percent = (
            "NA" if quality.get("cluster_ratio") is None
            else f"{float(quality.get('cluster_ratio')) * 100.0:.0f}%"
        )

        return (
            "[深度姿态] "
            f"扫描={scan_text} "
            f"目标={target_text} "
            f"姿态={pose_text} "
            f"质量={quality_text} "
            f"采摘点={pick_point_name} "
            f"多帧={fusion_text} "
            f"深度={_fmt(center[2], 1)}mm "
            f"有效={valid_percent} "
            f"主簇={cluster_percent} "
            f"抖动={_fmt(quality.get('z_mad_mm'), 1)}mm "
            f"endpoint_std={_fmt(endpoint_std_mm, 1)}mm "
            f"endpoint_z_gap={_fmt(endpoint_z_gap_mm, 1)}mm "
            f"p1p2_len={_fmt(endpoint_length_mm, 1)}mm"
        )

    def _valid_depth_mm(self, values: np.ndarray) -> np.ndarray:
        z = self._depth_raw_to_mm(values)
        valid = (
            np.isfinite(z)
            & (z >= self.endpoint_min_depth_mm)
            & (z > 0)
            & (z < float(Config.depth_threshold))
        )
        return z[valid].astype(np.float32)

    def _depth_stats_at_uv(self, depth, u: float, v: float, mask: Optional[np.ndarray]) -> Dict[str, Any]:
        depth = np.asarray(depth)
        if depth.ndim == 3:
            depth = depth[..., 0]

        h, w = depth.shape[:2]
        c = int(round(np.clip(u, 0, w - 1)))
        r = int(round(np.clip(v, 0, h - 1)))

        exact_values = self._valid_depth_mm(np.asarray([depth[r, c]], dtype=np.float32))
        exact_valid = bool(exact_values.size > 0)
        exact_z = float(exact_values[0]) if exact_valid else 0.0

        patch_values = exact_values
        best_radius = 0
        if mask is not None and mask.shape[:2] == depth.shape[:2]:
            radius = max(1, int(self.endpoint_depth_patch_radius_px))
            for search_radius in (radius, radius * 2, radius * 3):
                r0, r1 = max(0, r - search_radius), min(h, r + search_radius + 1)
                c0, c1 = max(0, c - search_radius), min(w, c + search_radius + 1)
                patch = depth[r0:r1, c0:c1]
                mask_patch = mask[r0:r1, c0:c1].astype(bool)
                values = self._valid_depth_mm(patch[mask_patch])
                if values.size:
                    patch_values = values
                    best_radius = int(search_radius)
                if values.size >= self.endpoint_min_depth_samples:
                    break

        if patch_values.size == 0:
            return {
                "z_mm": 0.0,
                "exact_z_mm": 0.0,
                "median_z_mm": 0.0,
                "std_mm": np.inf,
                "count": 0,
                "radius_px": best_radius,
                "exact_valid": False,
            }

        median_z = float(np.median(patch_values))
        z_mm = exact_z if exact_valid else median_z
        return {
            "z_mm": float(z_mm),
            "exact_z_mm": float(exact_z),
            "median_z_mm": float(median_z),
            "std_mm": float(np.std(patch_values)),
            "count": int(patch_values.size),
            "radius_px": int(best_radius),
            "exact_valid": exact_valid,
        }

    def _mask_axis_uv(self, instance) -> Optional[np.ndarray]:
        contour = instance.get("contour")
        if contour is None or len(contour) < 2:
            return None
        pts = np.asarray(contour, dtype=np.float32).reshape(-1, 2)
        centered = pts - np.mean(pts, axis=0, keepdims=True)
        if len(centered) < 2:
            return None
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        axis = vh[0].astype(np.float32)
        norm = float(np.linalg.norm(axis))
        if norm <= 1e-6:
            return None
        return axis / norm

    def _diagonal_mask_endpoints(self, diagonal: np.ndarray, mask: Optional[np.ndarray], depth) -> Tuple[np.ndarray, bool]:
        if mask is None:
            return diagonal.astype(np.float32), False

        depth = np.asarray(depth)
        if depth.ndim == 3:
            depth = depth[..., 0]
        mask = mask.astype(bool)
        h, w = mask.shape[:2]
        p0 = diagonal[0].astype(np.float32)
        p1 = diagonal[1].astype(np.float32)
        length = int(np.ceil(max(abs(float(p1[0] - p0[0])), abs(float(p1[1] - p0[1]))))) + 1
        if length < 2:
            return diagonal.astype(np.float32), False

        xs = np.linspace(float(p0[0]), float(p1[0]), length, dtype=np.float32)
        ys = np.linspace(float(p0[1]), float(p1[1]), length, dtype=np.float32)
        cols = np.clip(np.rint(xs).astype(np.int32), 0, w - 1)
        rows = np.clip(np.rint(ys).astype(np.int32), 0, h - 1)
        on_mask = mask[rows, cols]

        if depth.shape[:2] == mask.shape[:2]:
            z = self._depth_raw_to_mm(depth[rows, cols].astype(np.float32))
            depth_valid = (
                np.isfinite(z)
                & (z >= self.endpoint_min_depth_mm)
                & (z > 0)
                & (z < float(Config.depth_threshold))
            )
            on_mask = on_mask & depth_valid

        idx = np.flatnonzero(on_mask)
        if idx.size >= 2:
            uv = np.array(
                [[xs[int(idx[0])], ys[int(idx[0])]], [xs[int(idx[-1])], ys[int(idx[-1])]]],
                dtype=np.float32,
            )
            if np.linalg.norm(uv[1] - uv[0]) > 1.0:
                return uv, True
        return diagonal.astype(np.float32), False

    def _choose_bbox_diagonal_uv(self, instance, depth) -> Tuple[np.ndarray, bool]:
        depth = np.asarray(depth)
        if depth.ndim == 3:
            depth = depth[..., 0]
        h, w = depth.shape[:2]
        x1, y1, x2, y2 = np.asarray(instance["bbox"], dtype=np.float32).tolist()
        x1, x2 = sorted([float(np.clip(x1, 0, w - 1)), float(np.clip(x2, 0, w - 1))])
        y1, y2 = sorted([float(np.clip(y1, 0, h - 1)), float(np.clip(y2, 0, h - 1))])

        candidates = [
            np.array([[x1, y1], [x2, y2]], dtype=np.float32),
            np.array([[x1, y2], [x2, y1]], dtype=np.float32),
        ]
        axis = self._mask_axis_uv(instance)
        if axis is not None:
            def _alignment(diagonal):
                d = diagonal[1] - diagonal[0]
                d = d / (np.linalg.norm(d) + 1e-12)
                return abs(float(np.dot(d, axis)))

            diagonal = max(candidates, key=_alignment)
        else:
            diagonal = candidates[0]

        uv, on_mask = self._diagonal_mask_endpoints(diagonal, instance.get("mask"), depth)
        if uv[0, 1] > uv[1, 1] or (abs(float(uv[0, 1] - uv[1, 1])) < 1e-3 and uv[0, 0] > uv[1, 0]):
            uv = uv[[1, 0]]
        return uv.astype(np.float32), bool(on_mask)

    def _uv_to_camera_point_mm(self, u: float, v: float, z_mm: float) -> np.ndarray:
        if z_mm <= 0 or not np.isfinite(z_mm):
            return np.zeros(3, dtype=np.float32)
        rows = np.asarray([float(v)], dtype=np.float32)
        cols = np.asarray([float(u)], dtype=np.float32)
        z = np.asarray([float(z_mm)], dtype=np.float32)
        return self._pixels_to_camera_points(rows, cols, z)[0].astype(np.float32)

    def _transform_camera_points(self, T_base2cam, pc_cam: np.ndarray) -> np.ndarray:
        pc_cam = np.asarray(pc_cam, dtype=np.float32).reshape(-1, 3)
        if len(pc_cam) == 0:
            return np.zeros((0, 3), dtype=np.float32)
        ones_col = np.ones((pc_cam.shape[0], 1), dtype=np.float32)
        pc_homo = np.hstack([pc_cam, ones_col])
        pc_base_homo = (T_base2cam @ pc_homo.T).T
        return pc_base_homo[:, :3].astype(np.float32)

    def _depth_reject_reason(self, loc: Dict[str, Any]) -> str:
        if not loc.get("diagonal_mask_endpoint_valid", False):
            return "diagonal_mask_endpoint_invalid"
        if not loc.get("depth_valid", False):
            return "invalid_endpoint_depth"
        if (
            self.require_endpoint_exact_depth
            and not (loc.get("p1_exact_depth_valid", False) and loc.get("p2_exact_depth_valid", False))
        ):
            return "inexact_endpoint_depth"
        if int(loc.get("min_endpoint_depth_samples", 0)) < self.endpoint_min_depth_samples:
            return "too_few_endpoint_depth_samples"
        if float(loc.get("endpoint_depth_std_mm", np.inf)) > self.endpoint_max_depth_std_mm:
            return "unstable_endpoint_depth"
        if float(loc.get("endpoint_z_gap_mm", np.inf)) > self.endpoint_max_z_gap_mm:
            return "endpoint_depth_jump"
        if float(loc.get("endpoint_distance_mm", np.inf)) > self.endpoint_max_distance_mm:
            return "p1p2_distance_too_long"
        return "unreliable_endpoint_depth"

    def _bbox_diagonal_localization(self, instance, depth, T_base2cam, pc_base):
        uv, on_mask_diagonal = self._choose_bbox_diagonal_uv(instance, depth)
        mask = instance.get("mask")

        p1_depth = self._depth_stats_at_uv(depth, float(uv[0, 0]), float(uv[0, 1]), mask)
        p2_depth = self._depth_stats_at_uv(depth, float(uv[1, 0]), float(uv[1, 1]), mask)
        z1 = float(p1_depth["z_mm"])
        z2 = float(p2_depth["z_mm"])
        p1_cam = self._uv_to_camera_point_mm(float(uv[0, 0]), float(uv[0, 1]), z1)
        p2_cam = self._uv_to_camera_point_mm(float(uv[1, 0]), float(uv[1, 1]), z2)
        line_cam = np.stack([p1_cam, p2_cam], axis=0).astype(np.float32)
        line_base = self._transform_camera_points(T_base2cam, line_cam)

        if line_base.shape != (2, 3):
            line_base = np.zeros((2, 3), dtype=np.float32)
        p1 = line_base[0].astype(np.float32)
        p2 = line_base[1].astype(np.float32)
        center = ((p1 + p2) * 0.5).astype(np.float32)
        main_axis = (p2 - p1).astype(np.float32)
        axis_norm = float(np.linalg.norm(main_axis))
        if axis_norm > 1e-12:
            main_axis = main_axis / axis_norm
        else:
            main_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        depth_valid = bool(z1 > 0 and z2 > 0)
        endpoint_depth_std_mm = max(float(p1_depth["std_mm"]), float(p2_depth["std_mm"]))
        min_endpoint_depth_samples = min(int(p1_depth["count"]), int(p2_depth["count"]))
        endpoint_z_gap_mm = abs(z1 - z2) if depth_valid else np.inf
        endpoint_distance_mm = float(np.linalg.norm(p2 - p1)) if depth_valid else np.inf
        exact_depth_ok = bool(p1_depth["exact_valid"] and p2_depth["exact_valid"])
        depth_reliable = (
            bool(on_mask_diagonal)
            and depth_valid
            and (not self.require_endpoint_exact_depth or exact_depth_ok)
            and min_endpoint_depth_samples >= self.endpoint_min_depth_samples
            and endpoint_depth_std_mm <= self.endpoint_max_depth_std_mm
            and endpoint_z_gap_mm <= self.endpoint_max_z_gap_mm
            and endpoint_distance_mm <= self.endpoint_max_distance_mm
        )

        self._last_pc = np.asarray(pc_base, dtype=np.float32).reshape(-1, 3)
        self._last_line = line_base.astype(np.float32)

        loc = {
            "line": line_base.astype(np.float32),
            "line_cam": line_cam.astype(np.float32),
            "line_uv": uv.astype(np.float32),
            "p1": p1.astype(np.float32),
            "p2": p2.astype(np.float32),
            "center": center.astype(np.float32),
            "main_axis": main_axis.astype(np.float32),
            "pc": self._last_pc,
            "line_fit_method": "bbox_diagonal_exact_depth",
            "diagonal_mask_endpoint_valid": bool(on_mask_diagonal),
            "p1_depth_valid": bool(z1 > 0),
            "p2_depth_valid": bool(z2 > 0),
            "depth_valid": depth_valid,
            "depth_reliable": bool(depth_reliable),
            "p1_depth_mm": z1,
            "p2_depth_mm": z2,
            "p1_depth_std_mm": float(p1_depth["std_mm"]),
            "p2_depth_std_mm": float(p2_depth["std_mm"]),
            "endpoint_depth_std_mm": endpoint_depth_std_mm,
            "p1_depth_samples": int(p1_depth["count"]),
            "p2_depth_samples": int(p2_depth["count"]),
            "min_endpoint_depth_samples": min_endpoint_depth_samples,
            "p1_exact_depth_valid": bool(p1_depth["exact_valid"]),
            "p2_exact_depth_valid": bool(p2_depth["exact_valid"]),
            "endpoint_z_gap_mm": float(endpoint_z_gap_mm),
            "endpoint_distance_mm": endpoint_distance_mm,
        }
        if not depth_reliable:
            loc["reject_reason"] = self._depth_reject_reason(loc)
        return loc

    def _instance_to_pc(self, instance, depth=None, return_quality=False, enforce_quality=True):
        """
        将单个实例掩码区域转换为三维点云。
        深度原始值先按RealSense深度单位转成mm，再用相机畸变参数校正像素射线后反投影。
        """
        if depth is None:
            depth = self._last_depth
        if depth is None:
            raise ValueError("self._last_depth is None, please call run_localization(instances, depth) first.")

        depth = np.asarray(depth)
        if depth.ndim == 3:
            depth = depth[..., 0]

        mask = self._prepare_depth_sample_mask(instance["mask"])
        rows, cols = np.where(mask > 0)
        sample_count = int(len(rows))
        if len(rows) == 0:
            empty = np.zeros((0, 3), dtype=np.float32)
            quality = self._make_depth_quality(sample_count=0)
            return (empty, quality) if return_quality else empty

        z_raw = depth[rows, cols].astype(np.float32).reshape(-1)
        z = self._depth_raw_to_mm(z_raw)

        valid = np.isfinite(z) & (z > 0) & (z < Config.depth_threshold)
        rows = rows[valid]
        cols = cols[valid]
        z = z[valid]

        if len(z) == 0:
            empty = np.zeros((0, 3), dtype=np.float32)
            quality = self._make_depth_quality(sample_count=sample_count, valid_count=0)
            return (empty, quality) if return_quality else empty

        valid_count = int(len(z))
        rows, cols, z, cluster_info = self._keep_depth_main_cluster(rows, cols, z, return_info=True)
        quality = self._make_depth_quality(
            sample_count=sample_count,
            valid_count=valid_count,
            cluster_count=int(len(z)),
            z=z,
            cluster_info=cluster_info,
        )
        quality["accepted"] = self._is_depth_quality_acceptable(quality)
        if enforce_quality and not quality["accepted"]:
            empty = np.zeros((0, 3), dtype=np.float32)
            return (empty, quality) if return_quality else empty

        pc = self._pixels_to_camera_points(rows, cols, z)
        return (pc, quality) if return_quality else pc

    def _prepare_depth_sample_mask(self, mask):
        mask = mask.astype(np.uint8)
        if self.mask_erode_iterations <= 0 or self.mask_erode_kernel_size <= 1:
            return mask

        kernel_size = max(1, int(self.mask_erode_kernel_size))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        eroded = cv2.erode(mask, kernel, iterations=max(1, int(self.mask_erode_iterations)))
        # 果梗很细时腐蚀可能把mask吃空，回退到原mask避免定位失败。
        return eroded if int(eroded.sum()) >= 8 else mask

    def _depth_raw_to_mm(self, z_raw):
        z_raw = np.asarray(z_raw, dtype=np.float32)
        scale_m = float(self.depth_scale)
        if not np.isfinite(scale_m) or scale_m <= 0.0:
            scale_m = 0.001
        return z_raw * (scale_m * 1000.0)

    def _keep_depth_main_cluster(self, rows, cols, z, return_info=False):
        info = {
            "fallback": False,
            "raw_count": int(len(z)),
            "kept_count": int(len(z)),
            "median_mm": float(np.median(z)) if len(z) else None,
            "half_window_mm": None,
        }
        if len(z) < 12:
            return (rows, cols, z, info) if return_info else (rows, cols, z)

        z_median = float(np.median(z))
        mad = float(np.median(np.abs(z - z_median)))
        robust_sigma = 1.4826 * mad
        half_window = max(self.depth_cluster_min_window_mm, self.depth_cluster_mad_scale * robust_sigma)
        keep = np.abs(z - z_median) <= half_window
        info.update({
            "median_mm": z_median,
            "mad_mm": mad,
            "robust_sigma_mm": robust_sigma,
            "half_window_mm": float(half_window),
            "kept_count": int(np.count_nonzero(keep)),
        })

        min_keep = max(8, int(np.ceil(len(z) * self.depth_cluster_min_keep_ratio)))
        if int(np.count_nonzero(keep)) < min_keep:
            info["fallback"] = True
            info["kept_count"] = int(len(z))
            return (rows, cols, z, info) if return_info else (rows, cols, z)

        kept_rows, kept_cols, kept_z = rows[keep], cols[keep], z[keep]
        return (kept_rows, kept_cols, kept_z, info) if return_info else (kept_rows, kept_cols, kept_z)

    def _make_depth_quality(
        self,
        sample_count,
        valid_count=0,
        cluster_count=0,
        z=None,
        cluster_info=None,
    ):
        cluster_info = cluster_info or {}
        valid_ratio = float(valid_count / sample_count) if sample_count > 0 else 0.0
        cluster_ratio = float(cluster_count / valid_count) if valid_count > 0 else 0.0

        if z is None or len(z) == 0:
            z_median = None
            z_mad = None
            z_span = None
        else:
            z = np.asarray(z, dtype=np.float32).reshape(-1)
            z_median = float(np.median(z))
            z_mad = float(np.median(np.abs(z - z_median)))
            if len(z) >= 2:
                z_low, z_high = np.percentile(z, [5, 95])
                z_span = float(z_high - z_low)
            else:
                z_span = 0.0

        return {
            "sample_count": int(sample_count),
            "valid_count": int(valid_count),
            "cluster_count": int(cluster_count),
            "valid_ratio": valid_ratio,
            "cluster_ratio": cluster_ratio,
            "z_median_mm": z_median,
            "z_mad_mm": z_mad,
            "z_span_mm": z_span,
            "cluster_info": cluster_info,
        }

    def _is_depth_quality_acceptable(self, quality):
        if int(quality.get("valid_count", 0)) < self.depth_quality_min_points:
            return False
        if int(quality.get("cluster_count", 0)) < self.depth_quality_min_points:
            return False
        if float(quality.get("valid_ratio", 0.0)) < self.depth_quality_min_valid_ratio:
            return False
        if float(quality.get("cluster_ratio", 0.0)) < self.depth_quality_min_cluster_ratio:
            return False

        z_mad = quality.get("z_mad_mm")
        if z_mad is not None and float(z_mad) > self.depth_quality_max_mad_mm:
            return False

        z_span = quality.get("z_span_mm")
        if z_span is not None and float(z_span) > self.depth_quality_max_span_mm:
            return False

        return True

    def _pixels_to_camera_points(self, rows, cols, z_mm):
        rows = np.asarray(rows, dtype=np.float32).reshape(-1)
        cols = np.asarray(cols, dtype=np.float32).reshape(-1)
        z_mm = np.asarray(z_mm, dtype=np.float32).reshape(-1)
        if len(z_mm) == 0:
            return np.zeros((0, 3), dtype=np.float32)

        pixels = np.stack([cols, rows], axis=1).astype(np.float32).reshape(-1, 1, 2)
        if self.distortion.size > 0 and np.any(np.abs(self.distortion) > 1e-12):
            normalized = cv2.undistortPoints(pixels, self.camera_matrix, self.distortion).reshape(-1, 2)
            x_norm = normalized[:, 0].astype(np.float32)
            y_norm = normalized[:, 1].astype(np.float32)
        else:
            x_norm = (cols - float(self.cx)) / float(self.fx)
            y_norm = (rows - float(self.cy)) / float(self.fy)

        x = x_norm * z_mm
        y = y_norm * z_mm
        return np.stack([x, y, z_mm], axis=1).astype(np.float32)

    def _cam_to_robot(self, ee_pose, pc_cam):
        T_base2end = self.robot.pose2homography(ee_pose)
        T_end2cam = self.robot._get_homography_from_R_P(self.cam2end_R, self.cam2end_T.reshape(3, 1))
        T_base2cam = T_base2end @ T_end2cam
        ones_col = np.ones((pc_cam.shape[0], 1), dtype=np.float32)
        pc_homo = np.hstack([pc_cam, ones_col])
        pc_base_homo = (T_base2cam @ pc_homo.T).T
        return pc_base_homo[:, :3], T_base2cam

    def _pc_filter(self, pc):
        """
        用 K 近邻平均距离剔除统计离群点。

        该方法移植自 pcl.py 的 pcfilter：孤立噪点的近邻距离通常明显大于
        果梗主点群，因此用“全局均值 + 标准差倍数”作为阈值。
        """
        if pc is None or len(pc) == 0:
            return np.zeros((0, 3), dtype=np.float32)

        pc = np.asarray(pc, dtype=np.float32).reshape(-1, 3)
        pc = pc[np.all(np.isfinite(pc), axis=1)]
        if len(pc) < 2:
            return pc.astype(np.float32)

        neighbor_count = min(self.point_filter_neighbors, len(pc) - 1)
        if neighbor_count < 1 or len(pc) <= self.point_filter_neighbors:
            self._last_point_filter_info = {
                "input_count": int(len(pc)),
                "kept_count": int(len(pc)),
                "neighbor_count": int(neighbor_count),
                "fallback": False,
            }
            return pc.astype(np.float32)

        try:
            # OpenCV FLANN 只支持 float32；返回的 dists 是平方距离。
            flann_params = {"algorithm": 1, "trees": 4}  # 1 = KDTree
            kdtree = cv2.flann_Index(np.ascontiguousarray(pc), flann_params)
            _, squared_distances = kdtree.knnSearch(
                np.ascontiguousarray(pc),
                neighbor_count + 1,
                params={},
            )
            mean_neighbor_distance = np.mean(squared_distances[:, 1:], axis=1)
            distance_mean = float(np.mean(mean_neighbor_distance))
            distance_std = float(np.std(mean_neighbor_distance))
            threshold = distance_mean + self.point_filter_std_ratio * distance_std
            keep = mean_neighbor_distance <= threshold
            filtered = pc[keep]

            # 极端情况下不允许滤波器把可拟合点全部吃掉。
            if len(filtered) < 2:
                filtered = pc
                keep = np.ones(len(pc), dtype=bool)

            self._last_point_filter_info = {
                "input_count": int(len(pc)),
                "kept_count": int(len(filtered)),
                "neighbor_count": int(neighbor_count),
                "mean_squared_neighbor_distance": distance_mean,
                "std_squared_neighbor_distance": distance_std,
                "threshold": float(threshold),
                "fallback": False,
            }
            return filtered.astype(np.float32)
        except cv2.error:
            # 个别 OpenCV 构建不包含 FLANN 时，保留有限的径向分位数回退，避免主程序中断。
            center = np.median(pc, axis=0, keepdims=True)
            distances = np.linalg.norm(pc - center, axis=1)
            threshold = float(np.percentile(distances, 95))
            filtered = pc[distances <= threshold]
            if len(filtered) < 2:
                filtered = pc
            self._last_point_filter_info = {
                "input_count": int(len(pc)),
                "kept_count": int(len(filtered)),
                "neighbor_count": int(neighbor_count),
                "threshold": threshold,
                "fallback": True,
            }
            return filtered.astype(np.float32)

    def _ransac_line_fitting(self, pc):
        """
        先用 RANSAC 找到果梗直线内点，再对内点做 PCA 精修。

        相比直接对全部点做 PCA，RANSAC 不容易被 mask 中混入的叶片、果实或
        背景深度拉偏；PCA 精修则比直接使用随机两点的方向更稳定。
        """
        if pc is None:
            return self._pca_fitting(np.zeros((0, 3), dtype=np.float32))

        points = np.asarray(pc, dtype=np.float32).reshape(-1, 3)
        points = points[np.all(np.isfinite(points), axis=1)]
        point_count = int(len(points))
        if point_count < 3:
            result = self._pca_fitting(points)
            result["line_fit_method"] = "pca_fallback"
            result["ransac_info"] = {
                "point_count": point_count,
                "inlier_count": point_count,
                "inlier_ratio": 1.0 if point_count else 0.0,
                "accepted": False,
            }
            return result

        min_inlier_count = max(2, int(np.ceil(self.line_ransac_min_inlier_ratio * point_count)))
        best_mask = None
        best_count = 0
        best_mean_error = np.inf
        points_float64 = points.astype(np.float64)

        # 固定随机种子使同一帧点云重复计算时得到一致结果，便于机器人调试。
        rng = np.random.default_rng(0)
        for _ in range(self.line_ransac_max_iterations):
            sample_indices = rng.choice(point_count, size=2, replace=False)
            line_origin = points_float64[sample_indices[0]]
            line_vector = points_float64[sample_indices[1]] - line_origin
            line_norm = float(np.linalg.norm(line_vector))
            if not np.isfinite(line_norm) or line_norm < 1e-9:
                continue

            line_direction = line_vector / line_norm
            offsets = points_float64 - line_origin[None, :]
            perpendicular = offsets - np.outer(offsets @ line_direction, line_direction)
            distances = np.linalg.norm(perpendicular, axis=1)
            inlier_mask = distances < self.line_ransac_threshold_mm
            inlier_count = int(np.count_nonzero(inlier_mask))
            mean_error = float(np.mean(distances[inlier_mask])) if inlier_count else np.inf

            if inlier_count > best_count or (
                inlier_count == best_count and mean_error < best_mean_error
            ):
                best_mask = inlier_mask
                best_count = inlier_count
                best_mean_error = mean_error

            if best_count == point_count:
                break

        accepted = best_mask is not None and best_count >= min_inlier_count
        fitting_points = points[best_mask] if accepted else points
        result = self._pca_fitting(fitting_points)
        result["line_fit_method"] = "ransac_pca" if accepted else "pca_fallback"
        result["raw_pc"] = points.astype(np.float32)
        result["ransac_info"] = {
            "point_count": point_count,
            "inlier_count": int(best_count),
            "inlier_ratio": float(best_count / point_count),
            "required_inlier_count": int(min_inlier_count),
            "distance_threshold_mm": float(self.line_ransac_threshold_mm),
            "mean_inlier_error_mm": None if best_mask is None else float(best_mean_error),
            "accepted": bool(accepted),
        }
        return result

    def _pca_fitting(self, pc):
        """用 PCA 对三维点云拟合一条主轴直线，输出两个端点。"""
        if pc is None or len(pc) == 0:
            line = np.zeros((2, 3), dtype=np.float32)
            return {
                "line": line,
                "p1": line[0],
                "p2": line[1],
                "center": np.zeros(3, dtype=np.float32),
                "main_axis": np.array([1.0, 0.0, 0.0], dtype=np.float32),
                "pc": np.zeros((0, 3), dtype=np.float32),
            }

        cloud_center = np.mean(pc, axis=0).astype(np.float32)
        centered = pc - cloud_center[None, :]
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        main_axis = vh[0].astype(np.float32)
        main_axis = main_axis / (np.linalg.norm(main_axis) + 1e-12)

        proj = centered @ main_axis
        # 使用稳健分位数端点，避免少量残余离群点把线段拉得过长。
        t_min, t_max = np.percentile(proj, self.line_endpoint_percentiles)
        p1 = cloud_center + float(t_min) * main_axis
        p2 = cloud_center + float(t_max) * main_axis
        line = np.stack([p1, p2], axis=0).astype(np.float32)

        # 统一方向沿旧版 locator 的约定：果梗 X 轴尽量从采摘点沿果梗向上延伸，
        # 只按 Z 分量翻面，不按 Y 坐标排序，否则左右果梗会被翻成同一旋转方向。
        if main_axis[2] < 0:
            p1, p2 = p2, p1
            line = np.stack([p1, p2], axis=0).astype(np.float32)
            main_axis = (p2 - p1).astype(np.float32)
            main_axis = main_axis / (np.linalg.norm(main_axis) + 1e-12)

        # 此处中点只用于多帧一致性判断和重复目标过滤；
        # 对外输出前会由同名_line_midpoint函数改为较低端点。
        center = ((p1 + p2) * 0.5).astype(np.float32)

        self._last_pc = pc.astype(np.float32)
        self._last_line = line.astype(np.float32)

        return {
            "line": line.astype(np.float32),
            "p1": p1.astype(np.float32),
            "p2": p2.astype(np.float32),
            "center": center.astype(np.float32),
            "main_axis": main_axis.astype(np.float32),
            "pc": pc.astype(np.float32),
        }

    def cal_picking_pose(self, tomato_info, max_angle, picking_offset, force_vertical_fallback=False):
        if not self._is_valid_localization(tomato_info):
            raise ValueError("定位结果无效，跳过当前果实")
        if self._is_main_stem_class(tomato_info):
            raise ValueError("main_stem 是主茎保护目标，不能作为采摘目标")

        # 无论上游 center 字段是否为旧值，都从 P1/P2 重选相对地面更低的端点。
        pick_point = self._line_midpoint(tomato_info)
        if pick_point is None:
            raise ValueError("无法由 P1/P2 选择较低端点")
        p1 = np.asarray(tomato_info.get("p1"), dtype=np.float64)
        picked_p1 = p1.shape == (3,) and np.allclose(pick_point, p1)
        tomato_info["center"] = pick_point.astype(np.float32)
        tomato_info["pick_point"] = pick_point.astype(np.float32)
        tomato_info["pick_point_name"] = tomato_info.get(
            "point1_name" if picked_p1 else "point2_name",
            "P1" if picked_p1 else "P2",
        )
        tomato_info["pick_point_source"] = "p1p2_midpoint"
        tomato_info["pick_point_base_z_mm"] = float(pick_point[2])
        center = pick_point.copy()
        stem_direction = self._line_direction(tomato_info)
        if stem_direction is None:
            raise ValueError("果梗方向无效，跳过当前果实")

        # picking_offset 是机械结构/标定补偿，在较低端点基础上叠加。
        center[0] += picking_offset[0]
        center[1] += picking_offset[1]
        center[2] += picking_offset[2]

        restrained_stem_direction = self._stem_angle_restrict(stem_direction, self.ref_vec, max_angle)
        pose_direction = restrained_stem_direction
        adjusted_center, offset_info = self._apply_main_stem_offset(
            center,
            tomato_info,
            pose_direction=pose_direction,
        )
        tomato_info["main_stem_offset_info"] = offset_info
        pose_center = adjusted_center
        fallback_strategy_enabled = bool(Config.enable_fallback_picking_strategy)
        fallback_info = {
            "enabled": False,
            "strategy_enabled": fallback_strategy_enabled,
            "forced_by_ik_failure": bool(force_vertical_fallback),
            "triggered_by_angle": False,
            "down_pitch_deg": self._fallback_pose_down_pitch_deg(),
            "reference_axis_angle_deg": None,
            "reference_axis_pitch_deg": None,
            "reference_axis": self._vertical_x_fallback_reference_axis(),
            "threshold_deg": self._vertical_x_fallback_trigger_angle_deg(),
        }

        if force_vertical_fallback or not fallback_strategy_enabled:
            exceeds = False
            reference_axis_pitch_deg = None
        else:
            # 先沿用旧版 locator 的径向规则构造 TCP Z 轴，再判断它是否翻向机器侧。
            normal_ez = self._legacy_radial_z_axis(pose_direction, pose_center)
            if normal_ez is None:
                raise ValueError("无法构造有效的旧版径向 TCP Z 轴")

            exceeds, reference_axis_pitch_deg = self._tcp_z_exceeds_reference_axis_angle(
                pose_direction,
                normal_ez,
                pose_center,
            )
        fallback_info["reference_axis_angle_deg"] = reference_axis_pitch_deg
        fallback_info["reference_axis_pitch_deg"] = reference_axis_pitch_deg

        fallback_info["triggered_by_angle"] = bool(exceeds)

        if fallback_strategy_enabled and (force_vertical_fallback or exceeds):
            # 备用策略只改变姿态和补偿，位置基准仍使用同一个较低端点。
            fallback_center = pick_point.copy()
            fallback_offset = self._fallback_picking_offset()
            fallback_center[0] += fallback_offset[0]
            fallback_center[1] += fallback_offset[1]
            fallback_center[2] += fallback_offset[2]
            pose_direction = self._fallback_pose_direction()
            fallback_adjusted_center, fallback_offset_info = self._apply_main_stem_offset(
                fallback_center,
                tomato_info,
                pose_direction=pose_direction,
            )

            pose_center = fallback_adjusted_center
            tomato_info["main_stem_offset_info"] = fallback_offset_info
            fallback_info.update({
                "enabled": True,
                "base_pick_point": fallback_center.astype(np.float64),
                "adjusted_pick_point": fallback_adjusted_center.astype(np.float64),
                "x_axis": pose_direction.copy(),
            })
        tomato_info["vertical_x_fallback_info"] = fallback_info

        # 姿态保持最开始旧版 locator 的思路：Z 轴只由果梗和基座径向确定，不使用主茎法向。
        tcp_rotation = self._calculate_tcp_rotation(pose_direction, pose_center)
        tcp_rotation = self._align_tcp_x_parallel_to_stem(
            tcp_rotation,
            stem_direction,
        )
        if fallback_info["enabled"]:
            fallback_info["x_axis"] = tcp_rotation[:, 0].copy()
        rpy = self._rotation2rpy(tcp_rotation)

        if fallback_info["enabled"]:
            end_effector_offset = Config.fallback_picking_offset_end_effector
        else:
            end_effector_offset = Config.picking_offset_end_effector
        end_effector_offset = np.asarray(end_effector_offset, dtype=np.float64).reshape(3)

        # TCP 坐标系中的平移量通过目标姿态旋转到基坐标系，再与原有基坐标偏置叠加。
        pose_center = pose_center + tcp_rotation @ end_effector_offset

        target_pose = np.array([
            pose_center[0],
            pose_center[1],
            pose_center[2],
            rpy[0],
            rpy[1],
            rpy[2],
        ])

        previous_pose = target_pose.copy()
        if Config.use_previous_pose:
            previous_pose[:3] -= (
                float(Config.previous_pose_distance_mm) * tcp_rotation[:, 2]
            )
        else:
            previous_pose[0] += 100.0
            previous_pose[1] += 30.0
            previous_pose[2] += 30.0

        return target_pose, previous_pose

    def _apply_main_stem_offset(self, center, stem_info, pose_direction=None):
        main_info = stem_info.get("main_stem_info")
        nearest_point = stem_info.get("main_stem_nearest_point")
        distance = stem_info.get("main_stem_distance_mm")

        if main_info is not None:
            line = np.asarray(main_info.get("line"), dtype=np.float64)
            if line.shape == (2, 3):
                nearest_point = self._nearest_point_on_segment(center, line[0], line[1])
                distance = float(np.linalg.norm(center - nearest_point))
        elif main_info is None:
            main_stems = [
                loc for loc in getattr(self, "_last_localizations", [])
                if self._is_main_stem_class(loc)
            ]
            main_info, nearest_point, distance = self._nearest_main_stem(stem_info, main_stems, center=center)

        half_opening_mm = max(0.0, float(self.scissor_max_opening_mm) * 0.5)
        activation_distance_mm = float(self.main_stem_offset_activation_distance_mm)

        info = {
            "enabled": False,
            "dynamic_offset_enabled": bool(self.enable_main_stem_dynamic_offset),
            "distance_mm": None if distance is None or not np.isfinite(distance) else float(distance),
            "activation_distance_mm": activation_distance_mm,
            "safe_clearance_mm": self.main_stem_safe_clearance_mm,
            "scissor_max_opening_mm": self.scissor_max_opening_mm,
            "half_opening_mm": half_opening_mm,
            "blade_distance_mm": None,
            "signed_blade_distance_mm": None,
            "offset_mm": 0.0,
            "direction": np.zeros(3, dtype=np.float64),
            "blade_axis": np.zeros(3, dtype=np.float64),
        }

        if not self.enable_main_stem_dynamic_offset:
            return center, info
        if main_info is None or nearest_point is None or distance is None or not np.isfinite(distance):
            return center, info

        line = np.asarray(main_info.get("line"), dtype=np.float64)
        if line.shape != (2, 3):
            return center, info

        blade_axis = self._tool_blade_axis(center, stem_info, pose_direction=pose_direction)
        if blade_axis is None:
            return center, info

        signed_distance = self._signed_distance_along_axis_to_segment(center, blade_axis, line[0], line[1])
        if signed_distance is None or not np.isfinite(signed_distance):
            signed_distance = float(np.dot(np.asarray(nearest_point, dtype=np.float64).reshape(3) - center, blade_axis))

        blade_distance = abs(float(signed_distance))
        info.update({
            "blade_axis": blade_axis.astype(np.float64),
            "blade_distance_mm": float(blade_distance),
            "signed_blade_distance_mm": float(signed_distance),
        })

        if blade_distance >= activation_distance_mm:
            return center, info
        if blade_distance <= 1e-6 or half_opening_mm <= 1e-6:
            return center, info

        main_side = 1.0 if signed_distance >= 0.0 else -1.0
        direction = -main_side * blade_axis
        offset_mm = max(0.0, half_opening_mm - blade_distance * 0.8)
        if offset_mm <= 1e-6:
            return center, info

        adjusted_center = center + direction * offset_mm
        final_signed_distance = signed_distance + main_side * offset_mm
        final_blade_distance = abs(float(final_signed_distance))

        info.update({
            "enabled": True,
            "offset_mm": float(offset_mm),
            "final_blade_distance_mm": float(final_blade_distance),
            "direction": direction.astype(np.float64),
        })
        return adjusted_center, info

    @staticmethod
    def _line_length(localization):
        if not localization:
            return None
        line = np.asarray(localization.get("line"), dtype=np.float64)
        if line.shape != (2, 3):
            return None
        length = float(np.linalg.norm(line[1] - line[0]))
        return length if np.isfinite(length) else None

    @staticmethod
    def _distance_point_to_segment(point, seg_a, seg_b):
        nearest = Locator._nearest_point_on_segment(point, seg_a, seg_b)
        return float(np.linalg.norm(np.asarray(point, dtype=np.float64).reshape(3) - nearest))

    def _tool_blade_axis(self, center, stem_info, pose_direction=None):
        if pose_direction is not None:
            ex = self._normalize(pose_direction)
        else:
            ex = self._line_direction(stem_info)
        if ex is None:
            return None

        ez = self._legacy_radial_z_axis(ex, center)
        if ez is None:
            return None
        return self._normalize(np.cross(ez, ex))

    @staticmethod
    def _signed_distance_along_axis_to_segment(point, axis, seg_a, seg_b):
        point = np.asarray(point, dtype=np.float64).reshape(3)
        axis = np.asarray(axis, dtype=np.float64).reshape(3)
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-9 or not np.isfinite(axis_norm):
            return None
        axis = axis / axis_norm

        seg_a = np.asarray(seg_a, dtype=np.float64).reshape(3)
        seg_b = np.asarray(seg_b, dtype=np.float64).reshape(3)
        segment_vec = seg_b - seg_a
        segment_len2 = float(np.dot(segment_vec, segment_vec))
        if segment_len2 < 1e-9:
            return float(np.dot(seg_a - point, axis))

        w0 = point - seg_a
        axis_segment_dot = float(np.dot(axis, segment_vec))
        axis_w0_dot = float(np.dot(axis, w0))
        segment_w0_dot = float(np.dot(segment_vec, w0))
        denominator = segment_len2 - axis_segment_dot * axis_segment_dot

        if abs(denominator) > 1e-9:
            segment_param = (segment_w0_dot - axis_segment_dot * axis_w0_dot) / denominator
            segment_param = float(np.clip(segment_param, 0.0, 1.0))
        else:
            segment_param = float(np.clip(-segment_w0_dot / segment_len2, 0.0, 1.0))

        nearest_on_segment = seg_a + segment_param * segment_vec
        return float(np.dot(nearest_on_segment - point, axis))

    @staticmethod
    def _stem_angle_restrict(original_vec, reference_vec_normalized, max_angle):
        original_vec_normalized = original_vec / np.linalg.norm(original_vec)

        dot = np.clip(np.dot(original_vec_normalized, reference_vec_normalized), -1.0, 1.0)
        original_angle = np.arccos(dot)
        if original_angle <= max_angle:
            print("果梗角度无需调整")
            return original_vec_normalized

        rotation_axis = np.cross(reference_vec_normalized, original_vec_normalized)
        rotation_axis_norm = np.linalg.norm(rotation_axis)
        if rotation_axis_norm < 1e-6:
            print("向量几乎共线，果梗角度无需调整")
            return original_vec_normalized

        v = reference_vec_normalized
        k = rotation_axis / rotation_axis_norm
        theta = max_angle

        v_rot = v * np.cos(theta) + np.cross(k, v) * np.sin(theta)
        return v_rot / np.linalg.norm(v_rot)

    def _calculate_tcp_rotation(self, stem_vector, pp_base):
        ex = self._normalize(stem_vector)
        if ex is None:
            raise ValueError("果梗方向向量无效")

        ez = self._legacy_radial_z_axis(ex, pp_base)
        if ez is None:
            raise ValueError("无法构造有效的旧版径向 TCP Z 轴")

        ey = self._normalize(np.cross(ez, ex))
        if ey is None:
            raise ValueError("无法构造有效的 TCP Y 轴")

        ez = self._normalize(np.cross(ex, ey))
        if ez is None:
            raise ValueError("无法正交化 TCP Z 轴")

        return np.array([ex, ey, ez]).T

    def _align_tcp_x_parallel_to_stem(self, tcp_rotation, stem_vector):
        tcp_x = self._normalize(stem_vector)
        if tcp_x is None:
            raise ValueError("P1P2果梗方向向量无效")

        tcp_rotation = np.asarray(tcp_rotation, dtype=np.float64)
        tcp_z_reference = self._normalize(tcp_rotation[:, 2])
        if tcp_z_reference is None:
            raise ValueError("TCP Z轴参考方向无效")

        tcp_z = self._normalize(
            tcp_z_reference - np.dot(tcp_z_reference, tcp_x) * tcp_x
        )
        if tcp_z is None:
            for candidate in (
                tcp_rotation[:, 1],
                np.array([1.0, 0.0, 0.0], dtype=np.float64),
                np.array([0.0, 1.0, 0.0], dtype=np.float64),
                np.array([0.0, 0.0, 1.0], dtype=np.float64),
            ):
                candidate = candidate - np.dot(candidate, tcp_x) * tcp_x
                tcp_z = self._normalize(candidate)
                if tcp_z is not None:
                    break
        if tcp_z is None:
            raise ValueError("无法构造垂直于P1P2果梗方向的TCP Z轴")

        tcp_y = self._normalize(np.cross(tcp_z, tcp_x))
        if tcp_y is None:
            raise ValueError("无法构造有效的TCP Y轴")
        tcp_z = self._normalize(np.cross(tcp_x, tcp_y))
        return np.array([tcp_x, tcp_y, tcp_z]).T

    def _calculate_rpy(self, stem_vector, pp_base):
        return self._rotation2rpy(self._calculate_tcp_rotation(stem_vector, pp_base))

    @staticmethod
    def _vertical_x_fallback_trigger_angle_deg():
        return float(Config.vertical_x_fallback_trigger_angle_deg)

    @staticmethod
    def _fallback_pose_down_pitch_deg():
        return float(Config.fallback_pose_down_pitch_deg)

    @staticmethod
    def _fallback_picking_offset():
        return np.asarray(Config.fallback_picking_offset, dtype=np.float64).reshape(3)

    def _fallback_pose_direction(self):
        angle_rad = np.deg2rad(self._fallback_pose_down_pitch_deg())
        return self._normalize(np.array([
            -np.sin(angle_rad),
            0.0,
            np.cos(angle_rad),
        ], dtype=np.float64))

    @staticmethod
    def _vertical_x_fallback_reference_axis():
        """目标侧水平参考轴固定为基座 -X，不再从 Config 读取。"""
        return np.array([-1.0, 0.0, 0.0], dtype=np.float64)

    def _tcp_z_exceeds_reference_axis_angle(self, ex, ez, pp_base):
        """判断 TCP Z 相对目标侧水平轴的上仰/下俯角是否超过阈值。"""
        max_angle_deg = self._vertical_x_fallback_trigger_angle_deg()
        if max_angle_deg >= 180.0:
            return False, None

        max_angle_deg = float(np.clip(max_angle_deg, 0.0, 180.0))
        ez = self._normalize(ez)
        if ez is None:
            return False, None

        # 只在“目标侧水平轴-基座Z轴”这个垂直平面里计算有符号俯仰角。
        reference_axis = self._vertical_x_fallback_reference_axis()
        vertical_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        horizontal_component = float(np.dot(ez, reference_axis))
        vertical_component = float(np.dot(ez, vertical_axis))
        pitch_plane_norm = float(np.hypot(horizontal_component, vertical_component))
        if pitch_plane_norm < 1e-9:
            return False, None

        signed_pitch_angle = float(np.rad2deg(np.arctan2(vertical_component, horizontal_component)))
        return abs(signed_pitch_angle) > max_angle_deg + 1e-9, signed_pitch_angle

    def _legacy_radial_z_axis(self, ex, pp_base):
        pp_base = np.asarray(pp_base, dtype=np.float64).reshape(3)
        Px = float(pp_base[0])
        Py = float(pp_base[1])

        if abs(float(ex[2])) > 1e-8:
            z = -(ex[0] * Px + ex[1] * Py) / ex[2]
            ez = self._normalize(np.array([Px, Py, z], dtype=np.float64))
            if ez is not None:
                return ez

        radial = np.array([Px, Py, 0.0], dtype=np.float64)
        radial_norm = np.linalg.norm(radial)
        if radial_norm > 1e-8:
            radial_unit = radial / radial_norm
            ez = radial_unit - np.dot(radial_unit, ex) * ex
            ez = self._normalize(ez)
            if ez is not None:
                if float(np.dot(ez[:2], radial_unit[:2])) < 0.0:
                    ez = -ez
                return ez

        for candidate in (
            np.array([0.0, 0.0, 1.0], dtype=np.float64),
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
        ):
            ez = self._normalize(candidate - np.dot(candidate, ex) * ex)
            if ez is not None:
                return ez
        return None

    @staticmethod
    def _rotation2rpy(R):
        if R.shape == (4, 4):
            R = R[0:3, 0:3]

        pitch = np.arctan2(-R[2][0], np.sqrt(R[0][0] ** 2 + R[1][0] ** 2))

        cos_eps = 1e-10
        if np.abs(np.cos(pitch)) < cos_eps:
            yaw = 0.0
            pitch_sign = np.sign(pitch)
            roll = np.arctan2(pitch_sign * R[0][1], R[1][1])
        else:
            roll = np.arctan2(R[2][1], R[2][2])
            yaw = np.arctan2(R[1][0], R[0][0])

        return np.array([roll, pitch, yaw])

    def _project_xyz_to_uv(self, xyz: np.ndarray) -> np.ndarray:
        """将三维点投影回二维图像平面。"""
        z = np.clip(xyz[:, 2], 1e-6, None)
        u = xyz[:, 0] * float(self.fx) / z + float(self.cx)
        v = xyz[:, 1] * float(self.fy) / z + float(self.cy)
        return np.stack([u, v], axis=1)

    def _safe_project_line_to_image(self, line, image_shape):
        line = np.asarray(line, dtype=np.float64)
        if line.shape != (2, 3) or not np.all(np.isfinite(line)):
            return None
        if np.any(line[:, 2] <= 1e-6):
            return None

        uv = self._project_xyz_to_uv(line)
        if uv.shape != (2, 2) or not np.all(np.isfinite(uv)):
            return None

        height, width = int(image_shape[0]), int(image_shape[1])
        if height <= 0 or width <= 0:
            return None

        uv = np.round(uv)
        if not np.all(np.isfinite(uv)):
            return None
        uv = np.clip(uv, -1000000, 1000000).astype(np.int32)
        p1 = (int(uv[0, 0]), int(uv[0, 1]))
        p2 = (int(uv[1, 0]), int(uv[1, 1]))

        ok, clipped_p1, clipped_p2 = cv2.clipLine((0, 0, width, height), p1, p2)
        if not ok:
            return None
        return (int(clipped_p1[0]), int(clipped_p1[1])), (int(clipped_p2[0]), int(clipped_p2[1]))

    def show_instances(self, bgr):
        """在图像上绘制实例分割 mask、bbox、轮廓和类别名。"""
        canvas = bgr.copy()
        instances = self._last_instances if self._last_instances else self.run_detection(bgr)
        rng = np.random.default_rng(1234)

        for ins in instances:
            base_color = self._line_color(ins.get("class_name", ""))
            if ins.get("class_name") not in {"stem", "main_stem"}:
                base_color = tuple(int(x) for x in rng.integers(0, 255, size=3).tolist())

            mask = ins["mask"]
            overlay = np.zeros_like(canvas)
            overlay[mask] = base_color
            canvas = cv2.addWeighted(canvas, 1.0, overlay, 0.35, 0)

            x1, y1, x2, y2 = ins["bbox"].astype(int)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), base_color, 2)
            cv2.drawContours(canvas, [ins["contour"]], -1, base_color, 2)

            label = f"{ins.get('class_name', 'obj')} {ins.get('score', 0.0):.2f}"
            cv2.putText(canvas, label, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, base_color, 2)

        return canvas

    def show_all_infos_on_images(self, bgr):
        """在图像上显示实例分割结果和各实例三维拟合线段投影。"""
        canvas = self.show_instances(bgr)

        if self._last_localizations:
            for loc in self._last_localizations:
                line = loc["line"]
                class_name = loc.get("class_name", "")
                color = self._line_color(class_name)
                name1 = loc.get("point1_name", "P1")
                name2 = loc.get("point2_name", "P2")

                uv = loc.get("line_uv")
                if uv is not None:
                    uv = np.asarray(uv, dtype=np.float32)
                    if uv.shape != (2, 2) or not np.all(np.isfinite(uv)):
                        continue
                    p1 = tuple(np.round(uv[0]).astype(int).tolist())
                    p2 = tuple(np.round(uv[1]).astype(int).tolist())
                else:
                    projected_line = self._safe_project_line_to_image(line, canvas.shape)
                    if projected_line is None:
                        continue
                    p1, p2 = projected_line

                cv2.line(canvas, p1, p2, color, 2)
                cv2.circle(canvas, p1, 4, (0, 255, 0), -1)
                cv2.circle(canvas, p2, 4, (0, 0, 255), -1)
                cv2.putText(canvas, name1, p1, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(canvas, name2, p2, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                if self._is_stem_class(loc):
                    pick_name = str(loc.get("pick_point_name", ""))
                    pick_uv = p1 if pick_name == name1 else p2
                    cv2.circle(canvas, pick_uv, 9, (255, 255, 0), 2)
                    cv2.putText(
                        canvas,
                        "PICK",
                        (pick_uv[0] + 8, pick_uv[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 0),
                        2,
                    )

        return canvas

    def show_pc_line(self):
        """使用 open3d 显示最近一次点云和拟合直线。"""
        if self._last_pc is None or self._last_line is None:
            raise ValueError("请先执行 run_localization")

        try:
            import open3d as o3d
        except Exception as e:  # pragma: no cover
            raise ImportError("show_pc_line 需要 open3d，请先 pip install open3d") from e

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self._last_pc.astype(np.float64))

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(self._last_line.astype(np.float64))
        line_set.lines = o3d.utility.Vector2iVector(np.array([[0, 1]], dtype=np.int32))
        line_set.paint_uniform_color([1.0, 0.0, 0.0])

        o3d.visualization.draw_geometries([pcd, line_set])

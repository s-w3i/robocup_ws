#!/usr/bin/env python3
"""DeepSORT-based people follow node with 3D fusion for ROS2."""

from __future__ import annotations

import ctypes
import math
import os
import pathlib
import re
import sys
import sysconfig
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import TransformBroadcaster

from yoloe_detection_interfaces.msg import Detection3D, Detection3DArray, PeopleTrack2D, PeopleTrack2DArray
from yoloe_detection_interfaces.msg import PeopleTrack3D, PeopleTrack3DArray
from yoloe_detection_interfaces.srv import SetTracking

HUMAN_CLASS_NAME = "person"


def ensure_torch_runtime_libs() -> None:
    """Expose CUDA libraries from pip wheels and common system paths."""
    purelib = pathlib.Path(sysconfig.get_paths().get("purelib", ""))
    lib_dirs: list[str] = []

    nvidia_root = purelib / "nvidia"
    if nvidia_root.is_dir():
        for lib_dir in nvidia_root.glob("*/lib"):
            if lib_dir.is_dir():
                lib_dirs.append(str(lib_dir))

    for path in (
        "/usr/local/cuda/lib64",
        "/usr/local/cuda-12.6/lib64",
        "/usr/local/cuda/targets/aarch64-linux/lib",
        "/usr/local/cuda-12.6/targets/aarch64-linux/lib",
        "/usr/local/cuda/extras/CUPTI/lib64",
        "/usr/local/cuda-12.6/extras/CUPTI/lib64",
        "/usr/lib/aarch64-linux-gnu",
        "/lib/aarch64-linux-gnu",
    ):
        if pathlib.Path(path).is_dir():
            lib_dirs.append(path)

    current = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [entry for entry in current.split(":") if entry]
    for lib_dir in lib_dirs:
        if lib_dir not in parts:
            parts.insert(0, lib_dir)
    os.environ["LD_LIBRARY_PATH"] = ":".join(parts)


def inject_known_venv_site_packages() -> None:
    """Add common venv site-packages to sys.path for ros2 launch shebang contexts."""
    py_tag = f"python{sys.version_info.major}.{sys.version_info.minor}"
    candidates: list[pathlib.Path] = []

    current_venv = os.environ.get("VIRTUAL_ENV", "").strip()
    if current_venv:
        candidates.append(pathlib.Path(current_venv) / "lib" / py_tag / "site-packages")

    candidates.extend(
        [
            pathlib.Path("/home/usern/follow-venv/lib/python3.10/site-packages"),
            pathlib.Path("/home/usern/coqui-venv/lib/python3.10/site-packages"),
        ]
    )

    # Keep the first candidate as highest priority.
    for candidate in reversed(candidates):
        if candidate.is_dir():
            candidate_str = str(candidate)
            while candidate_str in sys.path:
                sys.path.remove(candidate_str)
            sys.path.insert(0, candidate_str)


def preload_cupti_if_needed() -> None:
    """Preload CUPTI when available to avoid lazy load failures."""
    py_tag = f"python{sys.version_info.major}.{sys.version_info.minor}"
    candidates: list[pathlib.Path] = []

    current_venv = os.environ.get("VIRTUAL_ENV", "").strip()
    if current_venv:
        candidates.append(
            pathlib.Path(current_venv)
            / "lib"
            / py_tag
            / "site-packages"
            / "nvidia"
            / "cuda_cupti"
            / "lib"
            / "libcupti.so.12"
        )

    candidates.extend(
        [
            pathlib.Path(
                "/home/usern/follow-venv/lib/python3.10/site-packages/nvidia/cuda_cupti/lib/libcupti.so.12"
            ),
            pathlib.Path(
                "/home/usern/coqui-venv/lib/python3.10/site-packages/nvidia/cuda_cupti/lib/libcupti.so.12"
            ),
            pathlib.Path("/usr/local/cuda-12.6/extras/CUPTI/lib64/libcupti.so.12"),
            pathlib.Path("/usr/local/cuda/extras/CUPTI/lib64/libcupti.so.12"),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
            return


def patch_torchvision_nms_fallback(
    logger: Any, prefer_ultralytics_torch_nms: bool = True
) -> bool:
    """Patch torchvision NMS when CUDA kernels are missing.

    Fallback priority:
    1) Ultralytics pure-Torch NMS on current tensor device (stays on CUDA)
    2) CPU NMS fallback
    """
    try:
        import torchvision  # type: ignore
    except Exception as exc:
        logger.warn(f"torchvision import failed for NMS patch: {exc}")
        return False

    if getattr(torchvision.ops, "_deepsort_cpu_nms_patch", False):
        return True

    torch_nms_impl = None
    if prefer_ultralytics_torch_nms:
        try:
            from ultralytics.utils.nms import TorchNMS  # type: ignore

            torch_nms_impl = TorchNMS.nms
        except Exception as exc:
            logger.warn(f"Ultralytics TorchNMS not available; will use CPU fallback: {exc}")

    original_nms = torchvision.ops.nms

    def nms_with_cpu_fallback(boxes: Any, scores: Any, iou_threshold: float) -> Any:
        try:
            return original_nms(boxes, scores, iou_threshold)
        except RuntimeError as exc:
            msg = str(exc)
            if "torchvision::nms" not in msg or "CUDA" not in msg:
                raise

            if torch_nms_impl is not None:
                return torch_nms_impl(boxes, scores, iou_threshold)

            output_device = getattr(boxes, "device", None)
            keep = original_nms(boxes.detach().cpu(), scores.detach().cpu(), iou_threshold)
            if output_device is not None and getattr(output_device, "type", "") != "cpu":
                keep = keep.to(output_device)
            return keep

    torchvision.ops.nms = nms_with_cpu_fallback
    setattr(torchvision.ops, "_deepsort_cpu_nms_patch", True)
    if torch_nms_impl is not None:
        logger.warn(
            "Applied torchvision NMS fallback patch: CUDA tensors will use Ultralytics TorchNMS."
        )
    else:
        logger.warn("Applied torchvision NMS CPU fallback patch (CUDA inference remains enabled).")
    return True


def clamp_box(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> tuple[int, int, int, int]:
    x1i = int(max(0, min(width - 1, round(x1))))
    y1i = int(max(0, min(height - 1, round(y1))))
    x2i = int(max(x1i + 1, min(width, round(x2))))
    y2i = int(max(y1i + 1, min(height, round(y2))))
    return x1i, y1i, x2i - x1i, y2i - y1i


def stamp_to_seconds(stamp: Any) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def point_distance(a: PoseStamped, b: PoseStamped) -> float:
    dx = a.pose.position.x - b.pose.position.x
    dy = a.pose.position.y - b.pose.position.y
    dz = a.pose.position.z - b.pose.position.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


@dataclass
class Track2DRecord:
    track_id: int
    class_name: str
    detector_confidence: float
    tracker_confidence: float
    x: int
    y: int
    width: int
    height: int
    reassociated: bool


@dataclass
class Detection2DRecord:
    class_name: str
    confidence: float
    x: int
    y: int
    width: int
    height: int


@dataclass
class ValidTrack3D:
    track_id: int
    pose: PoseStamped
    speed_mps: float


class DeepSortPeopleFollowNode(Node):
    def __init__(self) -> None:
        super().__init__("deepsort_people_follow_node")

        self.declare_parameter("tracking_service_name", "/yoloe/set_tracking")
        self.declare_parameter("people_tracks_2d_topic", "/people_tracks_2d")
        self.declare_parameter("people_tracks_3d_topic", "/people_tracks_3d")
        self.declare_parameter("tracking_compat_topic", "/yoloe/tracking_detections")
        self.declare_parameter("follow_pose_topic", "/people/follow_target_pose")

        self.declare_parameter("color_topic", "/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/color/camera_info")
        self.declare_parameter("camera_link_frame", "camera_link")
        self.declare_parameter("tf_prefix", "person_id")

        self.declare_parameter("model_path", "/home/usern/robocup_ws/yolo11s.pt")
        self.declare_parameter("device", "auto")
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("det_conf", 0.45)
        self.declare_parameter("det_iou", 0.50)
        self.declare_parameter("max_det", 100)
        # Deprecated: node is human-only; kept for backward-compatible parameter overrides.
        self.declare_parameter("tracking_class", HUMAN_CLASS_NAME)
        # Deprecated: open-vocabulary prompting is disabled for YOLO11 human-only mode.
        self.declare_parameter("enable_open_vocab_prompt", False)
        self.declare_parameter("patch_torchvision_nms_cpu_fallback", True)
        self.declare_parameter("prefer_ultralytics_torch_nms", True)

        self.declare_parameter("max_age", 1500)
        self.declare_parameter("n_init", 5)
        self.declare_parameter("max_cosine_distance", 0.25)
        self.declare_parameter("nn_budget", 100)
        self.declare_parameter("nms_max_overlap", 1.0)
        self.declare_parameter("reid_embedder", "auto")
        self.declare_parameter("use_torchreid_embedder", True)
        self.declare_parameter("torchreid_model_name", "osnet_ain_x0_5")
        self.declare_parameter("torchreid_embedder_gpu", True)
        self.declare_parameter("clip_model_name", "ViT-B/16")
        self.declare_parameter("embedder_weights_path", "")
        self.declare_parameter("fallback_to_hsv_embedder", True)

        self.declare_parameter("publish_rate_hz", 0.0)
        self.declare_parameter("diagnostics_period_s", 1.0)

        self.declare_parameter("debug_image_dir", "/home/usern/robocup_ws/yoloe_out")
        self.declare_parameter("debug_save_interval_s", 0.2)
        self.declare_parameter("enable_ui", False)
        self.declare_parameter("ui_window_name", "DeepSORT Tracking")
        self.declare_parameter("ui_show_depth_text", True)

        self.declare_parameter("depth_window_size", 5)
        self.declare_parameter("min_depth_m", 0.2)
        self.declare_parameter("max_depth_m", 8.0)
        self.declare_parameter("max_depth_age_ms", 200.0)

        self.declare_parameter("lost_hold_sec", 2.0)
        self.declare_parameter("reacquire_max_dist_m", 1.5)
        self.declare_parameter("reacquire_max_vel_mps", 1.5)
        self.declare_parameter("motion_weight", 0.4)

        self.tracking_service_name = str(self.get_parameter("tracking_service_name").value)
        self.people_tracks_2d_topic = str(self.get_parameter("people_tracks_2d_topic").value)
        self.people_tracks_3d_topic = str(self.get_parameter("people_tracks_3d_topic").value)
        self.tracking_compat_topic = str(self.get_parameter("tracking_compat_topic").value)
        self.follow_pose_topic = str(self.get_parameter("follow_pose_topic").value)

        self.color_topic = str(self.get_parameter("color_topic").value)
        self.depth_topic = str(self.get_parameter("depth_topic").value)
        self.camera_info_topic = str(self.get_parameter("camera_info_topic").value)
        self.camera_link_frame = str(self.get_parameter("camera_link_frame").value)
        self.tf_prefix = str(self.get_parameter("tf_prefix").value)

        self.model_path = str(self.get_parameter("model_path").value)
        self.device_request = str(self.get_parameter("device").value)
        self.imgsz = int(self.get_parameter("imgsz").value)
        self.det_conf = float(self.get_parameter("det_conf").value)
        self.det_iou = float(self.get_parameter("det_iou").value)
        self.max_det = int(self.get_parameter("max_det").value)
        requested_tracking_class = str(self.get_parameter("tracking_class").value).strip().lower()
        self.tracking_class = HUMAN_CLASS_NAME
        if requested_tracking_class and requested_tracking_class != self.tracking_class:
            self.get_logger().warn(
                f"Ignoring tracking_class='{requested_tracking_class}'. "
                f"This node always tracks '{self.tracking_class}'."
            )

        requested_open_vocab = bool(self.get_parameter("enable_open_vocab_prompt").value)
        if requested_open_vocab:
            self.get_logger().warn(
                "Ignoring enable_open_vocab_prompt=true. "
                "Open-vocabulary prompting is disabled in YOLO11 human-only mode."
            )
        self.enable_open_vocab_prompt = False
        self.patch_torchvision_nms_cpu_fallback = bool(
            self.get_parameter("patch_torchvision_nms_cpu_fallback").value
        )
        self.prefer_ultralytics_torch_nms = bool(
            self.get_parameter("prefer_ultralytics_torch_nms").value
        )

        self.max_age = int(self.get_parameter("max_age").value)
        self.n_init = int(self.get_parameter("n_init").value)
        self.max_cosine_distance = float(self.get_parameter("max_cosine_distance").value)
        self.nn_budget = int(self.get_parameter("nn_budget").value)
        self.nms_max_overlap = float(self.get_parameter("nms_max_overlap").value)
        self.reid_embedder = str(self.get_parameter("reid_embedder").value).strip()
        self.use_torchreid_embedder = bool(self.get_parameter("use_torchreid_embedder").value)
        self.torchreid_model_name = (
            str(self.get_parameter("torchreid_model_name").value).strip() or "osnet_ain_x0_5"
        )
        self.torchreid_embedder_gpu = bool(self.get_parameter("torchreid_embedder_gpu").value)
        self.clip_model_name = str(self.get_parameter("clip_model_name").value).strip() or "ViT-B/16"
        self.embedder_weights_path = str(self.get_parameter("embedder_weights_path").value).strip()
        self.fallback_to_hsv_embedder = bool(self.get_parameter("fallback_to_hsv_embedder").value)

        self.default_publish_rate_hz = max(0.0, float(self.get_parameter("publish_rate_hz").value))
        self.diagnostics_period_s = max(0.2, float(self.get_parameter("diagnostics_period_s").value))

        self.debug_image_dir = pathlib.Path(str(self.get_parameter("debug_image_dir").value)).expanduser().resolve()
        self.debug_save_interval_s = max(0.05, float(self.get_parameter("debug_save_interval_s").value))
        self.debug_image_dir.mkdir(parents=True, exist_ok=True)
        self.enable_ui = bool(self.get_parameter("enable_ui").value)
        self.ui_window_name = str(self.get_parameter("ui_window_name").value).strip() or "DeepSORT Tracking"
        self.ui_show_depth_text = bool(self.get_parameter("ui_show_depth_text").value)

        self.depth_window_size = max(1, int(self.get_parameter("depth_window_size").value))
        self.min_depth_m = float(self.get_parameter("min_depth_m").value)
        self.max_depth_m = float(self.get_parameter("max_depth_m").value)
        self.max_depth_age_ms = max(0.0, float(self.get_parameter("max_depth_age_ms").value))

        self.lost_hold_sec = max(0.0, float(self.get_parameter("lost_hold_sec").value))
        self.reacquire_max_dist_m = max(0.0, float(self.get_parameter("reacquire_max_dist_m").value))
        self.reacquire_max_vel_mps = max(0.0, float(self.get_parameter("reacquire_max_vel_mps").value))
        self.motion_weight = max(0.0, float(self.get_parameter("motion_weight").value))

        self._bridge = CvBridge()
        self._tf_broadcaster = TransformBroadcaster(self)

        self._data_lock = threading.Lock()
        self._state_lock = threading.Lock()

        self._latest_color_image: np.ndarray | None = None
        self._latest_color_stamp: Any = None
        self._latest_color_frame_id = "camera_color_optical_frame"
        self._latest_color_seq = -1

        self._latest_depth_image: np.ndarray | None = None
        self._latest_depth_stamp: Any = None
        self._latest_depth_encoding = ""
        self._latest_camera_info: CameraInfo | None = None

        self._running = False
        self._save_debug_images = False
        self._publish_rate_hz = self.default_publish_rate_hz

        self._last_publish_s = 0.0
        self._last_debug_save_s = 0.0
        self._last_error_log_s = 0.0

        self._processed_frames = 0
        self._published_frames = 0
        self._dropped_frames = 0
        self._active_tracks = 0
        self._last_yolo_detections = 0
        self._last_inference_ms = 0.0

        self._diag_last_time_s = time.monotonic()
        self._diag_last_processed = 0
        self._diag_last_published = 0

        self._kinematics: dict[int, tuple[np.ndarray, float, float]] = {}
        self._follow_track_id: int | None = None
        self._last_follow_pose: PoseStamped | None = None
        self._last_follow_seen_s = 0.0
        self._lost_events = 0
        self._reacquire_events = 0

        self._YOLO_cls: Any = None
        self._DeepSort_cls: Any = None
        self._torch_module: Any = None
        self._detector: Any = None
        self._tracker: Any = None
        self._runtime_error = ""
        self._device = "cpu"
        self._person_class_ids: list[int] = []
        self._assume_person_is_class_zero = False
        self._tracker_uses_internal_embedder = False
        self._tracker_embedder_label = "hsv_histogram"
        self._ui_enabled = self.enable_ui
        self._ui_window_created = False
        self._ui_lock = threading.Lock()
        self._ui_latest_frame: np.ndarray | None = None
        self._ui_timer: Any = None

        self._tracks_2d_pub = self.create_publisher(PeopleTrack2DArray, self.people_tracks_2d_topic, 10)
        self._tracks_3d_pub = self.create_publisher(PeopleTrack3DArray, self.people_tracks_3d_topic, 10)
        self._compat_pub = self.create_publisher(Detection3DArray, self.tracking_compat_topic, 10)
        self._follow_pose_pub = self.create_publisher(PoseStamped, self.follow_pose_topic, 10)

        self.create_subscription(Image, self.color_topic, self._on_color_image, qos_profile_sensor_data)
        self.create_subscription(Image, self.depth_topic, self._on_depth_image, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, self.camera_info_topic, self._on_camera_info, qos_profile_sensor_data)

        self.create_service(SetTracking, self.tracking_service_name, self._handle_tracking_service)

        self.create_timer(self.diagnostics_period_s, self._diagnostics_tick)

        self._init_backends()
        self._init_ui_window()
        if self._ui_enabled and self._ui_window_created:
            # Keep OpenCV event processing on executor thread (not worker thread).
            self._ui_timer = self.create_timer(1.0 / 30.0, self._ui_tick)

        self._worker_stop = threading.Event()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

        self.get_logger().info(f"Tracking service: {self.tracking_service_name}")
        self.get_logger().info(f"Tracks 2D topic: {self.people_tracks_2d_topic}")
        self.get_logger().info(f"Tracks 3D topic: {self.people_tracks_3d_topic}")
        self.get_logger().info(f"Compat topic: {self.tracking_compat_topic}")
        self.get_logger().info(f"Follow pose topic: {self.follow_pose_topic}")
        self.get_logger().info(f"Model path: {self.model_path}")
        self.get_logger().info(f"UI window enabled: {str(self._ui_enabled).lower()}")

    def destroy_node(self) -> bool:
        self._worker_stop.set()
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
        self._close_ui_window()
        return super().destroy_node()

    def _init_ui_window(self) -> None:
        if not self._ui_enabled:
            return

        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        if not has_display:
            self.get_logger().warn(
                "enable_ui=true but no DISPLAY/WAYLAND_DISPLAY is set; disabling tracking window."
            )
            self._ui_enabled = False
            return

        try:
            cv2.namedWindow(self.ui_window_name, cv2.WINDOW_NORMAL)
            try:
                cv2.startWindowThread()
            except Exception:
                pass
            self._ui_window_created = True
        except cv2.error as exc:
            self.get_logger().warn(f"Could not create tracking window '{self.ui_window_name}': {exc}")
            self._ui_enabled = False
            self._ui_window_created = False

    def _close_ui_window(self) -> None:
        if not self._ui_window_created:
            return

        try:
            cv2.destroyWindow(self.ui_window_name)
        except cv2.error:
            pass
        finally:
            self._ui_window_created = False
            self._ui_enabled = False
            with self._ui_lock:
                self._ui_latest_frame = None

    def _init_backends(self) -> None:
        self._runtime_error = ""
        try:
            inject_known_venv_site_packages()
            ensure_torch_runtime_libs()
            preload_cupti_if_needed()

            import torch
            from deep_sort_realtime.deepsort_tracker import DeepSort
            from ultralytics import YOLO

            self._torch_module = torch
            self._YOLO_cls = YOLO
            self._DeepSort_cls = DeepSort
            self._device = self._choose_device(self.device_request, torch)

            started = time.perf_counter()
            self._detector = YOLO(self.model_path)
            load_ms = (time.perf_counter() - started) * 1000.0

            self._tracker = self._create_tracker_instance()

            if self.patch_torchvision_nms_cpu_fallback:
                patch_torchvision_nms_fallback(
                    self.get_logger(), prefer_ultralytics_torch_nms=self.prefer_ultralytics_torch_nms
                )

            self._person_class_ids = self._resolve_class_ids(self._detector.names, self.tracking_class)
            self._assume_person_is_class_zero = False
            if self._person_class_ids:
                self.get_logger().info(f"Tracking class ids for '{self.tracking_class}': {self._person_class_ids}")
            else:
                # Some YOLO11 exports can report names in a way that fails direct text mapping.
                # In that case, keep all classes at predict-time and accept cls_id==0 at runtime.
                self._assume_person_is_class_zero = True
                self.get_logger().warn(
                    f"Could not map class '{self.tracking_class}' from model names. "
                    "Will fallback to runtime filtering (class text or cls_id==0)."
                )

            self.get_logger().info(f"Loaded detector in {load_ms:.1f} ms")
            self.get_logger().info(f"Torch CUDA available: {torch.cuda.is_available()}")
            self.get_logger().info(f"Detector device: {self._device}")
            self.get_logger().info(f"Tracker embedder: {self._tracker_embedder_label}")
        except Exception as exc:
            self._detector = None
            self._tracker = None
            self._runtime_error = f"Failed to initialize detector/tracker: {exc}"
            self.get_logger().error(self._runtime_error)

    def _reset_tracker(self) -> None:
        if self._DeepSort_cls is None:
            return
        self._tracker = self._create_tracker_instance()

    @staticmethod
    def _normalize_reid_embedder_name(raw_name: str) -> str:
        name = raw_name.strip().lower()
        if name in ("", "auto"):
            return "auto"
        if name in ("none", "hsv", "external", "manual", "disabled"):
            return "hsv"
        if name in ("torchreid", "torch-reid"):
            return "torchreid"
        if name in ("clip", "openai-clip", "openai_clip"):
            return "clip"
        if name in ("mobilenet", "mobilenetv2", "mobilenet_v2", "mobile"):
            return "mobilenet"
        return name

    @staticmethod
    def _normalize_clip_model_name(raw_name: str) -> str:
        name = raw_name.strip()
        if name.startswith("clip_"):
            name = name[len("clip_") :]
        key = name.lower()
        aliases = {
            "rn50": "RN50",
            "rn101": "RN101",
            "rn50x4": "RN50x4",
            "rn50x16": "RN50x16",
            "vit-b/32": "ViT-B/32",
            "vit-b/16": "ViT-B/16",
        }
        return aliases.get(key, name)

    def _create_tracker_instance(self) -> Any:
        if self._DeepSort_cls is None:
            raise RuntimeError("DeepSort class is not available.")

        common_kwargs: dict[str, Any] = {
            "max_age": self.max_age,
            "n_init": self.n_init,
            "nms_max_overlap": self.nms_max_overlap,
            "max_cosine_distance": self.max_cosine_distance,
            "nn_budget": self.nn_budget,
        }

        self._tracker_uses_internal_embedder = False
        self._tracker_embedder_label = "hsv_histogram"

        backend = self._normalize_reid_embedder_name(self.reid_embedder)
        if backend == "auto":
            backend = "torchreid" if self.use_torchreid_embedder else "hsv"

        embedder_gpu = bool(self.torchreid_embedder_gpu and self._device != "cpu")

        if backend == "torchreid":
            try:
                embedder_kwargs: dict[str, Any] = {
                    "embedder": "torchreid",
                    "bgr": True,
                    "embedder_gpu": embedder_gpu,
                    "embedder_model_name": self.torchreid_model_name,
                }
                if self.embedder_weights_path:
                    embedder_kwargs["embedder_wts"] = self.embedder_weights_path
                tracker = self._DeepSort_cls(**common_kwargs, **embedder_kwargs)
                self._tracker_uses_internal_embedder = True
                if self.embedder_weights_path:
                    model_file = pathlib.Path(self.embedder_weights_path).name
                    self._tracker_embedder_label = (
                        f"torchreid:{self.torchreid_model_name} ({model_file})"
                    )
                else:
                    self._tracker_embedder_label = f"torchreid:{self.torchreid_model_name}"
                return tracker
            except Exception as exc:
                if not self.fallback_to_hsv_embedder:
                    raise
                self.get_logger().warn(
                    f"torchreid embedder init failed ({exc}); "
                    "falling back to lightweight HSV embedder."
                )
        elif backend == "clip":
            clip_model_name = self._normalize_clip_model_name(self.clip_model_name)
            embedder_name = f"clip_{clip_model_name}"
            try:
                embedder_kwargs = {
                    "embedder": embedder_name,
                    "bgr": True,
                    "embedder_gpu": embedder_gpu,
                }
                if self.embedder_weights_path:
                    embedder_kwargs["embedder_wts"] = self.embedder_weights_path
                tracker = self._DeepSort_cls(**common_kwargs, **embedder_kwargs)
                self._tracker_uses_internal_embedder = True
                self._tracker_embedder_label = embedder_name
                return tracker
            except Exception as exc:
                if not self.fallback_to_hsv_embedder:
                    raise
                self.get_logger().warn(
                    f"CLIP embedder init failed ({exc}); "
                    "falling back to lightweight HSV embedder."
                )
        elif backend == "mobilenet":
            try:
                tracker = self._DeepSort_cls(
                    **common_kwargs,
                    embedder="mobilenet",
                    bgr=True,
                    embedder_gpu=embedder_gpu,
                )
                self._tracker_uses_internal_embedder = True
                self._tracker_embedder_label = "mobilenetv2"
                return tracker
            except Exception as exc:
                if not self.fallback_to_hsv_embedder:
                    raise
                self.get_logger().warn(
                    f"mobilenet embedder init failed ({exc}); "
                    "falling back to lightweight HSV embedder."
                )
        elif backend != "hsv":
            if not self.fallback_to_hsv_embedder:
                raise ValueError(
                    f"Unknown reid_embedder '{self.reid_embedder}'. "
                    "Use one of: auto, torchreid, clip, mobilenet, hsv."
                )
            self.get_logger().warn(
                f"Unknown reid_embedder '{self.reid_embedder}'. "
                "Falling back to lightweight HSV embedder."
            )

        return self._DeepSort_cls(**common_kwargs, embedder=None)

    @staticmethod
    def _choose_device(requested: str, torch_module: Any) -> str:
        req = requested.strip().lower()
        if req in ("cpu", "cuda", "cuda:0"):
            return req
        return "cuda:0" if torch_module.cuda.is_available() else "cpu"

    @staticmethod
    def _resolve_class_ids(names: Any, class_name: str) -> list[int]:
        target = class_name.strip().lower()
        ids: list[int] = []
        if isinstance(names, dict):
            for class_id, name in names.items():
                if str(name).strip().lower() == target:
                    ids.append(int(class_id))
        elif isinstance(names, list):
            for class_id, name in enumerate(names):
                if str(name).strip().lower() == target:
                    ids.append(class_id)
        return ids

    @staticmethod
    def _class_name(names: Any, class_id: int) -> str:
        if isinstance(names, dict):
            return str(names.get(class_id, class_id))
        if isinstance(names, list) and 0 <= class_id < len(names):
            return str(names[class_id])
        return str(class_id)

    def _on_color_image(self, msg: Image) -> None:
        try:
            image = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            self.get_logger().warn(f"Failed to convert color image: {exc}")
            return

        with self._data_lock:
            self._latest_color_image = image
            self._latest_color_stamp = msg.header.stamp
            self._latest_color_frame_id = msg.header.frame_id or "camera_color_optical_frame"
            self._latest_color_seq += 1

    def _on_depth_image(self, msg: Image) -> None:
        try:
            depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except CvBridgeError as exc:
            self.get_logger().warn(f"Failed to convert depth image: {exc}")
            return

        with self._data_lock:
            self._latest_depth_image = depth
            self._latest_depth_stamp = msg.header.stamp
            self._latest_depth_encoding = msg.encoding

    def _on_camera_info(self, msg: CameraInfo) -> None:
        with self._data_lock:
            self._latest_camera_info = msg

    def _handle_tracking_service(
        self, request: SetTracking.Request, response: SetTracking.Response
    ) -> SetTracking.Response:
        response.success = False
        response.running = False
        response.tracking_class = self.tracking_class

        if request.enable:
            if self._detector is None or self._tracker is None:
                self._init_backends()
            if self._detector is None or self._tracker is None:
                response.message = self._runtime_error or "Tracker backend is not available."
                response.running = False
                return response

            with self._state_lock:
                self._publish_rate_hz = float(request.rate_hz) if request.rate_hz > 0.0 else self.default_publish_rate_hz
                self._save_debug_images = bool(request.save_image)
                self._running = True
                self._last_publish_s = 0.0

            response.success = True
            response.running = True
            response.message = (
                f"DeepSORT human tracking enabled for class '{self.tracking_class}'. "
                f"publish_rate_hz={self._publish_rate_hz:.2f}, save_image={self._save_debug_images}"
            )
            return response

        with self._state_lock:
            self._running = False
            self._save_debug_images = False
            self._last_publish_s = 0.0

        self._reset_tracker()
        self._follow_track_id = None
        self._last_follow_pose = None
        self._kinematics.clear()

        response.success = True
        response.running = False
        response.message = "DeepSORT tracking disabled."
        return response

    def _worker_loop(self) -> None:
        last_seq = -1
        while rclpy.ok() and not self._worker_stop.is_set():
            with self._state_lock:
                running = self._running
                publish_rate_hz = self._publish_rate_hz
                last_publish_s = self._last_publish_s

            if not running:
                time.sleep(0.02)
                continue

            with self._data_lock:
                frame = None if self._latest_color_image is None else self._latest_color_image.copy()
                frame_stamp = self._latest_color_stamp
                frame_id = self._latest_color_frame_id
                frame_seq = self._latest_color_seq

            if frame is None:
                time.sleep(0.01)
                continue

            if frame_seq == last_seq:
                time.sleep(0.005)
                continue

            now_s = time.monotonic()
            if publish_rate_hz > 0.0:
                period_s = 1.0 / publish_rate_hz
                if now_s - last_publish_s < period_s:
                    time.sleep(0.002)
                    continue

            try:
                self._process_frame(frame, frame_stamp, frame_id)
                with self._state_lock:
                    self._last_publish_s = time.monotonic()
            except Exception as exc:
                self._dropped_frames += 1
                log_now_s = time.monotonic()
                if log_now_s - self._last_error_log_s > 1.0:
                    self._last_error_log_s = log_now_s
                    self.get_logger().error(f"Processing error: {exc}")

            last_seq = frame_seq

    def _process_frame(self, frame: np.ndarray, frame_stamp: Any, frame_id: str) -> None:
        if self._detector is None or self._tracker is None:
            raise RuntimeError("Detector/tracker backend is not initialized.")

        infer_started = time.perf_counter()
        predict_kwargs: dict[str, Any] = {
            "source": frame,
            "device": self._device,
            "imgsz": self.imgsz,
            "conf": self.det_conf,
            "iou": self.det_iou,
            "max_det": self.max_det,
            "verbose": False,
            "save": False,
        }
        if self._person_class_ids:
            predict_kwargs["classes"] = self._person_class_ids

        results = self._detector.predict(**predict_kwargs)
        inference_ms = float((time.perf_counter() - infer_started) * 1000.0)
        self._last_inference_ms = inference_ms

        raw_detections: list[tuple[list[float], float, str]] = []
        yolo_detections: list[Detection2DRecord] = []
        embeddings: list[np.ndarray] = []

        image_h, image_w = frame.shape[:2]

        if results:
            result = results[0]
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    cls_id = int(box.cls.item())
                    class_name = self._class_name(result.names, cls_id).strip().lower()
                    if self._person_class_ids:
                        is_person = cls_id in self._person_class_ids
                    else:
                        is_person = class_name == self.tracking_class
                        if not is_person and self._assume_person_is_class_zero and cls_id == 0:
                            is_person = True

                    if not is_person:
                        continue

                    conf = float(box.conf.item())
                    x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
                    bx, by, bw, bh = clamp_box(x1, y1, x2, y2, image_w, image_h)
                    if bw <= 1 or bh <= 1:
                        continue

                    det_class_name = self.tracking_class
                    raw_detections.append(([float(bx), float(by), float(bw), float(bh)], conf, det_class_name))
                    yolo_detections.append(
                        Detection2DRecord(
                            class_name=det_class_name,
                            confidence=conf,
                            x=bx,
                            y=by,
                            width=bw,
                            height=bh,
                        )
                    )
                    if not self._tracker_uses_internal_embedder:
                        crop = frame[by : by + bh, bx : bx + bw]
                        embeddings.append(self._compute_embedding(crop))

        if self._tracker_uses_internal_embedder:
            # Internal DeepSORT embedders (torchreid/clip/mobilenet) compute features from frame.
            tracks = self._tracker.update_tracks(raw_detections, frame=frame)
        else:
            # With embedder=None, DeepSORT requires `embeds` argument to always be present,
            # even when there are no detections in the current frame.
            tracks = self._tracker.update_tracks(raw_detections, embeds=embeddings)

        tracks2d: list[Track2DRecord] = []
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue

            left, top, right, bottom = track.to_ltrb(orig=True)
            x, y, w, h = clamp_box(left, top, right, bottom, image_w, image_h)
            class_name = str(track.get_det_class() or self.tracking_class)
            det_conf = float(track.get_det_conf() or 0.0)
            track_id = int(track.track_id)
            reassociated = bool(track.age > track.hits)

            tracks2d.append(
                Track2DRecord(
                    track_id=track_id,
                    class_name=class_name,
                    detector_confidence=det_conf,
                    tracker_confidence=det_conf,
                    x=x,
                    y=y,
                    width=w,
                    height=h,
                    reassociated=reassociated,
                )
            )

        if frame_stamp is None:
            stamp_msg = self.get_clock().now().to_msg()
        else:
            stamp_msg = frame_stamp

        tracks_2d_msg = PeopleTrack2DArray()
        tracks_2d_msg.header.stamp = stamp_msg
        tracks_2d_msg.header.frame_id = frame_id
        tracks_2d_msg.image_width = image_w
        tracks_2d_msg.image_height = image_h
        for rec in tracks2d:
            msg = PeopleTrack2D()
            msg.track_id = rec.track_id
            msg.class_name = rec.class_name
            msg.detector_confidence = rec.detector_confidence
            msg.tracker_confidence = rec.tracker_confidence
            msg.x = rec.x
            msg.y = rec.y
            msg.width = rec.width
            msg.height = rec.height
            msg.reassociated = rec.reassociated
            tracks_2d_msg.tracks.append(msg)
        self._tracks_2d_pub.publish(tracks_2d_msg)

        now_s = stamp_to_seconds(stamp_msg)

        with self._data_lock:
            depth_image = None if self._latest_depth_image is None else self._latest_depth_image.copy()
            depth_stamp = self._latest_depth_stamp
            depth_encoding = self._latest_depth_encoding
            camera_info = self._latest_camera_info

        depth_valid = depth_image is not None and camera_info is not None and depth_stamp is not None
        if depth_valid:
            depth_age_ms = (now_s - stamp_to_seconds(depth_stamp)) * 1000.0
            if depth_age_ms > self.max_depth_age_ms:
                depth_valid = False

        tracks_3d_msg = PeopleTrack3DArray()
        tracks_3d_msg.header.stamp = stamp_msg
        tracks_3d_msg.header.frame_id = self.camera_link_frame

        compat_msg = Detection3DArray()
        compat_msg.header.stamp = stamp_msg
        compat_msg.header.frame_id = self.camera_link_frame
        compat_msg.prompt_text = self.tracking_class
        compat_msg.detections_in_frame = len(tracks2d)
        compat_msg.inference_ms = inference_ms

        valid_tracks: dict[int, ValidTrack3D] = {}

        fx = fy = cx = cy = 0.0
        if depth_valid and camera_info is not None:
            fx = float(camera_info.k[0])
            fy = float(camera_info.k[4])
            cx = float(camera_info.k[2])
            cy = float(camera_info.k[5])
            if fx <= 0.0 or fy <= 0.0:
                depth_valid = False

        for rec in tracks2d:
            out = PeopleTrack3D()
            out.track_id = rec.track_id
            out.class_name = rec.class_name
            out.detector_confidence = rec.detector_confidence
            out.tracker_confidence = rec.tracker_confidence
            out.pose_camera_link.header.stamp = stamp_msg
            out.pose_camera_link.header.frame_id = self.camera_link_frame
            out.pose_camera_link.pose.orientation.w = 1.0
            out.depth_m = 0.0
            out.valid_depth = False
            out.tf_child_frame = f"{self.tf_prefix}_{rec.track_id}"

            if depth_valid and depth_image is not None:
                u = rec.x + rec.width // 2
                v = rec.y + rec.height // 2
                depth_m = self._sample_depth_meters(depth_image, depth_encoding, u, v)
                if depth_m is not None:
                    x_m = ((float(u) - cx) / fx) * depth_m
                    y_m = ((float(v) - cy) / fy) * depth_m
                    z_m = depth_m

                    out.pose_camera_link.pose.position.x = float(x_m)
                    out.pose_camera_link.pose.position.y = float(y_m)
                    out.pose_camera_link.pose.position.z = float(z_m)
                    out.depth_m = float(z_m)
                    out.valid_depth = True

                    self._publish_track_tf(out.tf_child_frame, out.pose_camera_link)

                    det = Detection3D()
                    det.class_name = out.class_name
                    det.confidence = out.detector_confidence
                    det.pose_camera_link = out.pose_camera_link
                    det.tf_child_frame = out.tf_child_frame
                    compat_msg.detections.append(det)

                    speed_mps = self._update_kinematics(rec.track_id, out.pose_camera_link, now_s)
                    valid_tracks[rec.track_id] = ValidTrack3D(
                        track_id=rec.track_id,
                        pose=out.pose_camera_link,
                        speed_mps=speed_mps,
                    )

            tracks_3d_msg.tracks.append(out)

        compat_msg.tf_published_count = len(compat_msg.detections)
        self._tracks_3d_pub.publish(tracks_3d_msg)
        self._compat_pub.publish(compat_msg)

        self._update_follow_target(valid_tracks, stamp_msg, now_s)
        self._prune_kinematics(now_s)
        self._render_tracking_window(
            frame,
            yolo_detections,
            tracks2d,
            valid_tracks,
            self._follow_track_id,
            inference_ms,
        )

        with self._state_lock:
            save_debug = self._save_debug_images

        if save_debug and (time.monotonic() - self._last_debug_save_s) >= self.debug_save_interval_s:
            self._save_debug_frame(frame, tracks2d)
            self._last_debug_save_s = time.monotonic()

        self._processed_frames += 1
        self._published_frames += 1
        self._active_tracks = len(tracks2d)
        self._last_yolo_detections = len(yolo_detections)

    @staticmethod
    def _compute_embedding(crop: np.ndarray) -> np.ndarray:
        if crop.size == 0:
            return np.zeros((48,), dtype=np.float32)

        resized = cv2.resize(crop, (64, 128), interpolation=cv2.INTER_LINEAR)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

        # Lightweight appearance vector for DeepSORT association.
        h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
        s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
        v_hist = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()

        feat = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)
        norm = float(np.linalg.norm(feat))
        if norm > 1e-6:
            feat /= norm
        return feat

    def _sample_depth_meters(
        self, depth_image: np.ndarray, encoding: str, u: int, v: int
    ) -> float | None:
        if depth_image.ndim > 2:
            depth_image = depth_image[:, :, 0]

        if u < 0 or v < 0 or u >= depth_image.shape[1] or v >= depth_image.shape[0]:
            return None

        radius = max(0, self.depth_window_size // 2)
        values: list[float] = []

        for yy in range(max(0, v - radius), min(depth_image.shape[0], v + radius + 1)):
            for xx in range(max(0, u - radius), min(depth_image.shape[1], u + radius + 1)):
                meters = self._depth_value_to_meters(depth_image[yy, xx], encoding, depth_image.dtype)
                if meters is None:
                    continue
                if meters < self.min_depth_m or meters > self.max_depth_m:
                    continue
                values.append(meters)

        if not values:
            return None

        return float(np.median(values))

    @staticmethod
    def _depth_value_to_meters(value: Any, encoding: str, dtype: np.dtype[Any]) -> float | None:
        depth_value = float(value)
        if not math.isfinite(depth_value) or depth_value <= 0.0:
            return None

        enc = encoding.upper() if encoding else ""
        if "16U" in enc or "MONO16" in enc:
            return depth_value * 0.001
        if "32F" in enc:
            return depth_value

        if dtype in (np.uint16, np.int16):
            return depth_value * 0.001
        if dtype in (np.float32, np.float64):
            return depth_value

        return None

    def _update_kinematics(self, track_id: int, pose: PoseStamped, now_s: float) -> float:
        point = np.array(
            [
                float(pose.pose.position.x),
                float(pose.pose.position.y),
                float(pose.pose.position.z),
            ],
            dtype=np.float64,
        )

        state = self._kinematics.get(track_id)
        if state is None:
            self._kinematics[track_id] = (point, now_s, 0.0)
            return 0.0

        last_point, last_s, _ = state
        dt = now_s - last_s
        speed = 0.0
        if dt > 1e-3:
            speed = float(np.linalg.norm(point - last_point) / dt)

        self._kinematics[track_id] = (point, now_s, speed)
        return speed

    def _prune_kinematics(self, now_s: float) -> None:
        stale_ids = [track_id for track_id, (_, stamp_s, _) in self._kinematics.items() if now_s - stamp_s > 5.0]
        for track_id in stale_ids:
            self._kinematics.pop(track_id, None)

    def _update_follow_target(
        self,
        valid_tracks: dict[int, ValidTrack3D],
        stamp_msg: Any,
        now_s: float,
    ) -> None:
        published = False

        if self._follow_track_id is not None:
            current = valid_tracks.get(self._follow_track_id)
            if current is not None:
                self._publish_follow_pose(current.pose, stamp_msg)
                self._last_follow_pose = current.pose
                self._last_follow_seen_s = now_s
                published = True
            else:
                self._follow_track_id = None
                self._lost_events += 1

        if not published and self._last_follow_pose is not None:
            if now_s - self._last_follow_seen_s <= self.lost_hold_sec:
                held_pose = PoseStamped()
                held_pose.header.stamp = stamp_msg
                held_pose.header.frame_id = self.camera_link_frame
                held_pose.pose = self._last_follow_pose.pose
                self._publish_follow_pose(held_pose, stamp_msg)
                published = True

        if not published and valid_tracks:
            best: ValidTrack3D | None = None
            best_score = float("inf")

            for candidate in valid_tracks.values():
                if self._last_follow_pose is not None:
                    dist = point_distance(candidate.pose, self._last_follow_pose)
                    if dist > self.reacquire_max_dist_m:
                        continue
                    if candidate.speed_mps > self.reacquire_max_vel_mps:
                        continue
                else:
                    dist = max(0.0, float(candidate.pose.pose.position.z))

                score = dist + self.motion_weight * candidate.speed_mps
                if score < best_score:
                    best_score = score
                    best = candidate

            if best is not None:
                self._follow_track_id = best.track_id
                self._publish_follow_pose(best.pose, stamp_msg)
                self._last_follow_pose = best.pose
                self._last_follow_seen_s = now_s
                self._reacquire_events += 1

    def _publish_track_tf(self, child_frame: str, pose: PoseStamped) -> None:
        tf_msg = TransformStamped()
        tf_msg.header.stamp = pose.header.stamp
        tf_msg.header.frame_id = self.camera_link_frame
        tf_msg.child_frame_id = child_frame
        tf_msg.transform.translation.x = float(pose.pose.position.x)
        tf_msg.transform.translation.y = float(pose.pose.position.y)
        tf_msg.transform.translation.z = float(pose.pose.position.z)
        tf_msg.transform.rotation.w = 1.0
        self._tf_broadcaster.sendTransform(tf_msg)

    def _publish_follow_pose(self, pose: PoseStamped, stamp_msg: Any) -> None:
        follow_pose = PoseStamped()
        follow_pose.header.stamp = stamp_msg
        follow_pose.header.frame_id = self.camera_link_frame
        follow_pose.pose = pose.pose
        self._follow_pose_pub.publish(follow_pose)

        tf_msg = TransformStamped()
        tf_msg.header.stamp = stamp_msg
        tf_msg.header.frame_id = self.camera_link_frame
        tf_msg.child_frame_id = "follow_target"
        tf_msg.transform.translation.x = float(follow_pose.pose.position.x)
        tf_msg.transform.translation.y = float(follow_pose.pose.position.y)
        tf_msg.transform.translation.z = float(follow_pose.pose.position.z)
        tf_msg.transform.rotation.w = 1.0
        self._tf_broadcaster.sendTransform(tf_msg)

    def _save_debug_frame(self, frame: np.ndarray, tracks2d: list[Track2DRecord]) -> None:
        annotated = frame.copy()
        for track in tracks2d:
            x1 = track.x
            y1 = track.y
            x2 = track.x + track.width
            y2 = track.y + track.height
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 0), 2)
            label = f"id={track.track_id} conf={track.detector_confidence:.2f}"
            cv2.putText(annotated, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"deepsort_{ts}.jpg"
        path = self.debug_image_dir / filename
        cv2.imwrite(str(path), annotated)

    def _render_tracking_window(
        self,
        frame: np.ndarray,
        yolo_detections: list[Detection2DRecord],
        tracks2d: list[Track2DRecord],
        valid_tracks: dict[int, ValidTrack3D],
        follow_track_id: int | None,
        inference_ms: float,
    ) -> None:
        if not self._ui_enabled or not self._ui_window_created:
            return

        annotated = frame.copy()

        for det in yolo_detections:
            x1 = det.x
            y1 = det.y
            x2 = det.x + det.width
            y2 = det.y + det.height
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 180, 0), 1)
            det_label = f"det {det.confidence:.2f}"
            cv2.putText(
                annotated,
                det_label,
                (x1, max(0, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 180, 0),
                1,
                cv2.LINE_AA,
            )

        for track in tracks2d:
            x1 = track.x
            y1 = track.y
            x2 = track.x + track.width
            y2 = track.y + track.height

            is_follow = follow_track_id is not None and track.track_id == follow_track_id
            color = (0, 0, 255) if is_follow else ((0, 165, 255) if track.reassociated else (0, 200, 0))
            thickness = 3 if is_follow else 2
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            label = f"id={track.track_id} conf={track.detector_confidence:.2f}"
            if self.ui_show_depth_text:
                valid = valid_tracks.get(track.track_id)
                if valid is not None:
                    label += f" z={float(valid.pose.pose.position.z):.2f}m"
            cv2.putText(
                annotated,
                label,
                (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        follow_text = "none" if follow_track_id is None else str(follow_track_id)
        status = (
            f"dets={len(yolo_detections)} tracks={len(tracks2d)} "
            f"follow={follow_text} infer={inference_ms:.1f}ms"
        )
        cv2.putText(
            annotated,
            status,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            status,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        with self._ui_lock:
            self._ui_latest_frame = annotated

    def _ui_tick(self) -> None:
        if not self._ui_enabled or not self._ui_window_created:
            return

        with self._ui_lock:
            frame = None if self._ui_latest_frame is None else self._ui_latest_frame.copy()

        if frame is None:
            with self._data_lock:
                frame = None if self._latest_color_image is None else self._latest_color_image.copy()

            if frame is None:
                # Keep UI responsive even before first camera frame arrives.
                frame = np.zeros((480, 848, 3), dtype=np.uint8)
                cv2.putText(
                    frame,
                    f"Waiting for camera stream: {self.color_topic}",
                    (18, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            else:
                with self._state_lock:
                    running = self._running
                status = "tracking=on (no overlays yet)" if running else "tracking=off (preview only)"
                cv2.putText(
                    frame,
                    status,
                    (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        try:
            cv2.imshow(self.ui_window_name, frame)
            cv2.waitKey(1)
            visible = cv2.getWindowProperty(self.ui_window_name, cv2.WND_PROP_VISIBLE)
            if visible < 1.0:
                self.get_logger().info("Tracking window closed by user; disabling UI rendering.")
                self._ui_window_created = False
                self._ui_enabled = False
        except cv2.error as exc:
            self.get_logger().warn(f"Tracking UI render failed; disabling UI rendering: {exc}")
            self._ui_window_created = False
            self._ui_enabled = False
            with self._ui_lock:
                self._ui_latest_frame = None

    def _diagnostics_tick(self) -> None:
        now_s = time.monotonic()
        elapsed_s = max(1e-6, now_s - self._diag_last_time_s)

        processed_delta = self._processed_frames - self._diag_last_processed
        published_delta = self._published_frames - self._diag_last_published

        proc_fps = float(processed_delta) / elapsed_s
        pub_fps = float(published_delta) / elapsed_s

        self._diag_last_time_s = now_s
        self._diag_last_processed = self._processed_frames
        self._diag_last_published = self._published_frames

        with self._state_lock:
            running = self._running

        follow_id = self._follow_track_id if self._follow_track_id is not None else -1
        self.get_logger().info(
            f"running={str(running).lower()} yolo_dets={self._last_yolo_detections} "
            f"active_tracks={self._active_tracks} "
            f"proc_fps={proc_fps:.2f} pub_fps={pub_fps:.2f} infer_ms={self._last_inference_ms:.1f} "
            f"follow_id={follow_id} lost_events={self._lost_events} "
            f"reacquire_events={self._reacquire_events}"
        )


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = DeepSortPeopleFollowNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

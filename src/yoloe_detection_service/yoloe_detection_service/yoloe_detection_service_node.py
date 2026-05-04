#!/usr/bin/env python3
"""YOLOE text-prompt detection service node for multiple camera streams.

This node loads YOLOE once and supports single-shot detection via
service (/yoloe/detect_prompt).

Edited version:
- keeps final published TF parent frame as base_link
- improves object position estimation using segmentation-mask depth points
- falls back to box-center depth when mask-based estimation is insufficient
"""

from __future__ import annotations

import ctypes
import gc
import math
import os
import pathlib
import re
import shutil
import sys
import sysconfig
import threading
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from rclpy.wait_for_message import wait_for_message
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformBroadcaster, TransformException, TransformListener

from yoloe_detection_interfaces.srv import DetectObjectPrompt


def _iter_site_packages_dirs() -> list[pathlib.Path]:
    """Return unique site-packages directories visible to this process."""
    candidates: list[pathlib.Path] = []
    seen: set[str] = set()

    raw_paths = [sysconfig.get_paths().get("purelib", ""), *sys.path]
    for raw_path in raw_paths:
        if not raw_path:
            continue
        path = pathlib.Path(raw_path).expanduser()
        if not path.is_dir():
            continue
        try:
            normalized = str(path.resolve())
        except OSError:
            normalized = str(path)
        if normalized in seen:
            continue
        seen.add(normalized)
        candidates.append(path)
    return candidates


def ensure_torch_runtime_libs() -> None:
    """Expose CUDA libraries from pip wheels and system paths before importing torch."""
    lib_dirs: list[str] = []

    for site_packages in _iter_site_packages_dirs():
        nvidia_root = site_packages / "nvidia"
        if nvidia_root.is_dir():
            for lib_dir in nvidia_root.glob("*/lib"):
                if lib_dir.is_dir():
                    lib_dirs.append(str(lib_dir))
        torch_lib = site_packages / "torch" / "lib"
        if torch_lib.is_dir():
            lib_dirs.append(str(torch_lib))

    for path in (
        "/usr/local/cuda/lib64",
        "/usr/local/cuda-12.6/lib64",
        "/usr/local/cuda/targets/aarch64-linux/lib",
        "/usr/local/cuda-12.6/targets/aarch64-linux/lib",
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


def preload_cupti_if_needed() -> None:
    """Preload CUPTI when available (helpful on Jetson CUDA setups)."""
    candidates: list[pathlib.Path] = []
    for site_packages in _iter_site_packages_dirs():
        cupti_root = site_packages / "nvidia" / "cuda_cupti" / "lib"
        if not cupti_root.is_dir():
            continue
        candidates.extend(
            [
                cupti_root / "libcupti.so.12",
                cupti_root / "libcupti.so.11",
                cupti_root / "libcupti.so",
            ]
        )
        candidates.extend(sorted(cupti_root.glob("libcupti.so.*"), reverse=True))

    candidates.extend(
        [
            pathlib.Path("/usr/local/cuda-12.6/extras/CUPTI/lib64/libcupti.so.12"),
            pathlib.Path("/usr/local/cuda/extras/CUPTI/lib64/libcupti.so.12"),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            try:
                ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
                return
            except OSError:
                continue


_ULTRALYTICS_TEXT_ASSET_NAMES = frozenset({"mobileclip2_b.ts", "mobileclip_blt.ts"})
_ULTRALYTICS_TEXT_ASSET_CACHE_DIR = (
    pathlib.Path.home() / ".cache" / "yoloe_detection_service" / "ultralytics_assets"
)
_ULTRALYTICS_TEXT_ASSET_PATCH_LOCK = threading.Lock()


def _is_valid_torchscript_archive(path: pathlib.Path) -> bool:
    """Return True when the asset is a readable TorchScript zip archive."""
    try:
        return path.is_file() and zipfile.is_zipfile(path)
    except OSError:
        return False


def _iter_ultralytics_text_asset_candidates(filename: str) -> list[pathlib.Path]:
    """Collect likely MobileCLIP asset locations in priority order."""
    candidates: list[pathlib.Path] = []
    seen: set[str] = set()

    search_dirs: list[pathlib.Path] = [_ULTRALYTICS_TEXT_ASSET_CACHE_DIR]

    env_dir = os.environ.get("YOLOE_TEXT_ASSET_DIR", "").strip()
    if env_dir:
        search_dirs.append(pathlib.Path(env_dir).expanduser())

    search_dirs.append(pathlib.Path.cwd())
    home_dir = pathlib.Path.home()
    search_dirs.extend([home_dir, home_dir / "robocup_ws"])

    virtual_env = os.environ.get("VIRTUAL_ENV", "").strip()
    if virtual_env:
        search_dirs.append(pathlib.Path(virtual_env).expanduser())

    search_dirs.extend(pathlib.Path(__file__).resolve().parents)

    try:
        from ultralytics.utils import SETTINGS

        search_dirs.append(pathlib.Path(str(SETTINGS["weights_dir"])).expanduser())
    except Exception:
        pass

    for directory in search_dirs:
        try:
            normalized_dir = str(directory.expanduser().resolve())
        except OSError:
            normalized_dir = str(directory.expanduser())
        if normalized_dir in seen:
            continue
        seen.add(normalized_dir)
        candidates.append(pathlib.Path(normalized_dir) / filename)
    return candidates


def _resolve_ultralytics_text_asset(
    filename: str,
    download_fn: Any,
    *,
    repo: str = "ultralytics/assets",
    release: str = "v8.4.0",
    **kwargs: Any,
) -> pathlib.Path:
    """Resolve a valid MobileCLIP asset without depending on the process cwd."""
    if filename not in _ULTRALYTICS_TEXT_ASSET_NAMES:
        raise ValueError(f"Unsupported Ultralytics text asset: {filename}")

    target_path = _ULTRALYTICS_TEXT_ASSET_CACHE_DIR / filename
    if _is_valid_torchscript_archive(target_path):
        return target_path.resolve()

    for candidate in _iter_ultralytics_text_asset_candidates(filename):
        if candidate == target_path:
            continue
        if not _is_valid_torchscript_archive(candidate):
            continue

        _ULTRALYTICS_TEXT_ASSET_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if target_path.exists() or target_path.is_symlink():
            target_path.unlink()
        try:
            target_path.symlink_to(candidate.resolve())
        except OSError:
            shutil.copy2(candidate, target_path)
        return target_path.resolve()

    _ULTRALYTICS_TEXT_ASSET_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if target_path.exists() or target_path.is_symlink():
        target_path.unlink()

    downloaded_path = pathlib.Path(
        download_fn(str(target_path), repo=repo, release=release, **kwargs)
    ).expanduser()
    if not _is_valid_torchscript_archive(downloaded_path):
        raise RuntimeError(
            f"Resolved Ultralytics text asset is invalid: {downloaded_path}"
        )
    return downloaded_path.resolve()


def configure_ultralytics_text_asset_resolution(
    preload_names: tuple[str, ...] = ("mobileclip2_b.ts",),
) -> dict[str, pathlib.Path]:
    """Patch Ultralytics asset lookup so MobileCLIP files always resolve to validated paths."""
    import ultralytics.utils.downloads as downloads

    with _ULTRALYTICS_TEXT_ASSET_PATCH_LOCK:
        original_download = getattr(
            downloads.attempt_download_asset,
            "_yoloe_original_attempt_download_asset",
            downloads.attempt_download_asset,
        )

        if not getattr(downloads.attempt_download_asset, "_yoloe_text_asset_patch", False):

            def _patched_attempt_download_asset(
                file: str | pathlib.Path,
                repo: str = "ultralytics/assets",
                release: str = "v8.4.0",
                **kwargs: Any,
            ) -> str:
                cleaned = pathlib.Path(str(file).strip().replace("'", ""))
                if cleaned.name in _ULTRALYTICS_TEXT_ASSET_NAMES:
                    return str(
                        _resolve_ultralytics_text_asset(
                            cleaned.name,
                            original_download,
                            repo=repo,
                            release=release,
                            **kwargs,
                        )
                    )
                return original_download(file, repo=repo, release=release, **kwargs)

            _patched_attempt_download_asset._yoloe_text_asset_patch = True
            _patched_attempt_download_asset._yoloe_original_attempt_download_asset = (
                original_download
            )
            downloads.attempt_download_asset = _patched_attempt_download_asset

        resolved_assets: dict[str, pathlib.Path] = {}
        for name in preload_names:
            if name not in _ULTRALYTICS_TEXT_ASSET_NAMES:
                continue
            resolved_assets[name] = _resolve_ultralytics_text_asset(name, original_download)
        return resolved_assets


@dataclass
class DetectionEntry:
    class_name: str
    confidence: float
    pose_camera_link: PoseStamped
    tf_child_frame: str


@dataclass
class DetectionRunResult:
    entries: list[DetectionEntry]
    detections_in_frame: int
    tf_published_count: int
    inference_ms: float
    saved_image_path: str
    skipped_count: int
    error_message: str


@dataclass(frozen=True)
class CameraStreamConfig:
    name: str
    color_topic: str
    depth_topic: str
    camera_info_topic: str
    camera_link_frame: str


@dataclass
class CameraStreamState:
    latest_color_image: np.ndarray | None = None
    latest_depth_image: np.ndarray | None = None
    latest_depth_frame: str = ""
    latest_depth_encoding: str = ""
    latest_camera_info: CameraInfo | None = None
    latest_color_received_s: float = 0.0
    latest_depth_received_s: float = 0.0
    latest_camera_info_received_s: float = 0.0


@dataclass
class CameraSubscriptionHandles:
    color: Any
    depth: Any
    camera_info: Any


@dataclass
class PublishedTFEntry:
    parent_frame: str
    translation: np.ndarray


@dataclass
class CachedFrameTransform:
    translation: np.ndarray
    rotation_xyzw: tuple[float, float, float, float]


class YoloeDetectionServiceNode(Node):
    def __init__(self) -> None:
        super().__init__("yoloe_detection_service_node")

        self.declare_parameter("service_name", "/yoloe/detect_prompt")
        self.declare_parameter("model_path", "/home/usern/yoloe-26l-seg.pt")
        self.declare_parameter("bag_model_path", "/home/usern/Kevin_yolo/best_latest.pt")
        self.declare_parameter(
            "bag_prompt_aliases", ["bag", "paper bag", "brown paper bag", "paper bags"]
        )
        self.declare_parameter("device", "auto")
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("conf", 0.25)
        self.declare_parameter("iou", 0.45)
        self.declare_parameter("max_det", 300)
        self.declare_parameter("force_torch_nms", True)

        self.declare_parameter("default_camera_name", "camera0")
        self.declare_parameter("color_topic", "/camera0/color/image_raw")
        self.declare_parameter("depth_topic", "/camera0/realsense_splitter_node/output/depth")
        self.declare_parameter("camera_info_topic", "/camera0/color/camera_info")
        self.declare_parameter("camera_link_frame", "camera0_link")
        self.declare_parameter("camera0_color_topic", "")
        self.declare_parameter("camera0_depth_topic", "")
        self.declare_parameter("camera0_camera_info_topic", "")
        self.declare_parameter("camera0_camera_link_frame", "")
        self.declare_parameter("camera_color_topic", "/gripper_camera/color/image_raw")
        self.declare_parameter("camera_depth_topic", "/gripper_camera/depth/image_raw")
        self.declare_parameter("camera_camera_info_topic", "/gripper_camera/color/camera_info")
        self.declare_parameter("camera_camera_link_frame", "gripper_camera_link")

        self.declare_parameter("pose_topic", "/yoloe/detected_pose")
        self.declare_parameter("object_frame_prefix", "")
        self.declare_parameter("save_dir", "/home/usern/robocup_ws/yoloe_out")
        self.declare_parameter("always_save_image", False)
        self.declare_parameter("lazy_subscriptions", False)
        self.declare_parameter("subscription_idle_timeout_sec", 3.0)
        self.declare_parameter("subscription_poll_period_sec", 0.5)
        self.declare_parameter("frame_wait_timeout_sec", 1.5)
        self.declare_parameter("max_frame_age_sec", 0.75)

        self.declare_parameter("depth_window_size", 5)
        self.declare_parameter("min_depth_m", 0.05)
        self.declare_parameter("max_depth_m", 10.0)

        # Added for mask-based 3D estimation
        self.declare_parameter("mask_depth_stride", 3)
        self.declare_parameter("mask_min_valid_points", 80)
        self.declare_parameter("mask_max_points", 2500)
        self.declare_parameter("mask_erode_pixels", 2)

        self.service_name = str(self.get_parameter("service_name").value)
        self.model_path = str(self.get_parameter("model_path").value)
        self.bag_model_path = str(self.get_parameter("bag_model_path").value)
        self.bag_prompt_aliases = tuple(
            self._normalize_class_label(str(value))
            for value in self.get_parameter("bag_prompt_aliases").value
            if self._normalize_class_label(str(value))
        )
        self.device_request = str(self.get_parameter("device").value)
        self.imgsz = int(self.get_parameter("imgsz").value)
        self.conf = float(self.get_parameter("conf").value)
        self.iou = float(self.get_parameter("iou").value)
        self.max_det = int(self.get_parameter("max_det").value)
        self.force_torch_nms = bool(self.get_parameter("force_torch_nms").value)

        self.default_camera_name = self._normalize_camera_name(
            str(self.get_parameter("default_camera_name").value)
        )
        self.base_link_frame = "base_link"

        legacy_color_topic = str(self.get_parameter("color_topic").value)
        legacy_depth_topic = str(self.get_parameter("depth_topic").value)
        legacy_camera_info_topic = str(self.get_parameter("camera_info_topic").value)
        legacy_camera_link_frame = str(self.get_parameter("camera_link_frame").value)

        camera0_color_topic = self._coalesce_non_empty(
            str(self.get_parameter("camera0_color_topic").value), legacy_color_topic
        )
        camera0_depth_topic = self._coalesce_non_empty(
            str(self.get_parameter("camera0_depth_topic").value), legacy_depth_topic
        )
        camera0_camera_info_topic = self._coalesce_non_empty(
            str(self.get_parameter("camera0_camera_info_topic").value), legacy_camera_info_topic
        )
        camera0_camera_link_frame = self._coalesce_non_empty(
            str(self.get_parameter("camera0_camera_link_frame").value), legacy_camera_link_frame
        )

        self._camera_configs: dict[str, CameraStreamConfig] = {
            "camera0": CameraStreamConfig(
                name="camera0",
                color_topic=camera0_color_topic,
                depth_topic=camera0_depth_topic,
                camera_info_topic=camera0_camera_info_topic,
                camera_link_frame=camera0_camera_link_frame,
            ),
            "gripper_camera": CameraStreamConfig(
                name="gripper_camera",
                color_topic=str(self.get_parameter("camera_color_topic").value),
                depth_topic=str(self.get_parameter("camera_depth_topic").value),
                camera_info_topic=str(self.get_parameter("camera_camera_info_topic").value),
                camera_link_frame=str(self.get_parameter("camera_camera_link_frame").value),
            ),
        }
        if self.default_camera_name not in self._camera_configs:
            self.get_logger().warn(
                f"Unsupported default_camera_name '{self.default_camera_name}', falling back to camera0."
            )
            self.default_camera_name = "camera0"

        self.pose_topic = str(self.get_parameter("pose_topic").value)
        self.object_frame_prefix = str(self.get_parameter("object_frame_prefix").value)
        self.save_dir = pathlib.Path(str(self.get_parameter("save_dir").value)).expanduser().resolve()
        self.always_save_image = bool(self.get_parameter("always_save_image").value)
        requested_lazy_subscriptions = bool(self.get_parameter("lazy_subscriptions").value)
        self.lazy_subscriptions = False
        self.subscription_idle_timeout_sec = max(
            0.0, float(self.get_parameter("subscription_idle_timeout_sec").value)
        )
        self.subscription_poll_period_sec = max(
            0.1, float(self.get_parameter("subscription_poll_period_sec").value)
        )
        self.frame_wait_timeout_sec = max(
            0.1, float(self.get_parameter("frame_wait_timeout_sec").value)
        )
        self.max_frame_age_sec = max(0.05, float(self.get_parameter("max_frame_age_sec").value))

        self.depth_window_size = max(1, int(self.get_parameter("depth_window_size").value))
        self.min_depth_m = float(self.get_parameter("min_depth_m").value)
        self.max_depth_m = float(self.get_parameter("max_depth_m").value)

        self.mask_depth_stride = max(1, int(self.get_parameter("mask_depth_stride").value))
        self.mask_min_valid_points = max(10, int(self.get_parameter("mask_min_valid_points").value))
        self.mask_max_points = max(100, int(self.get_parameter("mask_max_points").value))
        self.mask_erode_pixels = max(0, int(self.get_parameter("mask_erode_pixels").value))

        self.save_dir.mkdir(parents=True, exist_ok=True)

        self._bridge = CvBridge()
        self._lock = threading.Lock()
        self._inference_lock = threading.Lock()

        self._camera_states = {
            camera_name: CameraStreamState() for camera_name in self._camera_configs
        }
        self._camera_subscriptions: dict[str, CameraSubscriptionHandles] = {}
        self._camera_subscription_deadlines: dict[str, float] = {}
        self._active_detect_requests = 0

        self._last_tf_map: dict[str, PublishedTFEntry] = {}
        self._frame_transform_cache: dict[tuple[str, str], CachedFrameTransform] = {}
        self._tf_lookup_timeout = Duration(seconds=0.2)

        self._model: Any = None
        self._prompt_key: tuple[str, ...] | None = None
        self._supports_prompt_classes = False
        self._loaded_model_path = ""
        self._loaded_model_name = ""
        self._torch: Any = None
        self._device: str = "cpu"

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._tf_broadcaster = TransformBroadcaster(self)

        self._pose_pub = self.create_publisher(PoseStamped, self.pose_topic, 10)
        self.create_service(DetectObjectPrompt, self.service_name, self._handle_detect_request)
        self.create_timer(0.2, self._publish_last_tf)

        self._prepare_runtime()
        self._load_model(self.model_path, "default")

        for camera_name, config in self._camera_configs.items():
            self._subscribe_camera_streams(camera_name, config)

        self.get_logger().info(f"YOLOE service ready on {self.service_name}")
        self.get_logger().info(f"Default model path: {self.model_path}")
        self.get_logger().info(f"Bag model path: {self.bag_model_path}")
        self.get_logger().info(f"Device: {self._device}")
        self.get_logger().info(f"Default camera: {self.default_camera_name}")
        self.get_logger().info(f"Published pose/TF frame: {self.base_link_frame}")
        for config in self._camera_configs.values():
            self.get_logger().info(
                f"[{config.name}] color={config.color_topic} depth={config.depth_topic} "
                f"camera_info={config.camera_info_topic} frame={config.camera_link_frame}"
            )
        self.get_logger().info(f"Save dir: {self.save_dir}")
        self.get_logger().info(f"Always save image override: {self.always_save_image}")
        if requested_lazy_subscriptions:
            self.get_logger().warn(
                "lazy_subscriptions parameter is ignored; persistent camera subscriptions are forced."
            )
        self.get_logger().info(
            "Lazy subscriptions: %s (idle_timeout=%.2fs, frame_wait_timeout=%.2fs, "
            "max_frame_age=%.2fs)"
            % (
                str(self.lazy_subscriptions).lower(),
                self.subscription_idle_timeout_sec,
                self.frame_wait_timeout_sec,
                self.max_frame_age_sec,
            )
        )

    def _subscribe_camera_streams(self, camera_name: str, config: CameraStreamConfig) -> None:
        with self._lock:
            if camera_name in self._camera_subscriptions:
                if self.lazy_subscriptions:
                    self._camera_subscription_deadlines[camera_name] = (
                        time.monotonic() + self.subscription_idle_timeout_sec
                    )
                return

        handles = CameraSubscriptionHandles(
            color=self.create_subscription(
                Image,
                config.color_topic,
                lambda msg, camera_name=camera_name: self._on_color_image(camera_name, msg),
                qos_profile_sensor_data,
            ),
            depth=self.create_subscription(
                Image,
                config.depth_topic,
                lambda msg, camera_name=camera_name: self._on_depth_image(camera_name, msg),
                qos_profile_sensor_data,
            ),
            camera_info=self.create_subscription(
                CameraInfo,
                config.camera_info_topic,
                lambda msg, camera_name=camera_name: self._on_camera_info(camera_name, msg),
                qos_profile_sensor_data,
            ),
        )

        with self._lock:
            self._camera_subscriptions[camera_name] = handles
            if self.lazy_subscriptions:
                self._camera_subscription_deadlines[camera_name] = (
                    time.monotonic() + self.subscription_idle_timeout_sec
                )

        self.get_logger().info(
            f"[{camera_name}] Subscribed to camera streams on demand."
        )

    def _unsubscribe_camera_streams(self, camera_name: str) -> None:
        with self._lock:
            handles = self._camera_subscriptions.pop(camera_name, None)
            self._camera_subscription_deadlines.pop(camera_name, None)

        if handles is None:
            return

        for subscription in (handles.color, handles.depth, handles.camera_info):
            try:
                self.destroy_subscription(subscription)
            except Exception:
                pass

        self.get_logger().info(f"[{camera_name}] Unsubscribed from idle camera streams.")

    def _cleanup_idle_subscriptions(self) -> None:
        if not self.lazy_subscriptions or self.subscription_idle_timeout_sec <= 0.0:
            return

        with self._lock:
            if self._active_detect_requests > 0:
                return

        now = time.monotonic()
        with self._lock:
            expired_camera_names = [
                camera_name
                for camera_name, deadline in self._camera_subscription_deadlines.items()
                if deadline <= now
            ]

        for camera_name in expired_camera_names:
            self._unsubscribe_camera_streams(camera_name)

    def _touch_camera_subscription(self, camera_name: str) -> None:
        if not self.lazy_subscriptions:
            return
        with self._lock:
            if camera_name in self._camera_subscriptions:
                self._camera_subscription_deadlines[camera_name] = (
                    time.monotonic() + self.subscription_idle_timeout_sec
                )

    def _ensure_camera_ready(self, camera_name: str, *, timeout_sec: float) -> tuple[bool, str]:
        if camera_name not in self._camera_configs:
            return False, f"Unsupported camera_name '{camera_name}'."

        config = self._camera_configs[camera_name]
        self._subscribe_camera_streams(camera_name, config)
        deadline = time.monotonic() + timeout_sec
        stale_state_detected = False

        while time.monotonic() < deadline:
            self._touch_camera_subscription(camera_name)
            now = time.monotonic()
            with self._lock:
                state = self._camera_states[camera_name]
                has_color = state.latest_color_image is not None
                has_depth = state.latest_depth_image is not None
                has_camera_info = state.latest_camera_info is not None
                color_fresh = has_color and (now - state.latest_color_received_s) <= self.max_frame_age_sec
                depth_fresh = has_depth and (now - state.latest_depth_received_s) <= self.max_frame_age_sec
                color_age = (now - state.latest_color_received_s) if has_color else -1.0
                depth_age = (now - state.latest_depth_received_s) if has_depth else -1.0
                camera_info_age = (now - state.latest_camera_info_received_s) if has_camera_info else -1.0

            if has_color and has_depth and has_camera_info:
                if not color_fresh or not depth_fresh:
                    stale_state_detected = True
                return True, ""

            if color_fresh and depth_fresh and has_camera_info:
                return True, ""

            remaining_sec = max(0.0, deadline - time.monotonic())
            if remaining_sec <= 0.0:
                break

            missing_topic_fetched = False
            if not has_color:
                received, msg = wait_for_message(
                    Image,
                    self,
                    config.color_topic,
                    qos_profile=qos_profile_sensor_data,
                    time_to_wait=min(0.25, remaining_sec),
                )
                if received and msg is not None:
                    self._on_color_image(camera_name, msg)
                    missing_topic_fetched = True

            remaining_sec = max(0.0, deadline - time.monotonic())
            if remaining_sec <= 0.0:
                break

            if not has_depth:
                received, msg = wait_for_message(
                    Image,
                    self,
                    config.depth_topic,
                    qos_profile=qos_profile_sensor_data,
                    time_to_wait=min(0.25, remaining_sec),
                )
                if received and msg is not None:
                    self._on_depth_image(camera_name, msg)
                    missing_topic_fetched = True

            remaining_sec = max(0.0, deadline - time.monotonic())
            if remaining_sec <= 0.0:
                break

            if not has_camera_info:
                received, msg = wait_for_message(
                    CameraInfo,
                    self,
                    config.camera_info_topic,
                    qos_profile=qos_profile_sensor_data,
                    time_to_wait=min(0.25, remaining_sec),
                )
                if received and msg is not None:
                    self._on_camera_info(camera_name, msg)
                    missing_topic_fetched = True

            if not missing_topic_fetched:
                time.sleep(0.05)

        with self._lock:
            state = self._camera_states[camera_name]
            if state.latest_color_image is None:
                return False, f"No image received on {config.color_topic}."
            if state.latest_depth_image is None:
                return (
                    False,
                    f"No depth image received on {config.depth_topic}. Set depth_registration:=true "
                    "and verify topic publishing.",
                )
            if state.latest_camera_info is None:
                return False, f"No camera info received on {config.camera_info_topic}."

        if stale_state_detected:
            self.get_logger().warn(
                f"[{camera_name}] Using cached frames after freshness timeout. "
                f"color_age={color_age:.2f}s depth_age={depth_age:.2f}s camera_info_age={camera_info_age:.2f}s"
            )
            return True, ""

        return False, (
            f"Timed out waiting for frames from camera '{camera_name}'. "
            f"Increase frame_wait_timeout_sec or inspect topic rates."
        )

    def _prepare_runtime(self) -> None:
        if self._torch is not None:
            return

        ensure_torch_runtime_libs()
        preload_cupti_if_needed()

        import torch

        self._torch = torch
        self._device = self._choose_device(self.device_request, torch)
        self.get_logger().info(f"Torch CUDA available: {torch.cuda.is_available()}")

    def _load_model(self, model_path: str, model_name: str) -> None:
        from ultralytics import YOLOE
        from ultralytics.nn.tasks import YOLOEModel

        started = time.perf_counter()
        self._model = YOLOE(model_path)
        self._prompt_key = None
        self._supports_prompt_classes = isinstance(self._model.model, YOLOEModel)
        self._loaded_model_path = model_path
        self._loaded_model_name = model_name
        elapsed_s = time.perf_counter() - started
        self.get_logger().info(f"Loaded {model_name} model in {elapsed_s:.2f}s from {model_path}")
        if self._supports_prompt_classes:
            resolved_assets = configure_ultralytics_text_asset_resolution()
            asset_path = resolved_assets.get("mobileclip2_b.ts")
            if asset_path is not None:
                self.get_logger().info(f"Resolved MobileCLIP asset: {asset_path}")
            self.get_logger().info(f"{model_name.capitalize()} model supports YOLOE text prompts.")
        else:
            fixed_classes = ", ".join(str(name) for name in self._model.names.values())
            self.get_logger().warn(
                f"{model_name.capitalize()} model is fixed-class. "
                f"Prompt text will be matched against: {fixed_classes}"
            )

    def _unload_model(self) -> None:
        if self._model is None:
            return

        self.get_logger().info(
            f"Unloading {self._loaded_model_name or 'active'} model from {self._loaded_model_path}"
        )
        self._model = None
        self._prompt_key = None
        self._supports_prompt_classes = False
        self._loaded_model_path = ""
        self._loaded_model_name = ""
        gc.collect()
        if self._torch is not None and self._device == "cuda" and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    def _ensure_model_loaded(self, model_path: str, model_name: str) -> None:
        if self._model is not None and self._loaded_model_path == model_path:
            return
        self._unload_model()
        self._load_model(model_path, model_name)

    @staticmethod
    def _choose_device(requested: str, torch_module: Any) -> str:
        if requested in ("cpu", "cuda"):
            return requested
        return "cuda" if torch_module.cuda.is_available() else "cpu"

    def _on_color_image(self, camera_name: str, msg: Image) -> None:
        try:
            image = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            self.get_logger().warn(f"[{camera_name}] Failed to convert color image: {exc}")
            return

        with self._lock:
            state = self._camera_states[camera_name]
            state.latest_color_image = image
            state.latest_color_received_s = time.monotonic()

    def _on_depth_image(self, camera_name: str, msg: Image) -> None:
        try:
            depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except CvBridgeError as exc:
            self.get_logger().warn(f"[{camera_name}] Failed to convert depth image: {exc}")
            return

        with self._lock:
            state = self._camera_states[camera_name]
            state.latest_depth_image = depth
            state.latest_depth_frame = msg.header.frame_id
            state.latest_depth_encoding = msg.encoding
            state.latest_depth_received_s = time.monotonic()

    def _on_camera_info(self, camera_name: str, msg: CameraInfo) -> None:
        with self._lock:
            state = self._camera_states[camera_name]
            state.latest_camera_info = msg
            state.latest_camera_info_received_s = time.monotonic()

    def _handle_detect_request(
        self, request: DetectObjectPrompt.Request, response: DetectObjectPrompt.Response
    ) -> DetectObjectPrompt.Response:
        response.success = False
        response.message = ""
        response.detected_classes = []
        response.confidences = []
        response.poses_camera_link = []
        response.tf_child_frames = []
        response.selected_side = ""
        response.saved_image_path = ""
        response.detections_in_frame = 0
        response.tf_published_count = 0
        response.inference_ms = 0.0

        try:
            prompts = self._parse_prompts(request.prompt_text)
        except ValueError as exc:
            response.message = str(exc)
            return response

        try:
            with self._lock:
                self._active_detect_requests += 1
            camera_name = self._resolve_camera_name(request.camera_name)
        except ValueError as exc:
            response.message = str(exc)
            with self._lock:
                self._active_detect_requests = max(0, self._active_detect_requests - 1)
            return response

        try:
            ready, ready_message = self._ensure_camera_ready(
                camera_name,
                timeout_sec=self.frame_wait_timeout_sec,
            )
            if not ready:
                response.message = ready_message
                return response

            use_bag_model = self._is_bag_only_request(prompts)
            self._touch_camera_subscription(camera_name)
            run_result = self._run_detection(
                camera_name,
                prompts,
                request.save_image,
                use_bag_model=use_bag_model,
            )
        except Exception as exc:
            self.get_logger().error(f"Detection request failed: {exc}")
            response.message = f"Detection request failed: {exc}"
            return response
        finally:
            with self._lock:
                self._active_detect_requests = max(0, self._active_detect_requests - 1)
            if 'use_bag_model' in locals() and use_bag_model:
                with self._inference_lock:
                    if self._loaded_model_path == self.bag_model_path:
                        self._unload_model()

        response.detections_in_frame = run_result.detections_in_frame
        response.tf_published_count = run_result.tf_published_count
        response.inference_ms = run_result.inference_ms
        response.saved_image_path = run_result.saved_image_path

        if run_result.error_message:
            response.message = run_result.error_message
            return response

        for entry in run_result.entries:
            response.detected_classes.append(entry.class_name)
            response.confidences.append(entry.confidence)
            response.poses_camera_link.append(entry.pose_camera_link)
            response.tf_child_frames.append(entry.tf_child_frame)

        response.success = True
        response.message = (
            f"Published {response.tf_published_count}/{response.detections_in_frame} TF frames in "
            f"{self.base_link_frame} using camera '{camera_name}'."
        )
        if run_result.skipped_count > 0:
            response.message += (
                f" Skipped {run_result.skipped_count} detections due to depth/TF limits."
            )
        return response

    def _run_detection(
        self,
        camera_name: str,
        prompts: list[str],
        save_image_request: bool,
        *,
        use_bag_model: bool,
    ) -> DetectionRunResult:
        config = self._camera_configs[camera_name]
        with self._lock:
            state = self._camera_states[camera_name]
            color_image = (
                None if state.latest_color_image is None else state.latest_color_image.copy()
            )
            depth_image = (
                None if state.latest_depth_image is None else state.latest_depth_image.copy()
            )
            depth_encoding = state.latest_depth_encoding
            depth_frame = state.latest_depth_frame
            camera_info = state.latest_camera_info

        if color_image is None:
            return DetectionRunResult(
                [], 0, 0, 0.0, "", 0, f"No image received on {config.color_topic}."
            )
        if depth_image is None:
            return DetectionRunResult(
                [],
                0,
                0,
                0.0,
                "",
                0,
                f"No depth image received on {config.depth_topic}. Set depth_registration:=true "
                "and verify topic publishing.",
            )
        if camera_info is None:
            return DetectionRunResult(
                [],
                0,
                0,
                0.0,
                "",
                0,
                f"No camera info received on {config.camera_info_topic}.",
            )

        with self._inference_lock:
            selected_model_path = self.bag_model_path if use_bag_model else self.model_path
            selected_model_name = "bag" if use_bag_model else "default"
            self._ensure_model_loaded(selected_model_path, selected_model_name)

            prompt_key = tuple(prompts)
            if prompt_key != self._prompt_key:
                started = time.perf_counter()
                if self._supports_prompt_classes:
                    self._model.set_classes(prompts)
                else:
                    available_classes = ", ".join(str(name) for name in self._model.names.values())
                    self.get_logger().info(
                        "Fixed-class model active; prompt filter '%s' will be matched against: %s"
                        % (", ".join(prompts), available_classes)
                    )
                set_classes_ms = (time.perf_counter() - started) * 1000.0
                self._prompt_key = prompt_key
                self.get_logger().info(
                    f"Prepared prompt classes ({', '.join(prompts)}) in {set_classes_ms:.1f} ms"
                )

            if self.force_torch_nms:
                sys.modules.pop("torchvision", None)

            infer_started = time.perf_counter()
            results = self._model.predict(
                source=color_image,
                device=self._device,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                max_det=self.max_det,
                verbose=False,
                save=False,
            )
            inference_ms = float((time.perf_counter() - infer_started) * 1000.0)

        if not results:
            return DetectionRunResult([], 0, 0, inference_ms, "", 0, "YOLOE returned no results.")

        result = results[0]
        if not self._supports_prompt_classes:
            matched_model_classes = self._matching_model_classes(prompts)
            if not matched_model_classes:
                available_classes = ", ".join(str(name) for name in self._model.names.values())
                return DetectionRunResult(
                    [],
                    0,
                    0,
                    inference_ms,
                    "",
                    0,
                    "Prompt did not match any classes in the loaded fixed-class model. "
                    f"Requested: {', '.join(prompts)}. Available classes: {available_classes}.",
                )

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return DetectionRunResult(
                [],
                0,
                0,
                inference_ms,
                "",
                0,
                "No objects detected for the requested prompt.",
            )

        if not self._supports_prompt_classes:
            matched_indices = self._matching_detection_indices(result, prompts)
            if not matched_indices:
                return DetectionRunResult(
                    [],
                    0,
                    0,
                    inference_ms,
                    "",
                    0,
                    "No objects detected for the requested prompt.",
                )
            self._filter_result_to_indices(result, matched_indices)
            boxes = result.boxes

        detections_in_frame = int(len(boxes))

        if depth_image.ndim > 2:
            depth_image = depth_image[:, :, 0]

        fx = float(camera_info.k[0])
        fy = float(camera_info.k[4])
        cx = float(camera_info.k[2])
        cy = float(camera_info.k[5])
        if fx <= 0.0 or fy <= 0.0:
            return DetectionRunResult(
                [],
                detections_in_frame,
                0,
                inference_ms,
                "",
                detections_in_frame,
                "Invalid camera intrinsics (fx/fy <= 0).",
            )

        if not depth_frame:
            depth_frame = camera_info.header.frame_id

        confidences = boxes.conf.detach().cpu().numpy()
        sorted_indices = np.argsort(-confidences)
        now_msg = self.get_clock().now().to_msg()

        entries: list[DetectionEntry] = []
        frame_map: dict[str, PublishedTFEntry] = {}
        per_class_count: dict[str, int] = {}
        skipped_count = 0

        for idx in sorted_indices:
            box = boxes[int(idx)]
            cls_id = int(box.cls.item())
            class_name = self._class_name(result.names, cls_id)
            confidence = float(box.conf.item())

            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]

            point_in_depth = self._compute_object_point_in_depth_frame(
                result=result,
                detection_index=int(idx),
                depth_image=depth_image,
                depth_encoding=depth_encoding,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
            )
            if point_in_depth is None:
                skipped_count += 1
                continue

            point_in_camera = self._transform_point_to_frame(
                point_in_depth, depth_frame, config.camera_link_frame
            )
            if point_in_camera is None:
                skipped_count += 1
                continue

            point_in_base = self._transform_point_to_frame(
                point_in_camera, config.camera_link_frame, self.base_link_frame
            )
            if point_in_base is None:
                skipped_count += 1
                continue

            class_slug = self._slug(class_name)
            class_count = per_class_count.get(class_slug, 0) + 1
            per_class_count[class_slug] = class_count

            if self.object_frame_prefix:
                child_frame = f"{self.object_frame_prefix}_{class_slug}_{class_count}"
            else:
                child_frame = f"{class_slug}_{class_count}"

            pose_msg = PoseStamped()
            pose_msg.header.frame_id = self.base_link_frame
            pose_msg.header.stamp = now_msg
            pose_msg.pose.position.x = float(point_in_base[0])
            pose_msg.pose.position.y = float(point_in_base[1])
            pose_msg.pose.position.z = float(point_in_base[2])
            pose_msg.pose.orientation.w = 1.0

            self._pose_pub.publish(pose_msg)
            self._publish_tf(child_frame, point_in_base, self.base_link_frame)

            frame_map[child_frame] = PublishedTFEntry(
                parent_frame=self.base_link_frame,
                translation=point_in_base.copy(),
            )
            entries.append(DetectionEntry(class_name, confidence, pose_msg, child_frame))
            break

        self._set_last_tfs(frame_map)

        saved_path = ""
        if self.always_save_image or save_image_request:
            annotated = result.plot()
            label = "best" if entries else "no_valid_depth"
            saved_path = self._save_annotated_image(annotated, prompts, label)

        if not entries:
            return DetectionRunResult(
                [],
                detections_in_frame,
                0,
                inference_ms,
                saved_path,
                skipped_count,
                "Objects were detected, but no valid depth/TF could be computed.",
            )

        return DetectionRunResult(
            entries,
            detections_in_frame,
            len(entries),
            inference_ms,
            saved_path,
            skipped_count,
            "",
        )

    def _extract_mask_for_detection(
        self,
        result: Any,
        detection_index: int,
        target_shape: tuple[int, int],
    ) -> np.ndarray | None:
        if result.masks is None or result.masks.data is None:
            return None

        if detection_index < 0 or detection_index >= len(result.masks.data):
            return None

        try:
            mask = result.masks.data[detection_index].detach().cpu().numpy()
        except Exception:
            return None

        if mask.ndim != 2:
            return None

        mask_bin = (mask > 0.5).astype(np.uint8)
        target_h, target_w = target_shape

        if mask_bin.shape[0] != target_h or mask_bin.shape[1] != target_w:
            mask_bin = cv2.resize(
                mask_bin,
                (target_w, target_h),
                interpolation=cv2.INTER_NEAREST,
            )

        if self.mask_erode_pixels > 0:
            kernel_size = 2 * self.mask_erode_pixels + 1
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            mask_bin = cv2.erode(mask_bin, kernel, iterations=1)

        if int(mask_bin.sum()) == 0:
            return None

        return mask_bin

    def _compute_mask_3d_point_in_depth_frame(
        self,
        mask: np.ndarray,
        depth_image: np.ndarray,
        depth_encoding: str,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
    ) -> np.ndarray | None:
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return None

        stride = max(1, self.mask_depth_stride)
        if stride > 1:
            xs = xs[::stride]
            ys = ys[::stride]

        if len(xs) > self.mask_max_points:
            step = int(math.ceil(len(xs) / float(self.mask_max_points)))
            xs = xs[::step]
            ys = ys[::step]

        points: list[np.ndarray] = []

        for u, v in zip(xs, ys):
            depth_m = self._depth_value_to_meters(
                depth_image[v, u],
                depth_encoding,
                depth_image.dtype,
            )
            if depth_m is None:
                continue
            if depth_m < self.min_depth_m or depth_m > self.max_depth_m:
                continue

            x = ((float(u) - cx) / fx) * depth_m
            y = ((float(v) - cy) / fy) * depth_m
            z = depth_m
            points.append(np.array([x, y, z], dtype=np.float64))

        if len(points) < self.mask_min_valid_points:
            return None

        point_array = np.asarray(points, dtype=np.float64)

        median_z = float(np.median(point_array[:, 2]))
        abs_dev_z = np.abs(point_array[:, 2] - median_z)
        mad_z = float(np.median(abs_dev_z))

        if mad_z > 1e-6:
            inlier_mask = abs_dev_z <= (2.5 * 1.4826 * mad_z)
            filtered = point_array[inlier_mask]
            if len(filtered) >= max(10, self.mask_min_valid_points // 2):
                point_array = filtered

        return np.median(point_array, axis=0)

    def _compute_object_point_in_depth_frame(
        self,
        result: Any,
        detection_index: int,
        depth_image: np.ndarray,
        depth_encoding: str,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> np.ndarray | None:
        mask = self._extract_mask_for_detection(
            result,
            detection_index,
            depth_image.shape[:2],
        )
        if mask is not None:
            mask_point = self._compute_mask_3d_point_in_depth_frame(
                mask,
                depth_image,
                depth_encoding,
                fx,
                fy,
                cx,
                cy,
            )
            if mask_point is not None:
                return mask_point

        u = int(round((x1 + x2) * 0.5))
        v = int(round((y1 + y2) * 0.5))

        if u < 0 or v < 0 or u >= depth_image.shape[1] or v >= depth_image.shape[0]:
            return None

        depth_m = self._sample_depth_meters(depth_image, depth_encoding, u, v)
        if depth_m is None:
            depth_m = self._sample_depth_from_box_meters(
                depth_image,
                depth_encoding,
                x1,
                y1,
                x2,
                y2,
            )
        if depth_m is None:
            return None

        return np.array(
            [
                ((float(u) - cx) / fx) * depth_m,
                ((float(v) - cy) / fy) * depth_m,
                depth_m,
            ],
            dtype=np.float64,
        )

    def _publish_last_tf(self) -> None:
        with self._lock:
            if not self._last_tf_map:
                return
            frame_items = [
                (
                    child,
                    PublishedTFEntry(
                        parent_frame=entry.parent_frame,
                        translation=entry.translation.copy(),
                    ),
                )
                for child, entry in self._last_tf_map.items()
            ]

        for child_frame, entry in frame_items:
            self._publish_tf(child_frame, entry.translation, entry.parent_frame)

    def _set_last_tfs(self, frame_map: dict[str, PublishedTFEntry]) -> None:
        with self._lock:
            self._last_tf_map = {
                child: PublishedTFEntry(
                    parent_frame=entry.parent_frame,
                    translation=entry.translation.copy(),
                )
                for child, entry in frame_map.items()
            }

    def _publish_tf(
        self, child_frame: str, translation: np.ndarray, parent_frame: str
    ) -> None:
        tf_msg = TransformStamped()
        tf_msg.header.stamp = self.get_clock().now().to_msg()
        tf_msg.header.frame_id = parent_frame
        tf_msg.child_frame_id = child_frame
        tf_msg.transform.translation.x = float(translation[0])
        tf_msg.transform.translation.y = float(translation[1])
        tf_msg.transform.translation.z = float(translation[2])
        tf_msg.transform.rotation.w = 1.0
        self._tf_broadcaster.sendTransform(tf_msg)

    def _transform_point_to_frame(
        self, point_in_source: np.ndarray, source_frame: str, target_frame: str
    ) -> np.ndarray | None:
        if source_frame == target_frame or source_frame == "":
            return point_in_source

        cache_key = (target_frame, source_frame)
        try:
            transform = self._tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                Time(),
                timeout=self._tf_lookup_timeout,
            )
            cached_transform = CachedFrameTransform(
                translation=np.array(
                    [
                        transform.transform.translation.x,
                        transform.transform.translation.y,
                        transform.transform.translation.z,
                    ],
                    dtype=np.float64,
                ),
                rotation_xyzw=(
                    transform.transform.rotation.x,
                    transform.transform.rotation.y,
                    transform.transform.rotation.z,
                    transform.transform.rotation.w,
                ),
            )
            with self._lock:
                self._frame_transform_cache[cache_key] = CachedFrameTransform(
                    translation=cached_transform.translation.copy(),
                    rotation_xyzw=cached_transform.rotation_xyzw,
                )
        except TransformException as exc:
            with self._lock:
                cached_transform = self._frame_transform_cache.get(cache_key)
                if cached_transform is not None:
                    cached_transform = CachedFrameTransform(
                        translation=cached_transform.translation.copy(),
                        rotation_xyzw=cached_transform.rotation_xyzw,
                    )
            if cached_transform is None:
                self.get_logger().warn(f"TF lookup failed ({source_frame} -> {target_frame}): {exc}")
                return None

            self.get_logger().warn(
                f"TF lookup failed ({source_frame} -> {target_frame}); using cached transform: {exc}"
            )

        return self._apply_cached_transform(point_in_source, cached_transform)

    def _apply_cached_transform(
        self, point_in_source: np.ndarray, cached_transform: CachedFrameTransform
    ) -> np.ndarray:
        rotated = self._rotate_vector_by_quaternion(
            point_in_source,
            cached_transform.rotation_xyzw[0],
            cached_transform.rotation_xyzw[1],
            cached_transform.rotation_xyzw[2],
            cached_transform.rotation_xyzw[3],
        )
        return rotated + cached_transform.translation

    def _resolve_camera_name(self, requested_camera_name: str) -> str:
        camera_name = self._normalize_camera_name(requested_camera_name)
        if not camera_name:
            return self.default_camera_name
        if camera_name not in self._camera_configs:
            supported = ", ".join(sorted(self._camera_configs))
            raise ValueError(
                f"Unsupported camera_name '{requested_camera_name}'. Supported camera_name values: "
                f"{supported}."
            )
        return camera_name

    @staticmethod
    def _rotate_vector_by_quaternion(
        vector: np.ndarray,
        qx: float,
        qy: float,
        qz: float,
        qw: float,
    ) -> np.ndarray:
        norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        if norm == 0.0:
            return vector

        x = qx / norm
        y = qy / norm
        z = qz / norm
        w = qw / norm

        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z
        wx = w * x
        wy = w * y
        wz = w * z

        rotation_matrix = np.array(
            [
                [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
                [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
                [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
            ],
            dtype=np.float64,
        )
        return rotation_matrix.dot(vector)

    def _sample_depth_meters(
        self,
        depth_image: np.ndarray,
        encoding: str,
        u: int,
        v: int,
    ) -> float | None:
        height, width = depth_image.shape[:2]
        radius = max(0, self.depth_window_size // 2)

        values: list[float] = []
        for yy in range(max(0, v - radius), min(height, v + radius + 1)):
            for xx in range(max(0, u - radius), min(width, u + radius + 1)):
                depth_m = self._depth_value_to_meters(
                    depth_image[yy, xx], encoding, depth_image.dtype
                )
                if depth_m is None:
                    continue
                if depth_m < self.min_depth_m or depth_m > self.max_depth_m:
                    continue
                values.append(depth_m)

        if not values:
            return None

        return float(np.median(values))

    def _sample_depth_from_box_meters(
        self,
        depth_image: np.ndarray,
        encoding: str,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> float | None:
        height, width = depth_image.shape[:2]
        min_x = max(0, int(math.floor(min(x1, x2))))
        max_x = min(width - 1, int(math.ceil(max(x1, x2))))
        min_y = max(0, int(math.floor(min(y1, y2))))
        max_y = min(height - 1, int(math.ceil(max(y1, y2))))
        if min_x > max_x or min_y > max_y:
            return None

        trim_x = max(0, int((max_x - min_x + 1) * 0.15))
        trim_y = max(0, int((max_y - min_y + 1) * 0.15))
        inner_min_x = min(max_x, min_x + trim_x)
        inner_max_x = max(min_x, max_x - trim_x)
        inner_min_y = min(max_y, min_y + trim_y)
        inner_max_y = max(min_y, max_y - trim_y)

        values: list[float] = []
        for yy in range(inner_min_y, inner_max_y + 1):
            for xx in range(inner_min_x, inner_max_x + 1):
                depth_m = self._depth_value_to_meters(
                    depth_image[yy, xx], encoding, depth_image.dtype
                )
                if depth_m is None:
                    continue
                if depth_m < self.min_depth_m or depth_m > self.max_depth_m:
                    continue
                values.append(depth_m)

        if not values:
            return None

        return float(np.percentile(values, 25.0 if len(values) >= 8 else 50.0))

    @staticmethod
    def _depth_value_to_meters(
        value: Any, encoding: str, dtype: np.dtype[Any]
    ) -> float | None:
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

        return depth_value

    @staticmethod
    def _parse_prompts(prompt_text: str) -> list[str]:
        prompts = [entry.strip() for entry in prompt_text.split(",") if entry.strip()]
        if not prompts:
            raise ValueError("prompt_text is empty. Example: 'bottle' or 'cup,bottle'.")
        return prompts

    @staticmethod
    def _slug(text: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9_]+", "_", text).strip("_").lower()
        return slug or "object"

    @staticmethod
    def _normalize_camera_name(camera_name: str) -> str:
        return camera_name.strip().lower()

    @staticmethod
    def _coalesce_non_empty(value: str, fallback: str) -> str:
        stripped = value.strip()
        return stripped if stripped else fallback.strip()

    @staticmethod
    def _class_name(names: Any, class_id: int) -> str:
        if isinstance(names, dict):
            return str(names.get(class_id, class_id))
        if isinstance(names, list) and 0 <= class_id < len(names):
            return str(names[class_id])
        return str(class_id)

    @staticmethod
    def _normalize_class_label(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()

    def _is_bag_prompt(self, prompt: str) -> bool:
        return self._normalize_class_label(prompt) in self.bag_prompt_aliases

    def _is_bag_only_request(self, prompts: list[str]) -> bool:
        return bool(prompts) and all(self._is_bag_prompt(prompt) for prompt in prompts)

    def _prompt_matches_class(self, prompt: str, class_name: str) -> bool:
        normalized_prompt = self._normalize_class_label(prompt)
        normalized_class = self._normalize_class_label(class_name)
        if not normalized_prompt or not normalized_class:
            return False
        return (
            normalized_prompt == normalized_class
            or normalized_prompt in normalized_class
            or normalized_class in normalized_prompt
        )

    def _matching_detection_indices(self, result: Any, prompts: list[str]) -> list[int]:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        matched_indices: list[int] = []
        for idx, box in enumerate(boxes):
            class_name = self._class_name(result.names, int(box.cls.item()))
            if any(self._prompt_matches_class(prompt, class_name) for prompt in prompts):
                matched_indices.append(idx)
        return matched_indices

    def _matching_model_classes(self, prompts: list[str]) -> list[str]:
        matched_classes: list[str] = []
        for class_name in self._model.names.values():
            class_name = str(class_name)
            if any(self._prompt_matches_class(prompt, class_name) for prompt in prompts):
                matched_classes.append(class_name)
        return matched_classes

    @staticmethod
    def _filter_result_to_indices(result: Any, matched_indices: list[int]) -> None:
        result.update(
            boxes=result.boxes.data[matched_indices],
            masks=(
                result.masks.data[matched_indices]
                if result.masks is not None and result.masks.data is not None
                else None
            ),
        )

    def _save_annotated_image(
        self, image: np.ndarray, prompts: list[str], detected_class: str
    ) -> str:
        prompt_text = "_".join(prompts)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{ts}_{self._slug(prompt_text)}_{self._slug(detected_class)}.jpg"
        path = self.save_dir / filename
        cv2.imwrite(str(path), image)
        self.get_logger().info(f"Saved detection image: {path}")
        return str(path)

    def destroy_node(self) -> bool:
        for camera_name in list(self._camera_configs):
            self._unsubscribe_camera_streams(camera_name)
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = YoloeDetectionServiceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

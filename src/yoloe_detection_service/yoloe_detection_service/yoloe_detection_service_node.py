#!/usr/bin/env python3
"""YOLOE text-prompt detection service node for Dabai camera streams.

This node loads YOLOE once and supports single-shot detection via
service (/yoloe/detect_prompt).
"""

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
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformBroadcaster, TransformException, TransformListener

from yoloe_detection_interfaces.srv import DetectObjectPrompt


def ensure_torch_runtime_libs() -> None:
    """Expose CUDA libraries from pip wheels and system paths before importing torch."""
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
    candidates = [
        pathlib.Path(
            "/home/usern/coqui-venv/lib/python3.10/site-packages/nvidia/cuda_cupti/lib/libcupti.so.12"
        ),
        pathlib.Path("/usr/local/cuda-12.6/extras/CUPTI/lib64/libcupti.so.12"),
        pathlib.Path("/usr/local/cuda/extras/CUPTI/lib64/libcupti.so.12"),
    ]
    for candidate in candidates:
        if candidate.exists():
            ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
            return


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


class YoloeDetectionServiceNode(Node):
    def __init__(self) -> None:
        super().__init__("yoloe_detection_service_node")

        self.declare_parameter("service_name", "/yoloe/detect_prompt")
        self.declare_parameter("model_path", "/home/usern/yoloe-26l-seg.pt")
        self.declare_parameter("device", "auto")
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("conf", 0.25)
        self.declare_parameter("iou", 0.45)
        self.declare_parameter("max_det", 300)
        self.declare_parameter("force_torch_nms", True)

        self.declare_parameter("color_topic", "/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/depth/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/color/camera_info")
        self.declare_parameter("camera_link_frame", "camera_link")

        self.declare_parameter("pose_topic", "/yoloe/detected_pose")
        self.declare_parameter("object_frame_prefix", "")
        self.declare_parameter("save_dir", "/home/usern/robocup_ws/yoloe_out")
        self.declare_parameter("always_save_image", False)

        self.declare_parameter("depth_window_size", 5)
        self.declare_parameter("min_depth_m", 0.05)
        self.declare_parameter("max_depth_m", 10.0)

        self.service_name = str(self.get_parameter("service_name").value)
        self.model_path = str(self.get_parameter("model_path").value)
        self.device_request = str(self.get_parameter("device").value)
        self.imgsz = int(self.get_parameter("imgsz").value)
        self.conf = float(self.get_parameter("conf").value)
        self.iou = float(self.get_parameter("iou").value)
        self.max_det = int(self.get_parameter("max_det").value)
        self.force_torch_nms = bool(self.get_parameter("force_torch_nms").value)

        self.color_topic = str(self.get_parameter("color_topic").value)
        self.depth_topic = str(self.get_parameter("depth_topic").value)
        self.camera_info_topic = str(self.get_parameter("camera_info_topic").value)
        self.camera_link_frame = str(self.get_parameter("camera_link_frame").value)

        self.pose_topic = str(self.get_parameter("pose_topic").value)
        self.object_frame_prefix = str(self.get_parameter("object_frame_prefix").value)
        self.save_dir = pathlib.Path(str(self.get_parameter("save_dir").value)).expanduser().resolve()
        self.always_save_image = bool(self.get_parameter("always_save_image").value)

        self.depth_window_size = max(1, int(self.get_parameter("depth_window_size").value))
        self.min_depth_m = float(self.get_parameter("min_depth_m").value)
        self.max_depth_m = float(self.get_parameter("max_depth_m").value)

        self.save_dir.mkdir(parents=True, exist_ok=True)

        self._bridge = CvBridge()
        self._lock = threading.Lock()
        self._inference_lock = threading.Lock()

        self._latest_color_image: np.ndarray | None = None
        self._latest_depth_image: np.ndarray | None = None
        self._latest_depth_frame: str = ""
        self._latest_depth_encoding: str = ""
        self._latest_camera_info: CameraInfo | None = None

        self._last_tf_map: dict[str, np.ndarray] = {}

        self._model: Any = None
        self._prompt_key: tuple[str, ...] | None = None
        self._device: str = "cpu"

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._tf_broadcaster = TransformBroadcaster(self)

        self._pose_pub = self.create_publisher(PoseStamped, self.pose_topic, 10)
        self.create_subscription(Image, self.color_topic, self._on_color_image, qos_profile_sensor_data)
        self.create_subscription(Image, self.depth_topic, self._on_depth_image, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, self.camera_info_topic, self._on_camera_info, qos_profile_sensor_data)

        self.create_service(DetectObjectPrompt, self.service_name, self._handle_detect_request)
        self.create_timer(0.2, self._publish_last_tf)

        self._load_model()

        self.get_logger().info(f"YOLOE service ready on {self.service_name}")
        self.get_logger().info(f"Using model: {self.model_path}")
        self.get_logger().info(f"Device: {self._device}")
        self.get_logger().info(f"Color topic: {self.color_topic}")
        self.get_logger().info(f"Depth topic: {self.depth_topic}")
        self.get_logger().info(f"Camera info topic: {self.camera_info_topic}")
        self.get_logger().info(f"camera_link frame: {self.camera_link_frame}")
        self.get_logger().info(f"Save dir: {self.save_dir}")
        self.get_logger().info(f"Always save image override: {self.always_save_image}")

    def _load_model(self) -> None:
        ensure_torch_runtime_libs()
        preload_cupti_if_needed()

        import torch
        from ultralytics import YOLOE

        self._device = self._choose_device(self.device_request, torch)

        started = time.perf_counter()
        self._model = YOLOE(self.model_path)
        elapsed_s = time.perf_counter() - started
        self.get_logger().info(f"Loaded YOLOE model in {elapsed_s:.2f}s")
        self.get_logger().info(f"Torch CUDA available: {torch.cuda.is_available()}")

    @staticmethod
    def _choose_device(requested: str, torch_module: Any) -> str:
        if requested in ("cpu", "cuda"):
            return requested
        return "cuda" if torch_module.cuda.is_available() else "cpu"

    def _on_color_image(self, msg: Image) -> None:
        try:
            image = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            self.get_logger().warn(f"Failed to convert color image: {exc}")
            return

        with self._lock:
            self._latest_color_image = image

    def _on_depth_image(self, msg: Image) -> None:
        try:
            depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except CvBridgeError as exc:
            self.get_logger().warn(f"Failed to convert depth image: {exc}")
            return

        with self._lock:
            self._latest_depth_image = depth
            self._latest_depth_frame = msg.header.frame_id
            self._latest_depth_encoding = msg.encoding

    def _on_camera_info(self, msg: CameraInfo) -> None:
        with self._lock:
            self._latest_camera_info = msg

    def _handle_detect_request(
        self, request: DetectObjectPrompt.Request, response: DetectObjectPrompt.Response
    ) -> DetectObjectPrompt.Response:
        response.success = False
        response.message = ""
        response.detected_classes = []
        response.confidences = []
        response.poses_camera_link = []
        response.tf_child_frames = []
        response.saved_image_path = ""
        response.detections_in_frame = 0
        response.tf_published_count = 0
        response.inference_ms = 0.0

        try:
            prompts = self._parse_prompts(request.prompt_text)
        except ValueError as exc:
            response.message = str(exc)
            return response

        run_result = self._run_detection(prompts, request.save_image)
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
            f"{self.camera_link_frame}."
        )
        if run_result.skipped_count > 0:
            response.message += (
                f" Skipped {run_result.skipped_count} detections due to depth/TF limits."
            )
        return response

    def _run_detection(self, prompts: list[str], save_image_request: bool) -> DetectionRunResult:
        with self._lock:
            color_image = None if self._latest_color_image is None else self._latest_color_image.copy()
            depth_image = None if self._latest_depth_image is None else self._latest_depth_image.copy()
            depth_encoding = self._latest_depth_encoding
            depth_frame = self._latest_depth_frame
            camera_info = self._latest_camera_info

        if color_image is None:
            return DetectionRunResult([], 0, 0, 0.0, "", 0, f"No image received on {self.color_topic}.")
        if depth_image is None:
            return DetectionRunResult(
                [],
                0,
                0,
                0.0,
                "",
                0,
                f"No depth image received on {self.depth_topic}. Set depth_registration:=true and verify topic publishing.",
            )
        if camera_info is None:
            return DetectionRunResult(
                [],
                0,
                0,
                0.0,
                "",
                0,
                f"No camera info received on {self.camera_info_topic}.",
            )

        with self._inference_lock:
            prompt_key = tuple(prompts)
            if prompt_key != self._prompt_key:
                started = time.perf_counter()
                self._model.set_classes(prompts)
                set_classes_ms = (time.perf_counter() - started) * 1000.0
                self._prompt_key = prompt_key
                self.get_logger().info(
                    f"Updated prompt classes ({', '.join(prompts)}) in {set_classes_ms:.1f} ms"
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
        frame_map: dict[str, np.ndarray] = {}
        per_class_count: dict[str, int] = {}
        skipped_count = 0

        for idx in sorted_indices:
            box = boxes[int(idx)]
            cls_id = int(box.cls.item())
            class_name = self._class_name(result.names, cls_id)
            confidence = float(box.conf.item())

            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
            u = int(round((x1 + x2) * 0.5))
            v = int(round((y1 + y2) * 0.5))

            if u < 0 or v < 0 or u >= depth_image.shape[1] or v >= depth_image.shape[0]:
                skipped_count += 1
                continue

            depth_m = self._sample_depth_meters(depth_image, depth_encoding, u, v)
            if depth_m is None:
                skipped_count += 1
                continue

            point_in_depth = np.array(
                [
                    ((float(u) - cx) / fx) * depth_m,
                    ((float(v) - cy) / fy) * depth_m,
                    depth_m,
                ],
                dtype=np.float64,
            )
            point_in_camera = self._transform_point_to_camera_link(point_in_depth, depth_frame)
            if point_in_camera is None:
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
            pose_msg.header.frame_id = self.camera_link_frame
            pose_msg.header.stamp = now_msg
            pose_msg.pose.position.x = float(point_in_camera[0])
            pose_msg.pose.position.y = float(point_in_camera[1])
            pose_msg.pose.position.z = float(point_in_camera[2])
            pose_msg.pose.orientation.w = 1.0

            self._pose_pub.publish(pose_msg)
            self._publish_tf(child_frame, point_in_camera)

            frame_map[child_frame] = point_in_camera.copy()
            entries.append(DetectionEntry(class_name, confidence, pose_msg, child_frame))

        self._set_last_tfs(frame_map)

        saved_path = ""
        if self.always_save_image or save_image_request:
            annotated = result.plot()
            label = "multi" if entries else "no_valid_depth"
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

    def _publish_last_tf(self) -> None:
        with self._lock:
            if not self._last_tf_map:
                return
            frame_items = [
                (child, translation.copy()) for child, translation in self._last_tf_map.items()
            ]

        for child_frame, translation in frame_items:
            self._publish_tf(child_frame, translation)

    def _set_last_tfs(self, frame_map: dict[str, np.ndarray]) -> None:
        with self._lock:
            self._last_tf_map = {
                child: translation.copy() for child, translation in frame_map.items()
            }

    def _publish_tf(self, child_frame: str, translation: np.ndarray) -> None:
        tf_msg = TransformStamped()
        tf_msg.header.stamp = self.get_clock().now().to_msg()
        tf_msg.header.frame_id = self.camera_link_frame
        tf_msg.child_frame_id = child_frame
        tf_msg.transform.translation.x = float(translation[0])
        tf_msg.transform.translation.y = float(translation[1])
        tf_msg.transform.translation.z = float(translation[2])
        tf_msg.transform.rotation.w = 1.0
        self._tf_broadcaster.sendTransform(tf_msg)

    def _transform_point_to_camera_link(
        self, point_in_source: np.ndarray, source_frame: str
    ) -> np.ndarray | None:
        if source_frame == self.camera_link_frame or source_frame == "":
            return point_in_source

        try:
            transform = self._tf_buffer.lookup_transform(
                self.camera_link_frame,
                source_frame,
                Time(),
            )
        except TransformException as exc:
            self.get_logger().warn(
                f"TF lookup failed ({source_frame} -> {self.camera_link_frame}): {exc}"
            )
            return None

        rotation = transform.transform.rotation
        translation = transform.transform.translation

        rotated = self._rotate_vector_by_quaternion(
            point_in_source,
            rotation.x,
            rotation.y,
            rotation.z,
            rotation.w,
        )
        transformed = rotated + np.array(
            [translation.x, translation.y, translation.z], dtype=np.float64
        )
        return transformed

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
    def _class_name(names: Any, class_id: int) -> str:
        if isinstance(names, dict):
            return str(names.get(class_id, class_id))
        if isinstance(names, list) and 0 <= class_id < len(names):
            return str(names[class_id])
        return str(class_id)

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

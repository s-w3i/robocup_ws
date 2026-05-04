#!/usr/bin/env python3
"""ROS 2 service node that captures a single image from a configured camera stream."""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from vlm_interfaces.srv import CaptureImage


@dataclass
class CameraFrame:
    image_bgr: Any
    stamp_ns: int
    received_monotonic: float


class CameraSnapshotServiceNode(Node):
    def __init__(self) -> None:
        super().__init__("camera_snapshot_service_node")

        self.declare_parameter("service_name", "/camera/capture")
        self.declare_parameter("default_camera_name", "camera0")
        self.declare_parameter("default_camera_topic", "/camera0/color/image_raw")
        self.declare_parameter("camera_names", ["camera0"])
        self.declare_parameter("camera_topics", ["/camera0/color/image_raw"])
        self.declare_parameter("camera_names_csv", "")
        self.declare_parameter("camera_topics_csv", "")
        self.declare_parameter("default_save_dir", "/home/usern/robocup_ws/captures")
        self.declare_parameter("image_wait_timeout_sec", 3.0)
        self.declare_parameter("jpeg_quality", 95)
        self.declare_parameter("lazy_subscriptions", True)
        self.declare_parameter("subscription_idle_timeout_sec", 3.0)
        self.declare_parameter("subscription_poll_period_sec", 0.5)

        self.service_name = str(self.get_parameter("service_name").value).strip() or "/camera/capture"
        self.default_camera_name = (
            str(self.get_parameter("default_camera_name").value).strip() or "camera0"
        )
        self.default_camera_topic = (
            str(self.get_parameter("default_camera_topic").value).strip()
            or "/camera0/color/image_raw"
        )
        self.camera_names = [str(name).strip() for name in self.get_parameter("camera_names").value]
        self.camera_topics = [str(topic).strip() for topic in self.get_parameter("camera_topics").value]
        camera_names_csv = str(self.get_parameter("camera_names_csv").value).strip()
        camera_topics_csv = str(self.get_parameter("camera_topics_csv").value).strip()
        self.default_save_dir = Path(
            str(self.get_parameter("default_save_dir").value).strip()
            or "/home/usern/robocup_ws/captures"
        ).expanduser()
        self.image_wait_timeout_sec = max(0.1, float(self.get_parameter("image_wait_timeout_sec").value))
        self.jpeg_quality = min(100, max(50, int(self.get_parameter("jpeg_quality").value)))
        self.lazy_subscriptions = bool(self.get_parameter("lazy_subscriptions").value)
        self.subscription_idle_timeout_sec = max(
            0.0, float(self.get_parameter("subscription_idle_timeout_sec").value)
        )
        self.subscription_poll_period_sec = max(
            0.1, float(self.get_parameter("subscription_poll_period_sec").value)
        )

        if not camera_names_csv and not camera_topics_csv:
            self.camera_names = [self.default_camera_name]
            self.camera_topics = [self.default_camera_topic]
        if camera_names_csv:
            self.camera_names = [part.strip() for part in camera_names_csv.split(",") if part.strip()]
        if camera_topics_csv:
            self.camera_topics = [part.strip() for part in camera_topics_csv.split(",") if part.strip()]

        if len(self.camera_names) != len(self.camera_topics):
            raise ValueError("camera_names and camera_topics must have the same length")

        self._camera_topics_by_name = dict(zip(self.camera_names, self.camera_topics))
        if self.default_camera_name not in self._camera_topics_by_name:
            self._camera_topics_by_name[self.default_camera_name] = self.default_camera_topic

        self._callback_group = ReentrantCallbackGroup()
        self._bridge = CvBridge()
        self._frames: dict[str, CameraFrame] = {}
        self._lock = threading.Lock()
        self._camera_subscriptions: dict[str, Any] = {}
        self._camera_subscription_deadlines: dict[str, float] = {}

        self.create_timer(self.subscription_poll_period_sec, self._cleanup_idle_subscriptions)

        if not self.lazy_subscriptions:
            for camera_name in self._camera_topics_by_name:
                self._subscribe_camera(camera_name)

        self.create_service(
            CaptureImage,
            self.service_name,
            self._handle_capture,
            callback_group=self._callback_group,
        )

        self.default_save_dir.mkdir(parents=True, exist_ok=True)
        self.get_logger().info(
            f"Camera snapshot service ready on {self.service_name} | "
            f"default_camera={self.default_camera_name} | save_dir={self.default_save_dir}"
        )
        self.get_logger().info(
            "Lazy subscriptions: %s (idle_timeout=%.2fs)"
            % (str(self.lazy_subscriptions).lower(), self.subscription_idle_timeout_sec)
        )

    def _on_image(self, camera_name: str, msg: Image) -> None:
        try:
            image = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            self.get_logger().warn(f"Failed to convert image from '{camera_name}': {exc}")
            return

        stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        with self._lock:
            self._frames[camera_name] = CameraFrame(
                image_bgr=image,
                stamp_ns=stamp_ns,
                received_monotonic=time.monotonic(),
            )

    def _subscribe_camera(self, camera_name: str) -> None:
        if camera_name not in self._camera_topics_by_name:
            raise ValueError(
                f"Unknown camera '{camera_name}'. Available cameras: {sorted(self._camera_topics_by_name)}"
            )

        topic = self._camera_topics_by_name[camera_name]
        with self._lock:
            if camera_name in self._camera_subscriptions:
                if self.lazy_subscriptions:
                    self._camera_subscription_deadlines[camera_name] = (
                        time.monotonic() + self.subscription_idle_timeout_sec
                    )
                return

        subscription = self.create_subscription(
            Image,
            topic,
            lambda msg, name=camera_name: self._on_image(name, msg),
            qos_profile_sensor_data,
            callback_group=self._callback_group,
        )

        with self._lock:
            self._camera_subscriptions[camera_name] = subscription
            if self.lazy_subscriptions:
                self._camera_subscription_deadlines[camera_name] = (
                    time.monotonic() + self.subscription_idle_timeout_sec
                )

        self.get_logger().info(f"[{camera_name}] Subscribed to camera stream on demand.")

    def _unsubscribe_camera(self, camera_name: str) -> None:
        with self._lock:
            subscription = self._camera_subscriptions.pop(camera_name, None)
            self._camera_subscription_deadlines.pop(camera_name, None)
            self._frames.pop(camera_name, None)

        if subscription is None:
            return

        try:
            self.destroy_subscription(subscription)
        except Exception:
            pass

        self.get_logger().info(f"[{camera_name}] Unsubscribed from idle camera stream.")

    def _touch_camera_subscription(self, camera_name: str) -> None:
        if not self.lazy_subscriptions:
            return
        with self._lock:
            if camera_name in self._camera_subscriptions:
                self._camera_subscription_deadlines[camera_name] = (
                    time.monotonic() + self.subscription_idle_timeout_sec
                )

    def _cleanup_idle_subscriptions(self) -> None:
        if not self.lazy_subscriptions or self.subscription_idle_timeout_sec <= 0.0:
            return

        now = time.monotonic()
        with self._lock:
            expired_camera_names = [
                camera_name
                for camera_name, deadline in self._camera_subscription_deadlines.items()
                if deadline <= now
            ]

        for camera_name in expired_camera_names:
            self._unsubscribe_camera(camera_name)

    def _handle_capture(
        self, request: CaptureImage.Request, response: CaptureImage.Response
    ) -> CaptureImage.Response:
        response.success = False
        response.message = ""
        response.camera_used = ""
        response.saved_image_path = ""

        camera_name = self._resolve_camera_name(request.camera_name)
        response.camera_used = camera_name
        frame = self._wait_for_frame(
            camera_name,
            min_received_monotonic=time.monotonic(),
        )
        if frame is None:
            response.message = (
                f"No fresh image available from camera '{camera_name}'. "
                "Check the camera topic or increase image_wait_timeout_sec."
            )
            return response

        save_dir = Path(str(request.save_dir).strip()).expanduser() if str(request.save_dir).strip() else self.default_save_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        prefix = self._slug(str(request.file_prefix).strip() or "snapshot")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{prefix}_{self._slug(camera_name)}_{ts}.jpg"
        path = save_dir / filename
        ok = cv2.imwrite(
            str(path),
            frame.image_bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            response.message = f"Failed to save image to {path}"
            return response

        response.success = True
        response.message = "ok"
        response.saved_image_path = str(path)
        self.get_logger().info(f"Saved camera snapshot: {path}")
        return response

    def _wait_for_frame(
        self,
        camera_name: str,
        min_received_monotonic: float | None = None,
    ) -> CameraFrame | None:
        self._subscribe_camera(camera_name)
        deadline = time.time() + self.image_wait_timeout_sec
        while time.time() < deadline:
            self._touch_camera_subscription(camera_name)
            with self._lock:
                frame = self._frames.get(camera_name)
            if frame is not None:
                if min_received_monotonic is not None and frame.received_monotonic < min_received_monotonic:
                    time.sleep(0.05)
                    continue
                return frame
            time.sleep(0.05)
        return None

    def _resolve_camera_name(self, requested_camera_name: str) -> str:
        stripped = str(requested_camera_name).strip()
        if stripped and stripped in self._camera_topics_by_name:
            return stripped

        normalized = self._normalize_camera_name(stripped)
        for candidate in self._camera_topics_by_name:
            if self._normalize_camera_name(candidate) == normalized:
                return candidate
        return self.default_camera_name

    @staticmethod
    def _normalize_camera_name(camera_name: str) -> str:
        return camera_name.strip().lower()

    @staticmethod
    def _slug(text: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9_]+", "_", text).strip("_").lower()
        return slug or "snapshot"


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = CameraSnapshotServiceNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

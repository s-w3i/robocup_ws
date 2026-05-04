#!/usr/bin/env python3
"""Low-rate relay for tracking camera topics.

Republish the latest color, depth, and camera info messages at bounded rates so
remote consumers like DeepSORT do not subscribe directly to the full-rate raw
camera streams over DDS.
"""

from __future__ import annotations

import copy
import threading

import cv2
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image


class TrackingInputRelay(Node):
    def __init__(self) -> None:
        super().__init__("tracking_input_relay")

        self.declare_parameter("input_color_topic", "/camera0/color/image_raw")
        self.declare_parameter("input_depth_topic", "/camera0/realsense_splitter_node/output/depth")
        self.declare_parameter("input_camera_info_topic", "/camera0/color/camera_info")
        self.declare_parameter("output_color_topic", "/camera0/tracking/color/image_raw")
        self.declare_parameter("output_depth_topic", "/camera0/tracking/depth/image_raw")
        self.declare_parameter("output_camera_info_topic", "/camera0/tracking/color/camera_info")
        self.declare_parameter("color_rate_hz", 8.0)
        self.declare_parameter("depth_rate_hz", 4.0)
        self.declare_parameter("output_width", 320)
        self.declare_parameter("output_height", 240)

        self.input_color_topic = str(self.get_parameter("input_color_topic").value)
        self.input_depth_topic = str(self.get_parameter("input_depth_topic").value)
        self.input_camera_info_topic = str(self.get_parameter("input_camera_info_topic").value)
        self.output_color_topic = str(self.get_parameter("output_color_topic").value)
        self.output_depth_topic = str(self.get_parameter("output_depth_topic").value)
        self.output_camera_info_topic = str(self.get_parameter("output_camera_info_topic").value)
        self.color_rate_hz = max(0.0, float(self.get_parameter("color_rate_hz").value))
        self.depth_rate_hz = max(0.0, float(self.get_parameter("depth_rate_hz").value))
        self.output_width = max(0, int(self.get_parameter("output_width").value))
        self.output_height = max(0, int(self.get_parameter("output_height").value))

        self._lock = threading.Lock()
        self._bridge = CvBridge()
        self._latest_color: Image | None = None
        self._latest_depth: Image | None = None
        self._latest_camera_info: CameraInfo | None = None

        self._color_pub = self.create_publisher(Image, self.output_color_topic, qos_profile_sensor_data)
        self._depth_pub = self.create_publisher(Image, self.output_depth_topic, qos_profile_sensor_data)
        self._camera_info_pub = self.create_publisher(
            CameraInfo, self.output_camera_info_topic, qos_profile_sensor_data
        )

        self.create_subscription(Image, self.input_color_topic, self._on_color, qos_profile_sensor_data)
        self.create_subscription(Image, self.input_depth_topic, self._on_depth, qos_profile_sensor_data)
        self.create_subscription(
            CameraInfo, self.input_camera_info_topic, self._on_camera_info, qos_profile_sensor_data
        )

        self._color_timer = None
        self._depth_timer = None
        if self.color_rate_hz > 0.0:
            self._color_timer = self.create_timer(1.0 / self.color_rate_hz, self._publish_color_bundle)
        if self.depth_rate_hz > 0.0:
            self._depth_timer = self.create_timer(1.0 / self.depth_rate_hz, self._publish_depth)

        self.get_logger().info(
            "Tracking relay configured: "
            f"color {self.input_color_topic} -> {self.output_color_topic} @ {self.color_rate_hz:.2f} Hz, "
            f"depth {self.input_depth_topic} -> {self.output_depth_topic} @ {self.depth_rate_hz:.2f} Hz, "
            f"camera_info {self.input_camera_info_topic} -> {self.output_camera_info_topic}, "
            f"size={self.output_width}x{self.output_height}"
        )

    def _on_color(self, msg: Image) -> None:
        with self._lock:
            self._latest_color = msg

    def _on_depth(self, msg: Image) -> None:
        with self._lock:
            self._latest_depth = msg

    def _on_camera_info(self, msg: CameraInfo) -> None:
        with self._lock:
            self._latest_camera_info = msg

    def _publish_color_bundle(self) -> None:
        with self._lock:
            color_msg = self._latest_color
            camera_info_msg = self._latest_camera_info

        if color_msg is not None:
            resized_color = self._resize_image_msg(color_msg, is_depth=False)
            if resized_color is not None:
                self._color_pub.publish(resized_color)
        if camera_info_msg is not None:
            resized_info = self._resize_camera_info(camera_info_msg)
            self._camera_info_pub.publish(resized_info)

    def _publish_depth(self) -> None:
        with self._lock:
            depth_msg = self._latest_depth

        if depth_msg is not None:
            resized_depth = self._resize_image_msg(depth_msg, is_depth=True)
            if resized_depth is not None:
                self._depth_pub.publish(resized_depth)

    def _resize_image_msg(self, msg: Image, is_depth: bool) -> Image | None:
        if self.output_width <= 0 or self.output_height <= 0:
            return msg

        if msg.width == self.output_width and msg.height == self.output_height:
            return msg

        try:
            image = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except CvBridgeError as exc:
            self.get_logger().warn(f"Failed to decode {'depth' if is_depth else 'color'} image: {exc}")
            return None

        interpolation = cv2.INTER_NEAREST if is_depth else cv2.INTER_AREA
        resized = cv2.resize(image, (self.output_width, self.output_height), interpolation=interpolation)

        try:
            out_msg = self._bridge.cv2_to_imgmsg(resized, encoding=msg.encoding)
        except CvBridgeError as exc:
            self.get_logger().warn(f"Failed to encode {'depth' if is_depth else 'color'} image: {exc}")
            return None

        out_msg.header = msg.header
        return out_msg

    def _resize_camera_info(self, msg: CameraInfo) -> CameraInfo:
        if self.output_width <= 0 or self.output_height <= 0:
            return msg

        if msg.width == 0 or msg.height == 0:
            resized = copy.deepcopy(msg)
            resized.width = self.output_width
            resized.height = self.output_height
            return resized

        if msg.width == self.output_width and msg.height == self.output_height:
            return msg

        scale_x = float(self.output_width) / float(msg.width)
        scale_y = float(self.output_height) / float(msg.height)

        resized = copy.deepcopy(msg)
        resized.width = self.output_width
        resized.height = self.output_height

        resized.k[0] *= scale_x
        resized.k[2] *= scale_x
        resized.k[4] *= scale_y
        resized.k[5] *= scale_y

        resized.p[0] *= scale_x
        resized.p[2] *= scale_x
        resized.p[5] *= scale_y
        resized.p[6] *= scale_y

        if resized.roi.width > 0 and resized.roi.height > 0:
            resized.roi.x_offset = int(round(resized.roi.x_offset * scale_x))
            resized.roi.y_offset = int(round(resized.roi.y_offset * scale_y))
            resized.roi.width = int(round(resized.roi.width * scale_x))
            resized.roi.height = int(round(resized.roi.height * scale_y))

        return resized


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = TrackingInputRelay()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

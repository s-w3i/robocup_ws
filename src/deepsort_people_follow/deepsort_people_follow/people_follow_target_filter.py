#!/usr/bin/env python3
"""Predictive filtered follow-target generator for Nav2-based people following."""

from __future__ import annotations

import math
import time

from geometry_msgs.msg import PoseStamped, TransformStamped
import rclpy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time
from std_srvs.srv import SetBool
from tf2_ros import Buffer, TransformBroadcaster, TransformException, TransformListener


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


class PeopleFollowTargetFilter(Node):
    """Filter and predict the tracked person pose into a smoother follow target."""

    def __init__(self) -> None:
        super().__init__("people_follow_target_filter")

        self.declare_parameter("enabled", True)
        self.declare_parameter("global_frame", "map")
        self.declare_parameter("source_frame", "follow_target")
        self.declare_parameter("filtered_frame", "follow_target_filtered")
        self.declare_parameter("filtered_pose_topic", "/people/follow_target_filtered")
        self.declare_parameter("enable_service_name", "/people_follow_target_filter/set_enabled")
        self.declare_parameter("publish_rate_hz", 15.0)
        self.declare_parameter("transform_timeout_s", 0.1)
        self.declare_parameter("target_lost_timeout_s", 1.0)
        self.declare_parameter("prediction_horizon_s", 0.25)
        self.declare_parameter("position_gain", 0.65)
        self.declare_parameter("velocity_gain", 0.20)
        self.declare_parameter("max_target_speed_mps", 1.8)
        self.declare_parameter("reset_distance_m", 1.2)
        self.declare_parameter("measurement_outlier_distance_m", 1.0)
        self.declare_parameter("measurement_outlier_required_count", 3)

        self.enabled = bool(self.get_parameter("enabled").value)
        self.global_frame = str(self.get_parameter("global_frame").value)
        self.source_frame = str(self.get_parameter("source_frame").value)
        self.filtered_frame = str(self.get_parameter("filtered_frame").value)
        self.filtered_pose_topic = str(self.get_parameter("filtered_pose_topic").value)
        self.enable_service_name = str(self.get_parameter("enable_service_name").value)
        self.publish_rate_hz = max(1.0, float(self.get_parameter("publish_rate_hz").value))
        self.transform_timeout = Duration(seconds=float(self.get_parameter("transform_timeout_s").value))
        self.target_lost_timeout_s = max(0.0, float(self.get_parameter("target_lost_timeout_s").value))
        self.prediction_horizon_s = max(0.0, float(self.get_parameter("prediction_horizon_s").value))
        self.position_gain = clamp(float(self.get_parameter("position_gain").value), 0.0, 1.0)
        self.velocity_gain = clamp(float(self.get_parameter("velocity_gain").value), 0.0, 1.0)
        self.max_target_speed_mps = max(0.0, float(self.get_parameter("max_target_speed_mps").value))
        self.reset_distance_m = max(0.0, float(self.get_parameter("reset_distance_m").value))
        self.measurement_outlier_distance_m = max(
            0.0, float(self.get_parameter("measurement_outlier_distance_m").value)
        )
        self.measurement_outlier_required_count = max(
            1, int(self.get_parameter("measurement_outlier_required_count").value)
        )

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._tf_broadcaster = TransformBroadcaster(self)
        self._pose_pub = self.create_publisher(PoseStamped, self.filtered_pose_topic, 10)
        self._enable_service = self.create_service(
            SetBool, self.enable_service_name, self._handle_set_enabled
        )
        self._timer = self.create_timer(1.0 / self.publish_rate_hz, self._timer_cb)

        self._state_valid = False
        self._state_x = 0.0
        self._state_y = 0.0
        self._vel_x = 0.0
        self._vel_y = 0.0
        self._pending_outlier_x: float | None = None
        self._pending_outlier_y: float | None = None
        self._pending_outlier_count = 0
        self._last_update_monotonic = time.monotonic()
        self._last_measurement_monotonic = 0.0
        self._warned_target_lost = False

        self.get_logger().info(f"Enabled at startup: {self.enabled}")
        self.get_logger().info(f"Source frame: {self.source_frame}")
        self.get_logger().info(f"Filtered frame: {self.filtered_frame}")
        self.get_logger().info(f"Global frame: {self.global_frame}")
        self.get_logger().info(f"Prediction horizon: {self.prediction_horizon_s:.2f}s")

    def _handle_set_enabled(self, request: SetBool.Request, response: SetBool.Response) -> SetBool.Response:
        self.enabled = bool(request.data)
        self._warned_target_lost = False
        if not self.enabled:
            self._state_valid = False
            self._vel_x = 0.0
            self._vel_y = 0.0
            self._clear_pending_outlier()
            response.message = "Predictive follow-target filter disabled."
        else:
            response.message = "Predictive follow-target filter enabled."
        response.success = True
        self.get_logger().info(response.message)
        return response

    def _clear_pending_outlier(self) -> None:
        self._pending_outlier_x = None
        self._pending_outlier_y = None
        self._pending_outlier_count = 0

    def _lookup_source_pose(self) -> tuple[float, float, Time] | None:
        try:
            transform = self._tf_buffer.lookup_transform(
                self.global_frame,
                self.source_frame,
                Time(),
                timeout=self.transform_timeout,
            )
        except TransformException as exc:
            self.get_logger().debug(
                f"TF lookup failed ({self.global_frame} <- {self.source_frame}): {exc}"
            )
            return None

        return (
            float(transform.transform.translation.x),
            float(transform.transform.translation.y),
            transform.header.stamp,
        )

    def _publish_filtered_target(self, stamp: Time, x: float, y: float) -> None:
        del stamp

        current_stamp = self.get_clock().now().to_msg()
        pose = PoseStamped()
        pose.header.frame_id = self.global_frame
        pose.header.stamp = current_stamp
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = 0.0
        pose.pose.orientation.w = 1.0
        self._pose_pub.publish(pose)

        tf_msg = TransformStamped()
        tf_msg.header = pose.header
        tf_msg.child_frame_id = self.filtered_frame
        tf_msg.transform.translation.x = x
        tf_msg.transform.translation.y = y
        tf_msg.transform.translation.z = 0.0
        tf_msg.transform.rotation.w = 1.0
        self._tf_broadcaster.sendTransform(tf_msg)

    def _timer_cb(self) -> None:
        now_monotonic = time.monotonic()
        dt = max(1e-3, now_monotonic - self._last_update_monotonic)
        self._last_update_monotonic = now_monotonic

        if not self.enabled:
            return

        source_pose = self._lookup_source_pose()
        if source_pose is not None:
            meas_x, meas_y, stamp_msg = source_pose
            stamp = Time.from_msg(stamp_msg)
            if not self._state_valid:
                self._state_x = meas_x
                self._state_y = meas_y
                self._vel_x = 0.0
                self._vel_y = 0.0
                self._state_valid = True
                self._clear_pending_outlier()
                self._last_measurement_monotonic = now_monotonic
                self._warned_target_lost = False
            else:
                predicted_x = self._state_x + self._vel_x * dt
                predicted_y = self._state_y + self._vel_y * dt
                residual_x = meas_x - predicted_x
                residual_y = meas_y - predicted_y
                residual_distance = math.hypot(residual_x, residual_y)

                if (
                    self.measurement_outlier_distance_m > 0.0
                    and residual_distance >= self.measurement_outlier_distance_m
                ):
                    cluster_radius = max(0.25, 0.5 * self.measurement_outlier_distance_m)
                    if (
                        self._pending_outlier_x is not None
                        and self._pending_outlier_y is not None
                        and math.hypot(
                            meas_x - self._pending_outlier_x,
                            meas_y - self._pending_outlier_y,
                        )
                        <= cluster_radius
                    ):
                        self._pending_outlier_count += 1
                    else:
                        self._pending_outlier_count = 1

                    self._pending_outlier_x = meas_x
                    self._pending_outlier_y = meas_y

                    if self._pending_outlier_count < self.measurement_outlier_required_count:
                        # Hold the current predicted track until the large jump is consistent
                        # across several frames; this filters out one-off ID swaps.
                        self._state_x = predicted_x
                        self._state_y = predicted_y
                    else:
                        self.get_logger().warn(
                            "Accepting large follow-target jump after confirmation. "
                            f"jump={residual_distance:.2f}m "
                            f"confirmations={self._pending_outlier_count}"
                        )
                        self._state_x = meas_x
                        self._state_y = meas_y
                        self._vel_x = 0.0
                        self._vel_y = 0.0
                        self._clear_pending_outlier()
                        self._last_measurement_monotonic = now_monotonic
                        self._warned_target_lost = False
                elif self.reset_distance_m > 0.0 and residual_distance >= self.reset_distance_m:
                    self._state_x = meas_x
                    self._state_y = meas_y
                    self._vel_x = 0.0
                    self._vel_y = 0.0
                    self._clear_pending_outlier()
                    self._last_measurement_monotonic = now_monotonic
                    self._warned_target_lost = False
                else:
                    self._clear_pending_outlier()
                    self._state_x = predicted_x + self.position_gain * residual_x
                    self._state_y = predicted_y + self.position_gain * residual_y
                    velocity_dt = max(dt, 1e-3)
                    self._vel_x = self._vel_x + (self.velocity_gain * residual_x / velocity_dt)
                    self._vel_y = self._vel_y + (self.velocity_gain * residual_y / velocity_dt)
                    speed = math.hypot(self._vel_x, self._vel_y)
                    if self.max_target_speed_mps > 0.0 and speed > self.max_target_speed_mps:
                        scale = self.max_target_speed_mps / max(speed, 1e-6)
                        self._vel_x *= scale
                        self._vel_y *= scale
                    self._last_measurement_monotonic = now_monotonic
                    self._warned_target_lost = False
        elif not self._state_valid:
            return
        else:
            elapsed = now_monotonic - self._last_measurement_monotonic
            if self.target_lost_timeout_s > 0.0 and elapsed > self.target_lost_timeout_s:
                if not self._warned_target_lost:
                    self.get_logger().warn(
                        f"Lost source frame '{self.source_frame}' for {elapsed:.2f}s; stopping filtered target output."
                    )
                    self._warned_target_lost = True
                self._state_valid = False
                self._vel_x = 0.0
                self._vel_y = 0.0
                self._clear_pending_outlier()
                return
            stamp = self.get_clock().now()

        publish_x = self._state_x + self._vel_x * self.prediction_horizon_s
        publish_y = self._state_y + self._vel_y * self.prediction_horizon_s
        self._publish_filtered_target(stamp, publish_x, publish_y)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = PeopleFollowTargetFilter()
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

#!/usr/bin/env python3
"""Experimental close-range people follower that publishes cmd_vel directly."""

from __future__ import annotations

import math
import time

from geometry_msgs.msg import Twist
import rclpy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time
from std_srvs.srv import SetBool
from tf2_ros import Buffer, TransformException, TransformListener


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def rate_limit(target: float, current: float, max_delta: float) -> float:
    if target > current + max_delta:
        return current + max_delta
    if target < current - max_delta:
        return current - max_delta
    return target


class PeopleFollowCmdVelNode(Node):
    """Direct follower for a visible tracked person."""

    def __init__(self) -> None:
        super().__init__("people_follow_cmd_vel")

        self.declare_parameter("enabled", False)
        self.declare_parameter("robot_base_frame", "base_footprint")
        self.declare_parameter("tracked_frame", "follow_target")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("enable_service_name", "/people_follow_cmd_vel/set_enabled")
        self.declare_parameter("control_rate_hz", 15.0)
        self.declare_parameter("transform_timeout_s", 0.1)
        self.declare_parameter("target_lost_timeout_s", 0.75)
        self.declare_parameter("follow_standoff_distance_m", 0.7)
        self.declare_parameter("follow_distance_tolerance_m", 0.10)
        self.declare_parameter("bearing_deadband_rad", 0.05)
        self.declare_parameter("rotate_in_place_min_angle_rad", 0.45)
        self.declare_parameter("linear_heading_slowdown_angle_rad", 0.80)
        self.declare_parameter("min_target_distance_m", 0.20)
        self.declare_parameter("linear_kp", 0.9)
        self.declare_parameter("angular_kp", 1.8)
        self.declare_parameter("max_linear_speed_mps", 0.35)
        self.declare_parameter("max_angular_speed_radps", 1.2)
        self.declare_parameter("max_linear_accel_mps2", 0.8)
        self.declare_parameter("max_angular_accel_radps2", 2.5)

        self.enabled = bool(self.get_parameter("enabled").value)
        self.robot_base_frame = str(self.get_parameter("robot_base_frame").value)
        self.tracked_frame = str(self.get_parameter("tracked_frame").value)
        self.cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        self.enable_service_name = str(self.get_parameter("enable_service_name").value)
        self.control_rate_hz = max(1.0, float(self.get_parameter("control_rate_hz").value))
        self.transform_timeout = Duration(seconds=float(self.get_parameter("transform_timeout_s").value))
        self.target_lost_timeout_s = max(0.0, float(self.get_parameter("target_lost_timeout_s").value))
        self.follow_standoff_distance_m = max(
            0.0, float(self.get_parameter("follow_standoff_distance_m").value)
        )
        self.follow_distance_tolerance_m = max(
            0.0, float(self.get_parameter("follow_distance_tolerance_m").value)
        )
        self.bearing_deadband_rad = max(0.0, float(self.get_parameter("bearing_deadband_rad").value))
        self.rotate_in_place_min_angle_rad = max(
            0.0, float(self.get_parameter("rotate_in_place_min_angle_rad").value)
        )
        self.linear_heading_slowdown_angle_rad = max(
            self.bearing_deadband_rad, float(self.get_parameter("linear_heading_slowdown_angle_rad").value)
        )
        self.min_target_distance_m = max(0.0, float(self.get_parameter("min_target_distance_m").value))
        self.linear_kp = max(0.0, float(self.get_parameter("linear_kp").value))
        self.angular_kp = max(0.0, float(self.get_parameter("angular_kp").value))
        self.max_linear_speed_mps = max(0.0, float(self.get_parameter("max_linear_speed_mps").value))
        self.max_angular_speed_radps = max(0.0, float(self.get_parameter("max_angular_speed_radps").value))
        self.max_linear_accel_mps2 = max(0.0, float(self.get_parameter("max_linear_accel_mps2").value))
        self.max_angular_accel_radps2 = max(0.0, float(self.get_parameter("max_angular_accel_radps2").value))

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self._enable_service = self.create_service(
            SetBool, self.enable_service_name, self._handle_set_enabled
        )
        self._timer = self.create_timer(1.0 / self.control_rate_hz, self._timer_cb)

        self._last_target_seen_monotonic = 0.0
        self._last_control_monotonic = time.monotonic()
        self._current_linear = 0.0
        self._current_angular = 0.0
        self._warned_target_lost = False

        self.get_logger().info(f"Enabled at startup: {self.enabled}")
        self.get_logger().info(f"Target frame: {self.tracked_frame}")
        self.get_logger().info(f"Robot base frame: {self.robot_base_frame}")
        self.get_logger().info(f"cmd_vel topic: {self.cmd_vel_topic}")
        self.get_logger().info(f"Stand-off distance: {self.follow_standoff_distance_m:.2f} m")

    def _handle_set_enabled(self, request: SetBool.Request, response: SetBool.Response) -> SetBool.Response:
        self.enabled = bool(request.data)
        self._warned_target_lost = False
        if self.enabled:
            response.success = True
            response.message = "Direct cmd_vel people-follow enabled."
            self.get_logger().info(response.message)
        else:
            self._publish_stop(reset_state=True)
            response.success = True
            response.message = "Direct cmd_vel people-follow disabled."
            self.get_logger().info(response.message)
        return response

    def _lookup_target_in_base(self):
        try:
            return self._tf_buffer.lookup_transform(
                self.robot_base_frame,
                self.tracked_frame,
                Time(),
                timeout=self.transform_timeout,
            )
        except TransformException as exc:
            self.get_logger().debug(
                f"TF lookup failed ({self.robot_base_frame} <- {self.tracked_frame}): {exc}"
            )
            return None

    def _publish_cmd(self, linear_x: float, angular_z: float) -> None:
        msg = Twist()
        msg.linear.x = float(linear_x)
        msg.angular.z = float(angular_z)
        self._cmd_pub.publish(msg)

    def _publish_stop(self, reset_state: bool = False) -> None:
        self._publish_cmd(0.0, 0.0)
        if reset_state:
            self._current_linear = 0.0
            self._current_angular = 0.0

    def _timer_cb(self) -> None:
        now_monotonic = time.monotonic()
        dt = max(1e-3, now_monotonic - self._last_control_monotonic)
        self._last_control_monotonic = now_monotonic

        if not self.enabled:
            return

        target_tf = self._lookup_target_in_base()
        if target_tf is None:
            elapsed = now_monotonic - self._last_target_seen_monotonic
            if self.target_lost_timeout_s <= 0.0 or elapsed >= self.target_lost_timeout_s:
                if not self._warned_target_lost:
                    self.get_logger().warn(
                        f"Lost tracked frame '{self.tracked_frame}' for {elapsed:.2f}s; stopping cmd_vel follow."
                    )
                    self._warned_target_lost = True
                self._publish_stop(reset_state=True)
            return

        self._last_target_seen_monotonic = now_monotonic
        self._warned_target_lost = False

        target_x = float(target_tf.transform.translation.x)
        target_y = float(target_tf.transform.translation.y)
        distance = math.hypot(target_x, target_y)
        bearing = math.atan2(target_y, target_x)

        target_linear = 0.0
        target_angular = 0.0

        if abs(bearing) > self.bearing_deadband_rad:
            target_angular = clamp(
                self.angular_kp * bearing,
                -self.max_angular_speed_radps,
                self.max_angular_speed_radps,
            )

        distance_error = distance - self.follow_standoff_distance_m
        should_approach = (
            distance > self.min_target_distance_m
            and distance_error > self.follow_distance_tolerance_m
        )
        if should_approach:
            target_linear = clamp(
                self.linear_kp * (distance_error - self.follow_distance_tolerance_m),
                0.0,
                self.max_linear_speed_mps,
            )
            if abs(bearing) >= self.rotate_in_place_min_angle_rad:
                target_linear = 0.0
            else:
                slowdown_ratio = clamp(
                    1.0 - (abs(bearing) / max(1e-3, self.linear_heading_slowdown_angle_rad)),
                    0.0,
                    1.0,
                )
                target_linear *= slowdown_ratio

        max_linear_delta = self.max_linear_accel_mps2 * dt
        max_angular_delta = self.max_angular_accel_radps2 * dt
        self._current_linear = rate_limit(target_linear, self._current_linear, max_linear_delta)
        self._current_angular = rate_limit(target_angular, self._current_angular, max_angular_delta)
        self._publish_cmd(self._current_linear, self._current_angular)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = PeopleFollowCmdVelNode()
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

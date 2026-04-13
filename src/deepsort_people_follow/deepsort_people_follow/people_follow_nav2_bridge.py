#!/usr/bin/env python3
"""Bridge DeepSORT follow TF into a Nav2 follow BT session."""

from __future__ import annotations

import math
from pathlib import Path

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory

from geometry_msgs.msg import PoseStamped, Quaternion
from nav2_msgs.action import NavigateToPose
import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time
from std_srvs.srv import SetBool
from tf2_ros import Buffer, TransformException, TransformListener


class PeopleFollowNav2Bridge(Node):
    """Owns a follow-mode Nav2 session while streaming dynamic goal updates."""

    def __init__(self) -> None:
        super().__init__("people_follow_nav2_bridge")

        self.declare_parameter("enabled", False)
        self.declare_parameter("global_frame", "map")
        self.declare_parameter("robot_base_frame", "base_footprint")
        self.declare_parameter("tracked_frame", "follow_target")
        self.declare_parameter("goal_update_topic", "/goal_update")
        self.declare_parameter("navigate_to_pose_action", "/navigate_to_pose")
        self.declare_parameter("enable_service_name", "~/set_enabled")
        self.declare_parameter("goal_update_rate_hz", 5.0)
        self.declare_parameter("transform_timeout_s", 0.2)
        self.declare_parameter("target_lost_timeout_s", 2.0)
        self.declare_parameter("behavior_tree_path", "")
        self.declare_parameter("behavior_tree_package", "deepsort_people_follow")
        self.declare_parameter("behavior_tree_relative_path", "bt/follower_w_recovery.xml")

        self.enabled = bool(self.get_parameter("enabled").value)
        self.global_frame = str(self.get_parameter("global_frame").value)
        self.robot_base_frame = str(self.get_parameter("robot_base_frame").value)
        self.tracked_frame = str(self.get_parameter("tracked_frame").value)
        self.goal_update_topic = str(self.get_parameter("goal_update_topic").value)
        self.navigate_to_pose_action = str(self.get_parameter("navigate_to_pose_action").value)
        self.enable_service_name = str(self.get_parameter("enable_service_name").value)
        self.goal_update_rate_hz = max(0.5, float(self.get_parameter("goal_update_rate_hz").value))
        self.transform_timeout = Duration(seconds=float(self.get_parameter("transform_timeout_s").value))
        self.target_lost_timeout = max(0.0, float(self.get_parameter("target_lost_timeout_s").value))
        self.behavior_tree_path = self._resolve_behavior_tree_path(
            explicit_path=str(self.get_parameter("behavior_tree_path").value),
            package_name=str(self.get_parameter("behavior_tree_package").value),
            relative_path=str(self.get_parameter("behavior_tree_relative_path").value),
        )

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._goal_update_pub = self.create_publisher(PoseStamped, self.goal_update_topic, 10)
        self._enable_service = self.create_service(SetBool, self.enable_service_name, self._handle_set_enabled)
        self._navigate_client = ActionClient(self, NavigateToPose, self.navigate_to_pose_action)
        self._timer = self.create_timer(1.0 / self.goal_update_rate_hz, self._timer_cb)

        self._goal_handle = None
        self._pending_goal_future = None
        self._result_future = None
        self._cancel_future = None
        self._last_target_pose: PoseStamped | None = None
        self._last_target_seen_time = self.get_clock().now()
        self._warned_missing_bt = False
        self._warned_server_not_ready = False
        self._warned_target_lost = False

        self.get_logger().info(f"Enabled at startup: {self.enabled}")
        self.get_logger().info(f"Tracked frame: {self.tracked_frame}")
        self.get_logger().info(f"Nav2 action: {self.navigate_to_pose_action}")
        self.get_logger().info(f"Goal update topic: {self.goal_update_topic}")
        if self.behavior_tree_path:
            self.get_logger().info(f"Follow BT: {self.behavior_tree_path}")
        else:
            self.get_logger().warn(
                "No follow behavior tree path resolved. Set 'behavior_tree_path' before enabling people follow."
            )

    def _resolve_behavior_tree_path(
        self,
        explicit_path: str,
        package_name: str,
        relative_path: str,
    ) -> str:
        explicit = explicit_path.strip()
        if explicit:
            candidate = Path(explicit)
            if candidate.is_file():
                return str(candidate)
            self.get_logger().warn(f"Configured behavior tree path does not exist: {candidate}")

        try:
            package_share = Path(get_package_share_directory(package_name))
            candidate = package_share / relative_path
            if candidate.is_file():
                return str(candidate)
            self.get_logger().warn(f"Behavior tree not found under package share: {candidate}")
        except PackageNotFoundError:
            self.get_logger().warn(
                f"Package '{package_name}' not found in ament index while resolving the follow behavior tree."
            )

        source_tree_candidate = Path("/home/usern/robocup_ws/src/deepsort_people_follow") / relative_path
        if package_name == "deepsort_people_follow" and source_tree_candidate.is_file():
            return str(source_tree_candidate)

        return ""

    def _handle_set_enabled(self, request: SetBool.Request, response: SetBool.Response) -> SetBool.Response:
        self.enabled = bool(request.data)
        if self.enabled:
            response.success = True
            response.message = "People-follow mode enabled. Waiting for tracked target before starting Nav2 follow BT."
            self.get_logger().info(response.message)
        else:
            self._cancel_follow_goal()
            response.success = True
            response.message = "People-follow mode disabled. Active follow goal canceled."
            self.get_logger().info(response.message)
        return response

    def _timer_cb(self) -> None:
        if not self.enabled:
            return

        if not self.behavior_tree_path:
            if not self._warned_missing_bt:
                self.get_logger().warn("People-follow requested, but no follow behavior tree path is available.")
                self._warned_missing_bt = True
            return
        self._warned_missing_bt = False

        target_pose = self._lookup_target_pose()
        now = self.get_clock().now()
        if target_pose is not None:
            self._last_target_pose = target_pose
            self._last_target_seen_time = now
            self._warned_target_lost = False
        elif self._last_target_pose is None:
            return
        elif self.target_lost_timeout > 0.0:
            elapsed = (now - self._last_target_seen_time).nanoseconds / 1e9
            if elapsed > self.target_lost_timeout:
                if not self._warned_target_lost:
                    self.get_logger().warn(
                        f"Lost tracked frame '{self.tracked_frame}' for {elapsed:.1f}s; canceling follow goal."
                    )
                    self._warned_target_lost = True
                self._cancel_follow_goal()
                self._last_target_pose = None
                return

        if self._last_target_pose is None:
            return

        if self._goal_handle is None and self._pending_goal_future is None:
            self._send_initial_goal(self._last_target_pose)
            return

        if self._goal_handle is not None:
            self._goal_update_pub.publish(self._last_target_pose)

    def _lookup_target_pose(self) -> PoseStamped | None:
        try:
            target_tf = self._tf_buffer.lookup_transform(
                self.global_frame,
                self.tracked_frame,
                Time(),
                timeout=self.transform_timeout,
            )
        except TransformException as exc:
            self.get_logger().debug(
                f"TF lookup failed ({self.global_frame} <- {self.tracked_frame}): {exc}"
            )
            return None

        pose = PoseStamped()
        pose.header.stamp = target_tf.header.stamp
        pose.header.frame_id = self.global_frame
        pose.pose.position.x = float(target_tf.transform.translation.x)
        pose.pose.position.y = float(target_tf.transform.translation.y)
        pose.pose.position.z = float(target_tf.transform.translation.z)
        pose.pose.orientation = self._compute_goal_orientation(
            target_x=pose.pose.position.x,
            target_y=pose.pose.position.y,
        )
        return pose

    def _compute_goal_orientation(self, target_x: float, target_y: float) -> Quaternion:
        yaw = 0.0
        try:
            robot_tf = self._tf_buffer.lookup_transform(
                self.global_frame,
                self.robot_base_frame,
                Time(),
                timeout=self.transform_timeout,
            )
            dx = target_x - float(robot_tf.transform.translation.x)
            dy = target_y - float(robot_tf.transform.translation.y)
            if math.hypot(dx, dy) > 1e-3:
                yaw = math.atan2(dy, dx)
        except TransformException as exc:
            self.get_logger().debug(
                f"TF lookup failed ({self.global_frame} <- {self.robot_base_frame}) for yaw estimate: {exc}"
            )

        half_yaw = 0.5 * yaw
        return Quaternion(x=0.0, y=0.0, z=math.sin(half_yaw), w=math.cos(half_yaw))

    def _send_initial_goal(self, target_pose: PoseStamped) -> None:
        if not self._navigate_client.wait_for_server(timeout_sec=0.0):
            if not self._warned_server_not_ready:
                self.get_logger().warn("NavigateToPose action server is not ready yet.")
                self._warned_server_not_ready = True
            return
        self._warned_server_not_ready = False

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = target_pose
        goal_msg.behavior_tree = self.behavior_tree_path

        self.get_logger().info(
            "Starting people-follow Nav2 session with follower BT. "
            f"target=({target_pose.pose.position.x:.2f}, {target_pose.pose.position.y:.2f})"
        )
        self._pending_goal_future = self._navigate_client.send_goal_async(goal_msg)
        self._pending_goal_future.add_done_callback(self._on_goal_response)

    def _on_goal_response(self, future) -> None:
        self._pending_goal_future = None
        try:
            goal_handle = future.result()
        except Exception as exc:  # pragma: no cover - defensive logging for middleware errors
            self.get_logger().error(f"Failed to send follow goal to Nav2: {exc}")
            return

        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().warn("Nav2 rejected the people-follow goal request.")
            return

        self._goal_handle = goal_handle
        self.get_logger().info("People-follow goal accepted by Nav2.")
        self._result_future = goal_handle.get_result_async()
        self._result_future.add_done_callback(self._on_goal_result)

        if not self.enabled:
            self._cancel_follow_goal()

    def _on_goal_result(self, future) -> None:
        self._goal_handle = None
        self._result_future = None

        try:
            result = future.result()
            status = result.status
        except Exception as exc:  # pragma: no cover - defensive logging for middleware errors
            self.get_logger().error(f"Failed to receive follow goal result: {exc}")
            return

        self.get_logger().info(f"People-follow Nav2 goal finished with status {status}.")

    def _cancel_follow_goal(self) -> None:
        if self._goal_handle is not None and self._cancel_future is None:
            self.get_logger().info("Canceling active people-follow Nav2 goal.")
            self._cancel_future = self._goal_handle.cancel_goal_async()
            self._cancel_future.add_done_callback(self._on_cancel_complete)
            return

        if self._pending_goal_future is not None:
            self.get_logger().info("Follow goal is still pending; it will be canceled as soon as Nav2 accepts it.")

    def _on_cancel_complete(self, future) -> None:
        self._cancel_future = None
        try:
            future.result()
        except Exception as exc:  # pragma: no cover - defensive logging for middleware errors
            self.get_logger().error(f"Failed to cancel people-follow goal: {exc}")
            return

        self.get_logger().info("People-follow Nav2 goal canceled.")


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = PeopleFollowNav2Bridge()
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

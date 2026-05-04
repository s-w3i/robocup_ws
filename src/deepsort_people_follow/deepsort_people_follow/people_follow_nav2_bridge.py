#!/usr/bin/env python3
"""Bridge DeepSORT follow TF into a Nav2 follow BT session."""

from __future__ import annotations

import math
from pathlib import Path

from action_msgs.msg import GoalStatus
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
        self.declare_parameter("goal_command_mode", "goal_pose")
        self.declare_parameter("goal_pose_topic", "/goal_pose")
        self.declare_parameter("goal_update_topic", "/goal_update")
        self.declare_parameter("navigate_to_pose_action", "/navigate_to_pose")
        self.declare_parameter("enable_service_name", "~/set_enabled")
        self.declare_parameter("goal_update_rate_hz", 10.0)
        self.declare_parameter("goal_republish_period_s", 0.0)
        self.declare_parameter("min_goal_translation_delta_m", 0.05)
        self.declare_parameter("min_goal_yaw_delta_rad", 0.12)
        self.declare_parameter("follow_standoff_distance_m", 0.7)
        self.declare_parameter("follow_distance_tolerance_m", 0.10)
        self.declare_parameter("target_smoothing_alpha", 0.65)
        self.declare_parameter("target_smoothing_reset_distance_m", 0.40)
        self.declare_parameter("publish_hold_goal_on_disable", True)
        self.declare_parameter("transform_timeout_s", 0.2)
        self.declare_parameter("target_lost_timeout_s", 2.0)
        self.declare_parameter("action_retry_cooldown_s", 1.0)
        self.declare_parameter("behavior_tree_path", "")
        self.declare_parameter("behavior_tree_package", "deepsort_people_follow")
        self.declare_parameter("behavior_tree_relative_path", "bt/follower_w_recovery.xml")

        self.enabled = bool(self.get_parameter("enabled").value)
        self.global_frame = str(self.get_parameter("global_frame").value)
        self.robot_base_frame = str(self.get_parameter("robot_base_frame").value)
        self.tracked_frame = str(self.get_parameter("tracked_frame").value)
        self.goal_command_mode = str(self.get_parameter("goal_command_mode").value).strip().lower()
        self.goal_pose_topic = str(self.get_parameter("goal_pose_topic").value)
        self.goal_update_topic = str(self.get_parameter("goal_update_topic").value)
        self.navigate_to_pose_action = str(self.get_parameter("navigate_to_pose_action").value)
        self.enable_service_name = str(self.get_parameter("enable_service_name").value)
        self.goal_update_rate_hz = max(0.5, float(self.get_parameter("goal_update_rate_hz").value))
        self.goal_republish_period_s = max(0.0, float(self.get_parameter("goal_republish_period_s").value))
        self.min_goal_translation_delta_m = max(0.0, float(self.get_parameter("min_goal_translation_delta_m").value))
        self.min_goal_yaw_delta_rad = max(0.0, float(self.get_parameter("min_goal_yaw_delta_rad").value))
        self.follow_standoff_distance_m = max(
            0.0, float(self.get_parameter("follow_standoff_distance_m").value)
        )
        self.follow_distance_tolerance_m = max(
            0.0, float(self.get_parameter("follow_distance_tolerance_m").value)
        )
        self.target_smoothing_alpha = min(
            1.0, max(0.0, float(self.get_parameter("target_smoothing_alpha").value))
        )
        self.target_smoothing_reset_distance_m = max(
            0.0, float(self.get_parameter("target_smoothing_reset_distance_m").value)
        )
        self.publish_hold_goal_on_disable = bool(self.get_parameter("publish_hold_goal_on_disable").value)
        self.transform_timeout = Duration(seconds=float(self.get_parameter("transform_timeout_s").value))
        self.target_lost_timeout = max(0.0, float(self.get_parameter("target_lost_timeout_s").value))
        self.action_retry_cooldown_s = max(0.0, float(self.get_parameter("action_retry_cooldown_s").value))
        self.behavior_tree_path = self._resolve_behavior_tree_path(
            explicit_path=str(self.get_parameter("behavior_tree_path").value),
            package_name=str(self.get_parameter("behavior_tree_package").value),
            relative_path=str(self.get_parameter("behavior_tree_relative_path").value),
        )

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._goal_pose_pub = self.create_publisher(PoseStamped, self.goal_pose_topic, 10)
        self._goal_update_pub = self.create_publisher(PoseStamped, self.goal_update_topic, 10)
        self._enable_service = self.create_service(SetBool, self.enable_service_name, self._handle_set_enabled)
        self._navigate_client = ActionClient(self, NavigateToPose, self.navigate_to_pose_action)
        self._timer = self.create_timer(1.0 / self.goal_update_rate_hz, self._timer_cb)

        self._goal_handle = None
        self._pending_goal_future = None
        self._result_future = None
        self._cancel_future = None
        self._last_target_pose: PoseStamped | None = None
        self._smoothed_target_pose: PoseStamped | None = None
        self._last_target_seen_time = self.get_clock().now()
        self._last_commanded_pose: PoseStamped | None = None
        self._last_commanded_time = self.get_clock().now()
        self._warned_missing_bt = False
        self._warned_server_not_ready = False
        self._warned_target_lost = False
        self._last_goal_result_status: int | None = None
        self._last_goal_result_time = self.get_clock().now()

        self.get_logger().info(f"Enabled at startup: {self.enabled}")
        self.get_logger().info(f"Tracked frame: {self.tracked_frame}")
        self.get_logger().info(f"Goal command mode: {self.goal_command_mode}")
        self.get_logger().info(f"Goal pose topic: {self.goal_pose_topic}")
        self.get_logger().info(f"Nav2 action: {self.navigate_to_pose_action}")
        self.get_logger().info(f"Goal update topic: {self.goal_update_topic}")
        self.get_logger().info(f"Follow stand-off distance: {self.follow_standoff_distance_m:.2f} m")
        if self.behavior_tree_path:
            self.get_logger().info(f"Follow BT: {self.behavior_tree_path}")
        elif self.goal_command_mode != "goal_pose":
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
            self.get_logger().warn(
                f"Configured behavior tree path does not exist locally: {candidate}. "
                "Passing it through for remote Nav2 to resolve."
            )
            return explicit

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
            self._last_target_pose = None
            self._smoothed_target_pose = None
            self._last_commanded_pose = None
            self._warned_target_lost = False
            self._last_goal_result_status = None
            response.success = True
            response.message = (
                "People-follow mode enabled. Waiting for tracked target before commanding Nav2 follow goals."
            )
            self.get_logger().info(response.message)
        else:
            self._stop_following()
            response.success = True
            response.message = "People-follow mode disabled. Follow command stopped."
            self.get_logger().info(response.message)
        return response

    def _timer_cb(self) -> None:
        if not self.enabled:
            return

        if self.goal_command_mode != "goal_pose" and not self.behavior_tree_path:
            if not self._warned_missing_bt:
                self.get_logger().warn("People-follow requested, but no follow behavior tree path is available.")
                self._warned_missing_bt = True
            return
        self._warned_missing_bt = False

        target_pose = self._lookup_target_pose()
        now = self.get_clock().now()
        if target_pose is not None:
            target_pose = self._smooth_target_pose(target_pose)
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
                        f"Lost tracked frame '{self.tracked_frame}' for {elapsed:.1f}s; stopping follow goal updates."
                    )
                    self._warned_target_lost = True
                if self.goal_command_mode == "goal_pose":
                    self._publish_hold_goal()
                else:
                    self._cancel_follow_goal()
                self._last_target_pose = None
                self._smoothed_target_pose = None
                return

        if self._last_target_pose is None:
            return

        if self.goal_command_mode == "goal_pose":
            robot_pose = self._lookup_robot_pose()
            if robot_pose is None:
                self.get_logger().debug("Skipping follow update because the robot pose is unavailable.")
                return
            command_pose, target_distance = self._build_follow_command_pose(
                target_pose=self._last_target_pose,
                robot_pose=robot_pose,
            )
            self._publish_topic_goal_if_needed(
                command_pose,
                robot_pose=robot_pose,
                target_distance=target_distance,
            )
            return

        robot_pose = self._lookup_robot_pose()
        if robot_pose is None:
            self.get_logger().debug("Skipping follow update because the robot pose is unavailable.")
            return

        command_pose, target_distance = self._build_follow_command_pose(
            target_pose=self._last_target_pose,
            robot_pose=robot_pose,
        )

        if self._goal_handle is None and self._pending_goal_future is None:
            if not self._action_retry_cooldown_elapsed():
                return
            if not self._should_publish_topic_goal(
                command_pose,
                robot_pose=robot_pose,
                target_distance=target_distance,
            ):
                return
            self._send_initial_goal(command_pose)
            return

        if self._goal_handle is not None:
            self._publish_action_goal_update_if_needed(
                command_pose,
                robot_pose=robot_pose,
                target_distance=target_distance,
            )

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
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = self.global_frame
        pose.pose.position.x = float(target_tf.transform.translation.x)
        pose.pose.position.y = float(target_tf.transform.translation.y)
        pose.pose.position.z = 0.0
        pose.pose.orientation.w = 1.0
        return pose

    def _compute_goal_orientation(
        self,
        source_x: float,
        source_y: float,
        target_x: float,
        target_y: float,
        fallback_yaw: float = 0.0,
    ) -> Quaternion:
        yaw = fallback_yaw
        dx = target_x - source_x
        dy = target_y - source_y
        if math.hypot(dx, dy) > 1e-3:
            yaw = math.atan2(dy, dx)
        half_yaw = 0.5 * yaw
        return Quaternion(x=0.0, y=0.0, z=math.sin(half_yaw), w=math.cos(half_yaw))

    def _lookup_robot_pose(self) -> PoseStamped | None:
        try:
            robot_tf = self._tf_buffer.lookup_transform(
                self.global_frame,
                self.robot_base_frame,
                Time(),
                timeout=self.transform_timeout,
            )
        except TransformException as exc:
            self.get_logger().debug(
                f"TF lookup failed ({self.global_frame} <- {self.robot_base_frame}) for robot pose: {exc}"
            )
            return None

        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = self.global_frame
        pose.pose.position.x = float(robot_tf.transform.translation.x)
        pose.pose.position.y = float(robot_tf.transform.translation.y)
        pose.pose.position.z = 0.0
        pose.pose.orientation = robot_tf.transform.rotation
        return pose

    @staticmethod
    def _copy_pose(pose: PoseStamped) -> PoseStamped:
        copied = PoseStamped()
        copied.header = pose.header
        copied.pose = pose.pose
        return copied

    def _smooth_target_pose(self, target_pose: PoseStamped) -> PoseStamped:
        if self.target_smoothing_alpha >= 1.0 or self._smoothed_target_pose is None:
            self._smoothed_target_pose = self._copy_pose(target_pose)
            return self._copy_pose(target_pose)

        previous = self._smoothed_target_pose
        dx = float(target_pose.pose.position.x) - float(previous.pose.position.x)
        dy = float(target_pose.pose.position.y) - float(previous.pose.position.y)
        if self.target_smoothing_reset_distance_m > 0.0 and math.hypot(dx, dy) >= self.target_smoothing_reset_distance_m:
            self._smoothed_target_pose = self._copy_pose(target_pose)
            return self._copy_pose(target_pose)

        alpha = self.target_smoothing_alpha
        smoothed = PoseStamped()
        smoothed.header = target_pose.header
        smoothed.pose.position.x = alpha * float(target_pose.pose.position.x) + (1.0 - alpha) * float(
            previous.pose.position.x
        )
        smoothed.pose.position.y = alpha * float(target_pose.pose.position.y) + (1.0 - alpha) * float(
            previous.pose.position.y
        )
        smoothed.pose.position.z = 0.0
        smoothed.pose.orientation.w = 1.0
        self._smoothed_target_pose = self._copy_pose(smoothed)
        return smoothed

    def _build_follow_command_pose(
        self,
        target_pose: PoseStamped,
        robot_pose: PoseStamped,
    ) -> tuple[PoseStamped, float]:
        target_x = float(target_pose.pose.position.x)
        target_y = float(target_pose.pose.position.y)
        robot_x = float(robot_pose.pose.position.x)
        robot_y = float(robot_pose.pose.position.y)
        target_distance = math.hypot(target_x - robot_x, target_y - robot_y)

        command_pose = PoseStamped()
        command_pose.header.stamp = self.get_clock().now().to_msg()
        command_pose.header.frame_id = self.global_frame
        command_pose.pose.position.z = 0.0

        hold_upper_bound = self.follow_standoff_distance_m + self.follow_distance_tolerance_m
        if target_distance <= hold_upper_bound:
            command_pose.pose.position.x = robot_x
            command_pose.pose.position.y = robot_y
            command_pose.pose.orientation = self._compute_goal_orientation(
                source_x=robot_x,
                source_y=robot_y,
                target_x=target_x,
                target_y=target_y,
                fallback_yaw=self._pose_yaw(robot_pose),
            )
            return command_pose, target_distance

        if target_distance <= 1e-3:
            command_pose.pose.position.x = robot_x
            command_pose.pose.position.y = robot_y
            command_pose.pose.orientation = robot_pose.pose.orientation
            return command_pose, target_distance

        offset_x = (robot_x - target_x) / target_distance
        offset_y = (robot_y - target_y) / target_distance
        command_pose.pose.position.x = target_x + offset_x * self.follow_standoff_distance_m
        command_pose.pose.position.y = target_y + offset_y * self.follow_standoff_distance_m
        command_pose.pose.orientation = self._compute_goal_orientation(
            source_x=command_pose.pose.position.x,
            source_y=command_pose.pose.position.y,
            target_x=target_x,
            target_y=target_y,
            fallback_yaw=self._pose_yaw(robot_pose),
        )
        return command_pose, target_distance

    def _action_retry_cooldown_elapsed(self) -> bool:
        if self.action_retry_cooldown_s <= 0.0:
            return True
        if self._last_goal_result_status not in (
            GoalStatus.STATUS_ABORTED,
            GoalStatus.STATUS_CANCELED,
            GoalStatus.STATUS_UNKNOWN,
        ):
            return True

        elapsed = (self.get_clock().now() - self._last_goal_result_time).nanoseconds / 1e9
        return elapsed >= self.action_retry_cooldown_s

    def _pose_yaw(self, pose: PoseStamped) -> float:
        q = pose.pose.orientation
        siny_cosp = 2.0 * (float(q.w) * float(q.z) + float(q.x) * float(q.y))
        cosy_cosp = 1.0 - 2.0 * (float(q.y) * float(q.y) + float(q.z) * float(q.z))
        return math.atan2(siny_cosp, cosy_cosp)

    def _should_publish_topic_goal(
        self,
        target_pose: PoseStamped,
        robot_pose: PoseStamped | None = None,
        target_distance: float | None = None,
    ) -> bool:
        if self._last_commanded_pose is None:
            return True

        dx = float(target_pose.pose.position.x) - float(self._last_commanded_pose.pose.position.x)
        dy = float(target_pose.pose.position.y) - float(self._last_commanded_pose.pose.position.y)
        translation_delta = math.hypot(dx, dy)
        yaw_delta = abs(self._pose_yaw(target_pose) - self._pose_yaw(self._last_commanded_pose))
        yaw_delta = math.atan2(math.sin(yaw_delta), math.cos(yaw_delta))
        yaw_delta = abs(yaw_delta)

        if translation_delta >= self.min_goal_translation_delta_m:
            return True
        if yaw_delta >= self.min_goal_yaw_delta_rad:
            return True

        if robot_pose is not None and target_distance is not None:
            correction_delta = max(0.03, 0.5 * self.min_goal_translation_delta_m)
            robot_to_goal = math.hypot(
                float(target_pose.pose.position.x) - float(robot_pose.pose.position.x),
                float(target_pose.pose.position.y) - float(robot_pose.pose.position.y),
            )
            needs_distance_correction = (
                target_distance < self.follow_standoff_distance_m
                or target_distance > (self.follow_standoff_distance_m + self.follow_distance_tolerance_m)
            )
            if (
                needs_distance_correction
                and translation_delta >= correction_delta
                and robot_to_goal >= correction_delta
            ):
                return True

        if self.goal_republish_period_s <= 0.0:
            return False
        elapsed = (self.get_clock().now() - self._last_commanded_time).nanoseconds / 1e9
        return elapsed >= self.goal_republish_period_s

    def _publish_topic_goal_if_needed(
        self,
        target_pose: PoseStamped,
        robot_pose: PoseStamped | None = None,
        target_distance: float | None = None,
    ) -> None:
        if not self._should_publish_topic_goal(target_pose, robot_pose=robot_pose, target_distance=target_distance):
            return

        command_pose = PoseStamped()
        command_pose.header = target_pose.header
        command_pose.pose = target_pose.pose
        self._goal_pose_pub.publish(command_pose)
        self._last_commanded_pose = command_pose
        self._last_commanded_time = self.get_clock().now()
        self.get_logger().info(
            f"Published follow goal on {self.goal_pose_topic}. "
            f"goal=({command_pose.pose.position.x:.2f}, {command_pose.pose.position.y:.2f}) "
            f"target_distance={target_distance:.2f}m"
        )

    def _publish_action_goal_update_if_needed(
        self,
        target_pose: PoseStamped,
        robot_pose: PoseStamped | None = None,
        target_distance: float | None = None,
    ) -> None:
        if not self._should_publish_topic_goal(target_pose, robot_pose=robot_pose, target_distance=target_distance):
            return

        updated_goal = self._copy_pose(target_pose)
        updated_goal.header.stamp = self.get_clock().now().to_msg()
        self._goal_update_pub.publish(updated_goal)
        self._last_commanded_pose = updated_goal
        self._last_commanded_time = self.get_clock().now()
        self.get_logger().info(
            f"Published follow goal update on {self.goal_update_topic}. "
            f"goal=({updated_goal.pose.position.x:.2f}, {updated_goal.pose.position.y:.2f}) "
            f"target_distance={target_distance:.2f}m"
        )

    def _publish_action_goal_update_now(self, target_pose: PoseStamped, reason: str) -> None:
        updated_goal = self._copy_pose(target_pose)
        updated_goal.header.stamp = self.get_clock().now().to_msg()
        self._goal_update_pub.publish(updated_goal)
        self._last_commanded_pose = updated_goal
        self._last_commanded_time = self.get_clock().now()
        self.get_logger().info(
            f"Published follow goal update on {self.goal_update_topic} ({reason}). "
            f"goal=({updated_goal.pose.position.x:.2f}, {updated_goal.pose.position.y:.2f})"
        )

    def _publish_hold_goal(self) -> None:
        robot_pose = self._lookup_robot_pose()
        if robot_pose is None:
            self.get_logger().warn("Could not resolve robot pose to publish a hold goal on disable.")
            return

        self._goal_pose_pub.publish(robot_pose)
        self._last_commanded_pose = robot_pose
        self._last_commanded_time = self.get_clock().now()
        self.get_logger().info(
            f"Published hold goal on {self.goal_pose_topic} at current robot pose to stop follow mode cleanly."
        )

    def _send_initial_goal(self, target_pose: PoseStamped) -> None:
        if not self._navigate_client.wait_for_server(timeout_sec=0.0):
            if not self._warned_server_not_ready:
                self.get_logger().warn("NavigateToPose action server is not ready yet.")
                self._warned_server_not_ready = True
            return
        self._warned_server_not_ready = False

        goal_pose = self._copy_pose(target_pose)
        goal_pose.header.stamp = self.get_clock().now().to_msg()

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = goal_pose
        goal_msg.behavior_tree = self.behavior_tree_path
        self._last_commanded_pose = goal_pose
        self._last_commanded_time = self.get_clock().now()

        self.get_logger().info(
            "Starting people-follow Nav2 session with follower BT. "
            f"target=({goal_pose.pose.position.x:.2f}, {goal_pose.pose.position.y:.2f})"
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

        if self._last_commanded_pose is not None:
            # Refresh GoalUpdater's cached topic state for this BT session so it doesn't
            # reuse an old /goal_update from a previous follow attempt.
            self._publish_action_goal_update_now(self._last_commanded_pose, reason="initial sync")

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

        self._last_goal_result_status = status
        self._last_goal_result_time = self.get_clock().now()
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

    def _stop_following(self) -> None:
        self._last_target_pose = None
        self._smoothed_target_pose = None
        if self.goal_command_mode == "goal_pose":
            if self.publish_hold_goal_on_disable:
                self._publish_hold_goal()
            return

        self._cancel_follow_goal()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = PeopleFollowNav2Bridge()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RuntimeError as exc:
        if "Unable to convert call argument to Python object" not in str(exc):
            raise
        node.get_logger().warn(f"Ignoring shutdown-time bridge runtime error: {exc}")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

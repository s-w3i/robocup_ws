#!/usr/bin/env python3
"""YASMIN ROS 2 carry-my-luggage state machine with retry logic."""

from __future__ import annotations

import math
import os
import threading
import time
from typing import Any, Callable

import rclpy
from action_msgs.msg import GoalStatus
from action_msgs.srv import CancelGoal
from coqui_tts_interfaces.action import SpeakText
from coqui_tts_interfaces.srv import RobotStatus
from geometry_msgs.msg import PoseStamped, Quaternion
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from rclpy.task import Future
from rclpy.time import Time
from robot_arm_action.action import Pick
from robot_arm_action.srv import ArmPose, GripperCommand
from std_msgs.msg import Bool
from std_srvs.srv import SetBool, Trigger
from tf2_ros import Buffer, TransformException, TransformListener

import yasmin
from yasmin import Blackboard, State, StateMachine
from yasmin_ros import set_ros_loggers
from yasmin_ros.basic_outcomes import ABORT, TIMEOUT
from yasmin_ros.yasmin_node import YasminNode
from yasmin_viewer import YasminViewerPub

from yoloe_detection_interfaces.srv import DetectObjectPrompt


FINAL_OUTCOME = "task_finished"
NAVIGATION_DONE = "navigation_done"
AWAKE_DETECTED = "awake_detected"
BAG_DETECTED = "bag_detected"
SPOKEN = "spoken"
DELAY_DONE = "delay_done"
SELECTED_BAG_REACHED = "selected_bag_reached"
ARM_POSE_READY = "arm_pose_ready"
GRIPPER_BAG_DETECTED = "gripper_bag_detected"
BAG_PICKED = "bag_picked"
FOLLOWING_ENABLED = "following_enabled"
STOP_COMMAND_DETECTED = "stop_command_detected"
GRIPPER_OPENED = "gripper_opened"
GRIPPER_CLOSED = "gripper_closed"

DEFAULT_NAVIGATION_ACTION = "/navigate_to_pose"
DEFAULT_ROBOT_STATUS_SERVICE = "/robot_status"
DEFAULT_AWAKE_TOPIC = "/awake"
DEFAULT_FOLLOW_ENABLE_SERVICE = "/set_laser_follow_enabled"
DEFAULT_GET_COMMAND_SERVICE = "/get_command"
DEFAULT_POINTED_DETECTION_SERVICE = "/yoloe/detect_pointed_prompt"
DEFAULT_GRIPPER_DETECTION_SERVICE = "/yoloe/detect_prompt"
DEFAULT_SPEAK_ACTION = "/coqui_tts/speak"
DEFAULT_CAMERA_NAME = "camera0"
DEFAULT_GRIPPER_CAMERA_NAME = "gripper_camera"
DEFAULT_ARM_POSE_SERVICE = "/arm_pose"
DEFAULT_PICK_ACTION = "/pick_object"
DEFAULT_OPEN_GRIPPER_SERVICE = "/open_gripper"


# Edit this global navigation target as needed for the carry-my-luggage task.
#field1
initial_pose: dict[str, Any] = {
    "frame_id": "map",
    "x": -1.6,
    "y": -5.2,
    "z": 0.0,
    "orientation": {
        "x": 0.0,
        "y": 0.0,
        "z": 0.700932,
        "w": 0.713228,
    },
}
#field2
# initial_pose: dict[str, Any] = {
#     "frame_id": "map",
#     "x": -1.6,
#     "y": -2.1,
#     "z": 0.0,
#     "orientation": {
#         "x": 0.0,
#         "y": 0.0,
#         "z": 0.700932,
#         "w": 0.713228,
#     },
# }

#[INFO] [1777025681.168407360] [rviz2]: Setting estimate pose: Frame:map, Position(-1.79898, -2.12657, 0), Orientation(0, 0, 0.703546, 0.71065) = Angle: 1.56075

#Setting estimate pose: Frame:map, Position(-1.5817, -5.29398, 0), Orientation(0, 0, 0.719127, 0.694878) = Angle: 1.60509



def wait_for_future(_node, future: Future, timeout_sec: float) -> Any:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if future.done():
            return future.result()
        time.sleep(0.05)
    raise TimeoutError("Timed out waiting for ROS response.")

def quaternion_from_yaw(yaw: float) -> Quaternion:
    half_yaw = 0.5 * float(yaw)
    return Quaternion(x=0.0, y=0.0, z=math.sin(half_yaw), w=math.cos(half_yaw))


def create_pose_stamped(spec: Any) -> PoseStamped:
    if isinstance(spec, PoseStamped):
        return spec
    if not isinstance(spec, dict):
        raise ValueError("initial_pose must be a dict or PoseStamped.")

    pose = PoseStamped()
    pose.header.frame_id = str(spec.get("frame_id", "map")).strip() or "map"
    pose.pose.position.x = float(spec.get("x", 0.0))
    pose.pose.position.y = float(spec.get("y", 0.0))
    pose.pose.position.z = float(spec.get("z", 0.0))

    if isinstance(spec.get("orientation"), dict):
        orientation = spec["orientation"]
        pose.pose.orientation.x = float(orientation.get("x", 0.0))
        pose.pose.orientation.y = float(orientation.get("y", 0.0))
        pose.pose.orientation.z = float(orientation.get("z", 0.0))
        pose.pose.orientation.w = float(orientation.get("w", 1.0))
        return pose

    if all(key in spec for key in ("qx", "qy", "qz", "qw")):
        pose.pose.orientation.x = float(spec.get("qx", 0.0))
        pose.pose.orientation.y = float(spec.get("qy", 0.0))
        pose.pose.orientation.z = float(spec.get("qz", 0.0))
        pose.pose.orientation.w = float(spec.get("qw", 1.0))
        return pose

    pose.pose.orientation = quaternion_from_yaw(float(spec.get("yaw", 0.0)))
    return pose


def get_yolo_detection_camera_name() -> str:
    camera_name = os.environ.get("YOLO_DETECTION_CAMERA_NAME", DEFAULT_CAMERA_NAME).strip()
    return camera_name or DEFAULT_CAMERA_NAME


def normalize_command_text(text: Any) -> str:
    return " ".join(str(text).strip().lower().split())


def parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if not text:
        return default
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def get_sim_mode() -> bool:
    if "CARRY_MY_LUGGAGE_SIM" in os.environ:
        return parse_bool(os.environ.get("CARRY_MY_LUGGAGE_SIM"), default=False)
    if "SIM" in os.environ:
        return parse_bool(os.environ.get("SIM"), default=False)
    return False


def stamp_to_seconds(stamp: Any) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def pose_distance(a: PoseStamped, b: PoseStamped) -> float:
    dx = float(a.pose.position.x) - float(b.pose.position.x)
    dy = float(a.pose.position.y) - float(b.pose.position.y)
    dz = float(a.pose.position.z) - float(b.pose.position.z)
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def copy_pose_stamped(source: PoseStamped) -> PoseStamped:
    copied = PoseStamped()
    copied.header = source.header
    copied.pose = source.pose
    return copied


def pose_yaw(pose: PoseStamped) -> float:
    q = pose.pose.orientation
    siny_cosp = 2.0 * (float(q.w) * float(q.z) + float(q.x) * float(q.y))
    cosy_cosp = 1.0 - 2.0 * (float(q.y) * float(q.y) + float(q.z) * float(q.z))
    return math.atan2(siny_cosp, cosy_cosp)


def build_bag_approach_pose(
    bag_pose: PoseStamped,
    robot_pose: PoseStamped,
    standoff_distance_m: float,
) -> tuple[PoseStamped, float]:
    target_x = float(bag_pose.pose.position.x)
    target_y = float(bag_pose.pose.position.y)
    robot_x = float(robot_pose.pose.position.x)
    robot_y = float(robot_pose.pose.position.y)
    target_distance = math.hypot(target_x - robot_x, target_y - robot_y)

    approach_pose = PoseStamped()
    approach_pose.header.frame_id = str(bag_pose.header.frame_id).strip() or "map"
    approach_pose.header.stamp = bag_pose.header.stamp
    approach_pose.pose.position.z = 0.0

    if target_distance <= 1e-3:
        approach_pose.pose.position.x = robot_x
        approach_pose.pose.position.y = robot_y
        approach_pose.pose.orientation = quaternion_from_yaw(pose_yaw(robot_pose))
        return approach_pose, target_distance

    yaw_to_bag = math.atan2(target_y - robot_y, target_x - robot_x)
    if target_distance <= standoff_distance_m:
        approach_pose.pose.position.x = robot_x
        approach_pose.pose.position.y = robot_y
        approach_pose.pose.orientation = quaternion_from_yaw(yaw_to_bag)
        return approach_pose, target_distance

    offset_x = (robot_x - target_x) / target_distance
    offset_y = (robot_y - target_y) / target_distance
    approach_pose.pose.position.x = target_x + offset_x * standoff_distance_m
    approach_pose.pose.position.y = target_y + offset_y * standoff_distance_m
    approach_pose.pose.orientation = quaternion_from_yaw(
        math.atan2(
            target_y - float(approach_pose.pose.position.y),
            target_x - float(approach_pose.pose.position.x),
        )
    )
    return approach_pose, target_distance


def _retry_sleep(delay_sec: float) -> None:
    if delay_sec > 0.0:
        time.sleep(delay_sec)


def retry_wait_for_service(
    client,
    service_name: str,
    *,
    attempts: int,
    wait_timeout_sec: float,
    retry_delay_sec: float,
) -> bool:
    for attempt in range(1, max(1, attempts) + 1):
        if client.wait_for_service(timeout_sec=wait_timeout_sec):
            return True
        if attempt < attempts:
            _retry_sleep(retry_delay_sec)
    return False


def retry_service_call(
    node,
    client,
    request,
    *,
    service_name: str,
    wait_attempts: int,
    wait_timeout_sec: float,
    call_attempts: int,
    call_timeout_sec: float,
    retry_delay_sec: float,
    response_ok: Callable[[Any], bool] | None = None,
    response_error: Callable[[Any], str] | None = None,
) -> Any:
    if not retry_wait_for_service(
        client,
        service_name,
        attempts=wait_attempts,
        wait_timeout_sec=wait_timeout_sec,
        retry_delay_sec=retry_delay_sec,
    ):
        raise TimeoutError(
            f"Service '{service_name}' is not available after {wait_attempts} attempts."
        )

    last_error = ""
    for attempt in range(1, max(1, call_attempts) + 1):
        try:
            response = wait_for_future(node, client.call_async(request), call_timeout_sec)
            if response is None:
                last_error = f"Service '{service_name}' returned no response."
            elif response_ok is None or response_ok(response):
                return response
            else:
                if response_error is not None:
                    last_error = response_error(response)
                else:
                    last_error = f"Service '{service_name}' returned an unsuccessful response."
        except TimeoutError as exc:
            last_error = str(exc)
        except Exception as exc:
            last_error = str(exc)

        if attempt < call_attempts:
            _retry_sleep(retry_delay_sec)

    raise RuntimeError(last_error or f"Service '{service_name}' failed after retries.")


def retry_action_goal(
    node,
    client: ActionClient,
    goal,
    *,
    action_name: str,
    server_wait_attempts: int,
    server_wait_timeout_sec: float,
    send_goal_attempts: int,
    send_goal_timeout_sec: float,
    result_timeout_sec: float,
    retry_delay_sec: float,
    result_ok: Callable[[Any, Any], tuple[bool, str]] | None = None,
) -> Any:
    available = False
    for attempt in range(1, max(1, server_wait_attempts) + 1):
        if client.wait_for_server(timeout_sec=server_wait_timeout_sec):
            available = True
            break
        if attempt < server_wait_attempts:
            _retry_sleep(retry_delay_sec)

    if not available:
        raise TimeoutError(
            f"Action server '{action_name}' is not available after {server_wait_attempts} attempts."
        )

    last_error = ""
    for attempt in range(1, max(1, send_goal_attempts) + 1):
        try:
            goal_handle = wait_for_future(node, client.send_goal_async(goal), send_goal_timeout_sec)
            if goal_handle is None:
                last_error = f"Action '{action_name}' returned no goal handle."
            elif not goal_handle.accepted:
                last_error = f"Action '{action_name}' goal was rejected."
            else:
                result_wrapper = wait_for_future(node, goal_handle.get_result_async(), result_timeout_sec)
                result = None if result_wrapper is None else result_wrapper.result
                status = None if result_wrapper is None else int(result_wrapper.status)

                if result_ok is None:
                    return result_wrapper

                ok, error_message = result_ok(result_wrapper, result)
                if ok:
                    return result_wrapper
                last_error = error_message or f"Action '{action_name}' finished unsuccessfully."
        except TimeoutError as exc:
            last_error = str(exc)
        except Exception as exc:
            last_error = str(exc)

        if attempt < send_goal_attempts:
            _retry_sleep(retry_delay_sec)

    raise RuntimeError(last_error or f"Action '{action_name}' failed after retries.")


def call_speak_text(
    node,
    action_name: str,
    text: str,
    *,
    server_wait_attempts: int,
    server_wait_timeout_sec: float,
    send_goal_attempts: int,
    send_goal_timeout_sec: float,
    result_timeout_sec: float,
    retry_delay_sec: float,
) -> None:
    text = str(text).strip()
    if not text:
        return

    client = ActionClient(node, SpeakText, action_name)

    goal = SpeakText.Goal()
    goal.text = text

    def _result_ok(_result_wrapper, result) -> tuple[bool, str]:
        if result is not None and bool(result.success):
            return True, ""
        message = "" if result is None else str(result.message).strip()
        return False, message or "SpeakText action failed."

    retry_action_goal(
        node,
        client,
        goal,
        action_name=action_name,
        server_wait_attempts=server_wait_attempts,
        server_wait_timeout_sec=server_wait_timeout_sec,
        send_goal_attempts=send_goal_attempts,
        send_goal_timeout_sec=send_goal_timeout_sec,
        result_timeout_sec=result_timeout_sec,
        retry_delay_sec=retry_delay_sec,
        result_ok=_result_ok,
    )


def shutdown_yasmin_node() -> None:
    node = YasminNode._instance
    if node is None:
        return

    executor = getattr(node, "_executor", None)
    spin_thread = getattr(node, "_spin_thread", None)

    try:
        if executor is not None:
            try:
                executor.remove_node(node)
            except Exception:
                pass
            try:
                executor.shutdown()
            except Exception:
                pass

        if spin_thread is not None and spin_thread.is_alive():
            spin_thread.join(timeout=2.0)

        try:
            node.destroy_node()
        except Exception:
            pass
    finally:
        YasminNode._instance = None


class SpeakState(State):
    def __init__(self, text_factory) -> None:
        super().__init__({SPOKEN, ABORT})
        self._text_factory = text_factory
        self._node = YasminNode.get_instance()
        self._action_name = DEFAULT_SPEAK_ACTION

    def execute(self, blackboard: Blackboard) -> str:
        try:
            text = str(self._text_factory(blackboard)).strip()
            blackboard["last_spoken_text"] = text
            call_speak_text(
                self._node,
                self._action_name,
                text,
                server_wait_attempts=int(blackboard["speak_server_wait_attempts"]),
                server_wait_timeout_sec=float(blackboard["speak_server_wait_timeout_sec"]),
                send_goal_attempts=int(blackboard["speak_send_goal_attempts"]),
                send_goal_timeout_sec=float(blackboard["speak_send_goal_timeout_sec"]),
                result_timeout_sec=float(blackboard["speak_result_timeout_sec"]),
                retry_delay_sec=float(blackboard["retry_delay_sec"]),
            )
            blackboard["last_error"] = ""
            return SPOKEN
        except Exception as exc:
            blackboard["last_error"] = str(exc)
            return ABORT


class DelayState(State):
    def __init__(self, duration_factory) -> None:
        super().__init__({DELAY_DONE})
        self._duration_factory = duration_factory

    def execute(self, blackboard: Blackboard) -> str:
        duration_sec = max(0.0, float(self._duration_factory(blackboard)))
        if duration_sec > 0.0:
            time.sleep(duration_sec)
        blackboard["last_error"] = ""
        return DELAY_DONE


class NavigateToInitialPoseState(State):
    def __init__(self) -> None:
        super().__init__({NAVIGATION_DONE, ABORT, TIMEOUT})
        self._node = YasminNode.get_instance()
        self._action_name = DEFAULT_NAVIGATION_ACTION
        self._client = ActionClient(self._node, NavigateToPose, self._action_name)

    def execute(self, blackboard: Blackboard) -> str:
        try:
            if parse_bool(blackboard["sim"], default=False):
                blackboard["navigation_skipped"] = True
                blackboard["navigation_status"] = GoalStatus.STATUS_SUCCEEDED
                blackboard["navigation_error_code"] = 0
                blackboard["last_error"] = ""
                return NAVIGATION_DONE

            goal = NavigateToPose.Goal()
            goal.pose = create_pose_stamped(blackboard["initial_pose"])

            def _result_ok(result_wrapper, result) -> tuple[bool, str]:
                status = None if result_wrapper is None else int(result_wrapper.status)
                error_code = 0 if result is None else int(getattr(result, "error_code", 0))

                blackboard["navigation_status"] = -1 if status is None else status
                blackboard["navigation_error_code"] = error_code

                if status == GoalStatus.STATUS_SUCCEEDED and error_code == 0:
                    return True, ""

                return (
                    False,
                    f"Navigation failed with status={status} error_code={error_code}.",
                )

            retry_action_goal(
                self._node,
                self._client,
                goal,
                action_name=self._action_name,
                server_wait_attempts=int(blackboard["navigation_server_wait_attempts"]),
                server_wait_timeout_sec=float(blackboard["navigation_server_wait_timeout_sec"]),
                send_goal_attempts=int(blackboard["navigation_goal_attempts"]),
                send_goal_timeout_sec=float(blackboard["navigation_send_goal_timeout_sec"]),
                result_timeout_sec=float(blackboard["navigation_timeout_sec"]),
                retry_delay_sec=float(blackboard["retry_delay_sec"]),
                result_ok=_result_ok,
            )

            blackboard["last_error"] = ""
            blackboard["navigation_skipped"] = False
            return NAVIGATION_DONE
        except TimeoutError as exc:
            blackboard["last_error"] = str(exc)
            return TIMEOUT
        except Exception as exc:
            blackboard["last_error"] = str(exc)
            return ABORT


class NavigateToSelectedBagState(State):
    def __init__(self) -> None:
        super().__init__({SELECTED_BAG_REACHED, ABORT, TIMEOUT})
        self._node = YasminNode.get_instance()
        self._action_name = DEFAULT_NAVIGATION_ACTION
        self._client = ActionClient(self._node, NavigateToPose, self._action_name)
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self._node)

    def _lookup_robot_pose_in_frame(self, frame_id: str, blackboard: Blackboard) -> PoseStamped | None:
        timeout = Duration(seconds=float(blackboard["transform_timeout_sec"]))
        candidate_frames = [
            str(blackboard["robot_base_frame"]).strip(),
            str(blackboard["robot_base_frame_fallback"]).strip(),
        ]

        for robot_frame in candidate_frames:
            if not robot_frame:
                continue
            try:
                robot_tf = self._tf_buffer.lookup_transform(
                    frame_id,
                    robot_frame,
                    Time(),
                    timeout=timeout,
                )
            except TransformException:
                continue

            pose = PoseStamped()
            pose.header.stamp = robot_tf.header.stamp
            pose.header.frame_id = str(frame_id).strip() or "map"
            pose.pose.position.x = float(robot_tf.transform.translation.x)
            pose.pose.position.y = float(robot_tf.transform.translation.y)
            pose.pose.position.z = 0.0
            pose.pose.orientation = robot_tf.transform.rotation
            return pose

        return None

    def _fallback_robot_pose(self, bag_pose: PoseStamped, blackboard: Blackboard) -> PoseStamped | None:
        fallback_pose = create_pose_stamped(blackboard["initial_pose"])
        if str(fallback_pose.header.frame_id).strip() != str(bag_pose.header.frame_id).strip():
            return None
        return fallback_pose

    def execute(self, blackboard: Blackboard) -> str:
        try:
            if parse_bool(blackboard["sim"], default=False):
                blackboard["bag_navigation_skipped"] = True
                blackboard["bag_navigation_status"] = GoalStatus.STATUS_SUCCEEDED
                blackboard["bag_navigation_error_code"] = 0
                blackboard["bag_navigation_used_fallback_pose"] = False
                blackboard["last_error"] = ""
                return SELECTED_BAG_REACHED

            bag_pose = blackboard["selected_bag_pose"]
            if not isinstance(bag_pose, PoseStamped):
                raise RuntimeError("No selected bag pose is available for navigation.")

            target_frame = str(bag_pose.header.frame_id).strip() or "map"
            robot_pose = self._lookup_robot_pose_in_frame(target_frame, blackboard)
            used_fallback_pose = False
            if robot_pose is None:
                robot_pose = self._fallback_robot_pose(bag_pose, blackboard)
                used_fallback_pose = robot_pose is not None

            if robot_pose is None:
                raise RuntimeError(
                    "Failed to resolve the robot pose for bag approach navigation. "
                    "TF lookup and initial_pose fallback were unavailable."
                )

            approach_pose, bag_distance = build_bag_approach_pose(
                bag_pose=bag_pose,
                robot_pose=robot_pose,
                standoff_distance_m=float(blackboard["bag_navigation_standoff_distance_m"]),
            )
            blackboard["approach_bag_pose"] = copy_pose_stamped(approach_pose)
            blackboard["bag_navigation_target_distance_m"] = float(bag_distance)
            blackboard["bag_navigation_used_fallback_pose"] = bool(used_fallback_pose)

            goal = NavigateToPose.Goal()
            goal.pose = approach_pose

            def _result_ok(result_wrapper, result) -> tuple[bool, str]:
                status = None if result_wrapper is None else int(result_wrapper.status)
                error_code = 0 if result is None else int(getattr(result, "error_code", 0))

                blackboard["bag_navigation_status"] = -1 if status is None else status
                blackboard["bag_navigation_error_code"] = error_code

                if status == GoalStatus.STATUS_SUCCEEDED and error_code == 0:
                    return True, ""

                return (
                    False,
                    f"Bag navigation failed with status={status} error_code={error_code}.",
                )

            retry_action_goal(
                self._node,
                self._client,
                goal,
                action_name=self._action_name,
                server_wait_attempts=int(blackboard["navigation_server_wait_attempts"]),
                server_wait_timeout_sec=float(blackboard["navigation_server_wait_timeout_sec"]),
                send_goal_attempts=int(blackboard["navigation_goal_attempts"]),
                send_goal_timeout_sec=float(blackboard["navigation_send_goal_timeout_sec"]),
                result_timeout_sec=float(blackboard["navigation_timeout_sec"]),
                retry_delay_sec=float(blackboard["retry_delay_sec"]),
                result_ok=_result_ok,
            )

            blackboard["bag_navigation_skipped"] = False
            blackboard["last_error"] = ""
            return SELECTED_BAG_REACHED
        except TimeoutError as exc:
            blackboard["last_error"] = str(exc)
            return TIMEOUT
        except Exception as exc:
            blackboard["last_error"] = str(exc)
            return ABORT


class WaitForAwakeState(State):
    def __init__(self) -> None:
        super().__init__({AWAKE_DETECTED, ABORT, TIMEOUT})
        self._node = YasminNode.get_instance()
        self._robot_status_service_name = DEFAULT_ROBOT_STATUS_SERVICE
        self._robot_status_client = self._node.create_client(
            RobotStatus,
            self._robot_status_service_name,
        )
        self._lock = threading.Lock()
        self._awake_seen_true = False
        self._latest_awake = False

        qos = QoSProfile(depth=1)
        qos.reliability = QoSReliabilityPolicy.RELIABLE
        qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        self._node.create_subscription(
            Bool,
            DEFAULT_AWAKE_TOPIC,
            self._awake_callback,
            qos,
        )

    def _awake_callback(self, msg: Bool) -> None:
        with self._lock:
            self._latest_awake = bool(msg.data)
            if self._latest_awake:
                self._awake_seen_true = True

    def _reset_awake_state(self) -> None:
        with self._lock:
            self._awake_seen_true = False
            self._latest_awake = False

    def _has_awake_event(self) -> bool:
        with self._lock:
            return self._awake_seen_true

    def execute(self, blackboard: Blackboard) -> str:
        self._reset_awake_state()

        request = RobotStatus.Request()
        request.status = "sleep"

        try:
            response = retry_service_call(
                self._node,
                self._robot_status_client,
                request,
                service_name=self._robot_status_service_name,
                wait_attempts=int(blackboard["service_wait_attempts"]),
                wait_timeout_sec=float(blackboard["service_wait_timeout_sec"]),
                call_attempts=int(blackboard["robot_status_call_attempts"]),
                call_timeout_sec=float(blackboard["robot_status_call_timeout_sec"]),
                retry_delay_sec=float(blackboard["retry_delay_sec"]),
                response_ok=lambda r: r is not None and bool(r.success),
                response_error=lambda r: (
                    "" if r is None else str(r.message).strip()
                ) or "Robot status service failed to enter sleep mode.",
            )
        except TimeoutError as exc:
            blackboard["last_error"] = str(exc)
            return TIMEOUT
        except Exception as exc:
            blackboard["last_error"] = str(exc)
            return ABORT

        blackboard["robot_status_message"] = "" if response is None else str(response.message).strip()
        blackboard["last_error"] = ""

        timeout_sec = float(blackboard["awake_wait_timeout_sec"])
        deadline = None if timeout_sec <= 0.0 else time.monotonic() + timeout_sec
        poll_delay_sec = float(blackboard["awake_poll_delay_sec"])

        while deadline is None or time.monotonic() < deadline:
            if self._has_awake_event():
                blackboard["last_error"] = ""
                return AWAKE_DETECTED
            time.sleep(poll_delay_sec)

        blackboard["last_error"] = "Timed out waiting for /awake=true."
        return TIMEOUT


class DetectPointedBagState(State):
    def __init__(self) -> None:
        super().__init__({BAG_DETECTED, ABORT, TIMEOUT})
        self._node = YasminNode.get_instance()
        self._service_name = DEFAULT_POINTED_DETECTION_SERVICE
        self._client = self._node.create_client(DetectObjectPrompt, self._service_name)

    def execute(self, blackboard: Blackboard) -> str:
        request = DetectObjectPrompt.Request()
        request.prompt_text = str(blackboard["pointed_bag_prompt"]).strip() or "brown paper bag"
        request.save_image = True
        request.camera_name = (
            str(blackboard["yolo_camera_name"]).strip() or get_yolo_detection_camera_name()
        )

        try:
            response = retry_service_call(
                self._node,
                self._client,
                request,
                service_name=self._service_name,
                wait_attempts=int(blackboard["service_wait_attempts"]),
                wait_timeout_sec=float(blackboard["service_wait_timeout_sec"]),
                call_attempts=int(blackboard["pointed_detection_call_attempts"]),
                call_timeout_sec=float(blackboard["pointed_detection_timeout_sec"]),
                retry_delay_sec=float(blackboard["retry_delay_sec"]),
                response_ok=lambda r: r is not None and bool(r.success),
                response_error=lambda r: (
                    "" if r is None else str(r.message).strip()
                ) or "Pointed bag detection failed.",
            )
        except TimeoutError as exc:
            blackboard["last_error"] = str(exc)
            return TIMEOUT
        except Exception as exc:
            blackboard["last_error"] = str(exc)
            return ABORT

        if not response.poses_camera_link:
            blackboard["last_error"] = "Pointed bag detection did not return a pose."
            return ABORT

        blackboard["detected_side"] = str(response.selected_side).strip() or "that side"
        blackboard["pointed_detection_message"] = str(response.message).strip()
        blackboard["pointed_saved_image_path"] = str(response.saved_image_path).strip()
        blackboard["pointed_detections_in_frame"] = int(response.detections_in_frame)
        blackboard["selected_bag_pose"] = copy_pose_stamped(response.poses_camera_link[0])
        blackboard["selected_bag_tf"] = (
            str(response.tf_child_frames[0]).strip() if response.tf_child_frames else ""
        )
        blackboard["last_error"] = ""
        return BAG_DETECTED


class MoveArmToDetectPoseState(State):
    def __init__(self) -> None:
        super().__init__({ARM_POSE_READY, ABORT, TIMEOUT})
        self._node = YasminNode.get_instance()
        self._service_name = DEFAULT_ARM_POSE_SERVICE
        self._client = self._node.create_client(ArmPose, self._service_name)
        self._pose_name_key = "arm_detect_pose_name"

    def execute(self, blackboard: Blackboard) -> str:
        request = ArmPose.Request()
        request.pose_name = str(blackboard[self._pose_name_key]).strip() or "detect"

        try:
            response = retry_service_call(
                self._node,
                self._client,
                request,
                service_name=self._service_name,
                wait_attempts=int(blackboard["service_wait_attempts"]),
                wait_timeout_sec=float(blackboard["service_wait_timeout_sec"]),
                call_attempts=int(blackboard["arm_pose_call_attempts"]),
                call_timeout_sec=float(blackboard["arm_pose_call_timeout_sec"]),
                retry_delay_sec=float(blackboard["retry_delay_sec"]),
                response_ok=lambda r: r is not None and bool(r.success),
                response_error=lambda r: (
                    "" if r is None else str(r.message).strip()
                ) or "Failed to move the arm to the detect pose.",
            )
        except TimeoutError as exc:
            blackboard["last_error"] = str(exc)
            return TIMEOUT
        except Exception as exc:
            blackboard["last_error"] = str(exc)
            return ABORT

        blackboard["arm_pose_message"] = "" if response is None else str(response.message).strip()
        blackboard["last_error"] = ""
        return ARM_POSE_READY


class MoveArmToZeroPoseState(State):
    def __init__(self) -> None:
        super().__init__({ARM_POSE_READY, ABORT, TIMEOUT})
        self._node = YasminNode.get_instance()
        self._service_name = DEFAULT_ARM_POSE_SERVICE
        self._client = self._node.create_client(ArmPose, self._service_name)

    def execute(self, blackboard: Blackboard) -> str:
        request = ArmPose.Request()
        request.pose_name = str(blackboard["arm_zero_pose_name"]).strip() or "zero"

        try:
            response = retry_service_call(
                self._node,
                self._client,
                request,
                service_name=self._service_name,
                wait_attempts=int(blackboard["service_wait_attempts"]),
                wait_timeout_sec=float(blackboard["service_wait_timeout_sec"]),
                call_attempts=int(blackboard["arm_pose_call_attempts"]),
                call_timeout_sec=float(blackboard["arm_pose_call_timeout_sec"]),
                retry_delay_sec=float(blackboard["retry_delay_sec"]),
                response_ok=lambda r: r is not None and bool(r.success),
                response_error=lambda r: (
                    "" if r is None else str(r.message).strip()
                ) or "Failed to move the arm to the zero pose.",
            )
        except TimeoutError as exc:
            blackboard["last_error"] = str(exc)
            return TIMEOUT
        except Exception as exc:
            blackboard["last_error"] = str(exc)
            return ABORT

        blackboard["arm_pose_message"] = "" if response is None else str(response.message).strip()
        blackboard["last_error"] = ""
        return ARM_POSE_READY


class SetFollowModeState(State):
    def __init__(self, enable: bool) -> None:
        outcomes = {FOLLOWING_ENABLED, ABORT, TIMEOUT}
        super().__init__(outcomes)
        self._enable = bool(enable)
        self._node = YasminNode.get_instance()
        self._service_name = DEFAULT_FOLLOW_ENABLE_SERVICE
        self._client = self._node.create_client(SetBool, self._service_name)
        self._navigation_cancel_service_name = f"{DEFAULT_NAVIGATION_ACTION}/_action/cancel_goal"
        self._navigation_cancel_client = self._node.create_client(
            CancelGoal,
            self._navigation_cancel_service_name,
        )

    def _cancel_navigation_goals(self, blackboard: Blackboard) -> None:
        request = CancelGoal.Request()

        response = retry_service_call(
            self._node,
            self._navigation_cancel_client,
            request,
            service_name=self._navigation_cancel_service_name,
            wait_attempts=int(blackboard["service_wait_attempts"]),
            wait_timeout_sec=float(blackboard["service_wait_timeout_sec"]),
            call_attempts=int(blackboard["follow_disable_call_attempts"]),
            call_timeout_sec=float(blackboard["follow_disable_call_timeout_sec"]),
            retry_delay_sec=float(blackboard["retry_delay_sec"]),
            response_ok=lambda r: r is not None and int(r.return_code) in (
                CancelGoal.Response.ERROR_NONE,
                CancelGoal.Response.ERROR_UNKNOWN_GOAL_ID,
                CancelGoal.Response.ERROR_GOAL_TERMINATED,
            ),
            response_error=lambda r: (
                "" if r is None else f"Nav2 cancel_goal returned code {int(r.return_code)}."
            )
            or "Failed to cancel Nav2 goal while stopping people follow.",
        )

        if response is None:
            blackboard["navigation_cancel_message"] = ""
            return

        return_code = int(response.return_code)
        if return_code == CancelGoal.Response.ERROR_NONE:
            count = len(getattr(response, "goals_canceling", []))
            blackboard["navigation_cancel_message"] = (
                f"Requested cancellation for {count} active Nav2 goal(s)."
            )
        elif return_code == CancelGoal.Response.ERROR_UNKNOWN_GOAL_ID:
            blackboard["navigation_cancel_message"] = "No active Nav2 goal was running."
        else:
            blackboard["navigation_cancel_message"] = "Nav2 goal was already in a terminal state."

    def execute(self, blackboard: Blackboard) -> str:
        sim_mode = parse_bool(blackboard["sim"], default=False)
        if sim_mode and self._enable:
            blackboard["follow_enabled"] = False
            blackboard["follow_service_message"] = (
                "Sim mode enabled; skipped people-follow service call."
            )
            blackboard["last_error"] = ""
            return FOLLOWING_ENABLED

        if not sim_mode:
            request = SetBool.Request()
            request.data = self._enable

            call_attempts_key = (
                "follow_enable_call_attempts" if self._enable else "follow_disable_call_attempts"
            )
            call_timeout_key = (
                "follow_enable_call_timeout_sec" if self._enable else "follow_disable_call_timeout_sec"
            )

            try:
                response = retry_service_call(
                    self._node,
                    self._client,
                    request,
                    service_name=self._service_name,
                    wait_attempts=int(blackboard["service_wait_attempts"]),
                    wait_timeout_sec=float(blackboard["service_wait_timeout_sec"]),
                    call_attempts=int(blackboard[call_attempts_key]),
                    call_timeout_sec=float(blackboard[call_timeout_key]),
                    retry_delay_sec=float(blackboard["retry_delay_sec"]),
                    response_ok=lambda r: r is not None and bool(r.success),
                    response_error=lambda r: (
                        "" if r is None else str(getattr(r, "message", "")).strip()
                    )
                    or (
                        "Failed to enable laser following."
                        if self._enable
                        else "Failed to disable laser following."
                    ),
                )
            except TimeoutError as exc:
                blackboard["last_error"] = str(exc)
                return TIMEOUT
            except Exception as exc:
                blackboard["last_error"] = str(exc)
                return ABORT

            blackboard["follow_enabled"] = bool(self._enable)
            blackboard["follow_service_message"] = (
                "" if response is None else str(getattr(response, "message", "")).strip()
            )
        else:
            blackboard["follow_enabled"] = False
            blackboard["follow_service_message"] = (
                "Sim mode enabled; skipped people-follow service call."
            )

        if not self._enable:
            try:
                self._cancel_navigation_goals(blackboard)
            except TimeoutError as exc:
                blackboard["last_error"] = str(exc)
                return TIMEOUT
            except Exception as exc:
                blackboard["last_error"] = str(exc)
                return ABORT

        blackboard["last_error"] = ""
        return FOLLOWING_ENABLED


def ensure_follow_mode_disabled_on_startup(blackboard: Blackboard) -> None:
    node = YasminNode.get_instance()
    if parse_bool(blackboard["sim"], default=False):
        node.get_logger().info(
            "Sim mode enabled; skipping startup follow disable and assuming follow mode is off."
        )
        blackboard["follow_enabled"] = False
        blackboard["follow_service_message"] = (
            "Sim mode enabled; skipped startup follow disable."
        )
        return

    service_name = DEFAULT_FOLLOW_ENABLE_SERVICE
    client = node.create_client(SetBool, service_name)
    request = SetBool.Request()
    request.data = False

    response = retry_service_call(
        node,
        client,
        request,
        service_name=service_name,
        wait_attempts=int(blackboard["service_wait_attempts"]),
        wait_timeout_sec=float(blackboard["service_wait_timeout_sec"]),
        call_attempts=int(blackboard["follow_disable_call_attempts"]),
        call_timeout_sec=float(blackboard["follow_disable_call_timeout_sec"]),
        retry_delay_sec=float(blackboard["retry_delay_sec"]),
        response_ok=lambda r: r is not None and bool(r.success),
        response_error=lambda r: (
            "" if r is None else str(getattr(r, "message", "")).strip()
        ) or "Failed to disable follow mode during startup initialization.",
    )
    blackboard["follow_enabled"] = False
    blackboard["follow_service_message"] = (
        "" if response is None else str(getattr(response, "message", "")).strip()
    )
    node.get_logger().info(
        "Startup safety check confirmed follow mode is disabled. "
        f"message='{blackboard['follow_service_message']}'"
    )


class WaitForStopCommandState(State):
    def __init__(self) -> None:
        super().__init__({STOP_COMMAND_DETECTED, ABORT, TIMEOUT})
        self._node = YasminNode.get_instance()
        self._service_name = DEFAULT_GET_COMMAND_SERVICE
        self._client = self._node.create_client(Trigger, self._service_name)
        self._follow_enable_service_name = DEFAULT_FOLLOW_ENABLE_SERVICE
        self._follow_enable_client = self._node.create_client(
            SetBool,
            self._follow_enable_service_name,
        )

    def _disable_follow_best_effort(self, blackboard: Blackboard) -> None:
        if parse_bool(blackboard["sim"], default=False):
            blackboard["follow_enabled"] = False
            return
        request = SetBool.Request()
        request.data = False
        try:
            retry_service_call(
                self._node,
                self._follow_enable_client,
                request,
                service_name=self._follow_enable_service_name,
                wait_attempts=int(blackboard["best_effort_wait_attempts"]),
                wait_timeout_sec=float(blackboard["best_effort_wait_timeout_sec"]),
                call_attempts=int(blackboard["best_effort_call_attempts"]),
                call_timeout_sec=float(blackboard["best_effort_call_timeout_sec"]),
                retry_delay_sec=float(blackboard["retry_delay_sec"]),
                response_ok=lambda r: r is not None,
                response_error=lambda r: (
                    "" if r is None else str(getattr(r, "message", "")).strip()
                ) or "Failed to disable people following.",
            )
            blackboard["follow_enabled"] = False
        except Exception:
            pass

    def execute(self, blackboard: Blackboard) -> str:
        timeout_sec = float(blackboard["stop_command_wait_timeout_sec"])
        deadline = None if timeout_sec <= 0.0 else time.monotonic() + timeout_sec
        listen_retry_delay_sec = float(blackboard["stop_command_retry_delay_sec"])
        stop_keywords = [
            normalize_command_text(keyword)
            for keyword in blackboard["stop_command_keywords"]
            if normalize_command_text(keyword)
        ]

        while deadline is None or time.monotonic() < deadline:
            request = Trigger.Request()

            try:
                response = retry_service_call(
                    self._node,
                    self._client,
                    request,
                    service_name=self._service_name,
                    wait_attempts=int(blackboard["service_wait_attempts"]),
                    wait_timeout_sec=float(blackboard["service_wait_timeout_sec"]),
                    call_attempts=int(blackboard["get_command_call_attempts"]),
                    call_timeout_sec=float(blackboard["get_command_call_timeout_sec"]),
                    retry_delay_sec=float(blackboard["retry_delay_sec"]),
                    response_ok=lambda r: r is not None,
                    response_error=lambda r: (
                        "" if r is None else str(getattr(r, "message", "")).strip()
                    ) or "Get-command service returned no response.",
                )
            except TimeoutError as exc:
                self._disable_follow_best_effort(blackboard)
                blackboard["last_error"] = str(exc)
                return TIMEOUT
            except Exception as exc:
                self._disable_follow_best_effort(blackboard)
                blackboard["last_error"] = str(exc)
                return ABORT

            transcript = "" if response is None else str(getattr(response, "message", "")).strip()
            normalized_transcript = normalize_command_text(transcript)
            blackboard["last_heard_command"] = transcript

            if bool(getattr(response, "success", False)):
                if any(keyword in normalized_transcript for keyword in stop_keywords):
                    blackboard["stop_command_transcript"] = transcript
                    blackboard["last_error"] = ""
                    return STOP_COMMAND_DETECTED

            if deadline is not None and time.monotonic() >= deadline:
                break
            _retry_sleep(listen_retry_delay_sec)

        self._disable_follow_best_effort(blackboard)
        blackboard["last_error"] = "Timed out waiting for a spoken stop command."
        return TIMEOUT


class DetectBagWithGripperCameraState(State):
    def __init__(self) -> None:
        super().__init__({GRIPPER_BAG_DETECTED, ABORT, TIMEOUT})
        self._node = YasminNode.get_instance()
        self._service_name = DEFAULT_GRIPPER_DETECTION_SERVICE
        self._client = self._node.create_client(DetectObjectPrompt, self._service_name)

    def execute(self, blackboard: Blackboard) -> str:
        request = DetectObjectPrompt.Request()
        request.prompt_text = str(blackboard["gripper_bag_prompt"]).strip() or "brown paper bag"
        request.save_image = True
        request.camera_name = str(blackboard["gripper_camera_name"]).strip() or DEFAULT_GRIPPER_CAMERA_NAME

        try:
            response = retry_service_call(
                self._node,
                self._client,
                request,
                service_name=self._service_name,
                wait_attempts=int(blackboard["service_wait_attempts"]),
                wait_timeout_sec=float(blackboard["service_wait_timeout_sec"]),
                call_attempts=int(blackboard["gripper_detection_call_attempts"]),
                call_timeout_sec=float(blackboard["gripper_detection_timeout_sec"]),
                retry_delay_sec=float(blackboard["retry_delay_sec"]),
                response_ok=lambda r: r is not None and bool(r.success),
                response_error=lambda r: (
                    "" if r is None else str(r.message).strip()
                ) or "Gripper camera bag detection failed.",
            )
        except TimeoutError as exc:
            blackboard["last_error"] = str(exc)
            return TIMEOUT
        except Exception as exc:
            blackboard["last_error"] = str(exc)
            return ABORT

        blackboard["gripper_detection_message"] = str(response.message).strip()
        blackboard["gripper_saved_image_path"] = str(response.saved_image_path).strip()
        blackboard["gripper_detections_in_frame"] = int(response.detections_in_frame)
        blackboard["pick_target_tf"] = (
            str(response.tf_child_frames[0]).strip()
            if response.tf_child_frames
            else str(blackboard["pick_target_tf"]).strip() or "brown_paper_bag_1"
        )
        blackboard["last_error"] = ""
        return GRIPPER_BAG_DETECTED


class PickBagState(State):
    def __init__(self) -> None:
        super().__init__({BAG_PICKED, ABORT, TIMEOUT})
        self._node = YasminNode.get_instance()
        self._action_name = DEFAULT_PICK_ACTION
        self._client = ActionClient(self._node, Pick, self._action_name)

    def execute(self, blackboard: Blackboard) -> str:
        target_tf = str(blackboard["pick_target_tf"]).strip() or "brown_paper_bag_1"

        goal = Pick.Goal()
        goal.target_tf = target_tf

        def _result_ok(result_wrapper, result) -> tuple[bool, str]:
            status_code = None if result_wrapper is None else int(result_wrapper.status)
            blackboard["pick_action_status_code"] = -1 if status_code is None else status_code
            blackboard["pick_action_status_text"] = (
                "" if result is None else str(getattr(result, "status", "")).strip()
            )
            blackboard["pick_action_message"] = (
                "" if result is None else str(getattr(result, "message", "")).strip()
            )

            if (
                status_code == GoalStatus.STATUS_SUCCEEDED
                and result is not None
                and bool(result.success)
            ):
                return True, ""

            if result is not None and str(result.message).strip():
                return False, str(result.message).strip()

            return (
                False,
                f"Pick action failed with status_code={status_code} target_tf='{target_tf}'.",
            )

        try:
            retry_action_goal(
                self._node,
                self._client,
                goal,
                action_name=self._action_name,
                server_wait_attempts=int(blackboard["pick_server_wait_attempts"]),
                server_wait_timeout_sec=float(blackboard["pick_server_wait_timeout_sec"]),
                send_goal_attempts=int(blackboard["pick_send_goal_attempts"]),
                send_goal_timeout_sec=float(blackboard["pick_send_goal_timeout_sec"]),
                result_timeout_sec=float(blackboard["pick_result_timeout_sec"]),
                retry_delay_sec=float(blackboard["retry_delay_sec"]),
                result_ok=_result_ok,
            )
        except TimeoutError as exc:
            blackboard["last_error"] = str(exc)
            return TIMEOUT
        except Exception as exc:
            blackboard["last_error"] = str(exc)
            return ABORT

        blackboard["last_error"] = ""
        return BAG_PICKED


class GripperCommandState(State):
    def __init__(
        self,
        *,
        success_outcome: str,
        command_factory: Callable[[Blackboard], str],
        call_attempts_key: str,
        call_timeout_key: str,
        failure_message: str,
    ) -> None:
        super().__init__({success_outcome, ABORT, TIMEOUT})
        self._success_outcome = success_outcome
        self._command_factory = command_factory
        self._call_attempts_key = call_attempts_key
        self._call_timeout_key = call_timeout_key
        self._failure_message = failure_message
        self._node = YasminNode.get_instance()
        self._service_name = DEFAULT_OPEN_GRIPPER_SERVICE
        self._client = self._node.create_client(GripperCommand, self._service_name)

    def execute(self, blackboard: Blackboard) -> str:
        request = GripperCommand.Request()
        request.command = str(self._command_factory(blackboard)).strip()

        try:
            response = retry_service_call(
                self._node,
                self._client,
                request,
                service_name=self._service_name,
                wait_attempts=int(blackboard["service_wait_attempts"]),
                wait_timeout_sec=float(blackboard["service_wait_timeout_sec"]),
                call_attempts=int(blackboard[self._call_attempts_key]),
                call_timeout_sec=float(blackboard[self._call_timeout_key]),
                retry_delay_sec=float(blackboard["retry_delay_sec"]),
                response_ok=lambda r: r is not None and bool(r.success),
                response_error=lambda r: (
                    "" if r is None else str(getattr(r, "message", "")).strip()
                ) or self._failure_message,
            )
        except TimeoutError as exc:
            blackboard["last_error"] = str(exc)
            return TIMEOUT
        except Exception as exc:
            blackboard["last_error"] = str(exc)
            return ABORT

        blackboard["gripper_command_message"] = (
            "" if response is None else str(getattr(response, "message", "")).strip()
        )
        blackboard["last_gripper_command"] = request.command
        blackboard["last_error"] = ""
        return self._success_outcome


def build_state_machine() -> StateMachine:
    sm = StateMachine(outcomes=[FINAL_OUTCOME, ABORT])

    sm.add_state(
        "NAVIGATE_TO_INITIAL_POSE",
        NavigateToInitialPoseState(),
        transitions={
            NAVIGATION_DONE: "ANNOUNCE_READY_TO_START",
            ABORT: ABORT,
            TIMEOUT: ABORT,
        },
    )
    sm.add_state(
        "ANNOUNCE_READY_TO_START",
        SpeakState(lambda bb: str(bb["ready_to_start_text"]).strip()),
        transitions={
            SPOKEN: "WAIT_FOR_AWAKE",
            ABORT: ABORT,
        },
    )
    sm.add_state(
        "WAIT_FOR_AWAKE",
        WaitForAwakeState(),
        transitions={
            AWAKE_DETECTED: "DETECT_POINTED_BAG",
            ABORT: ABORT,
            TIMEOUT: ABORT,
        },
    )
    sm.add_state(
        "DETECT_POINTED_BAG",
        DetectPointedBagState(),
        transitions={
            BAG_DETECTED: "ANNOUNCE_SELECTED_BAG",
            ABORT: ABORT,
            TIMEOUT: ABORT,
        },
    )
    sm.add_state(
        "ANNOUNCE_SELECTED_BAG",
        SpeakState(
            lambda bb: f"I see you selected the bag on your {bb['detected_side']}"
        ),
        transitions={
            SPOKEN: "NAVIGATE_TO_SELECTED_BAG",
            ABORT: ABORT,
        },
    )
    sm.add_state(
        "NAVIGATE_TO_SELECTED_BAG",
        NavigateToSelectedBagState(),
        transitions={
            SELECTED_BAG_REACHED: "WAIT_BEFORE_PICKUP",
            ABORT: ABORT,
            TIMEOUT: ABORT,
        },
    )
    sm.add_state(
        "WAIT_BEFORE_PICKUP",
        DelayState(lambda bb: bb["pickup_delay_sec"]),
        transitions={
            DELAY_DONE: "ANNOUNCE_PICKUP_INTENT",
        },
    )
    sm.add_state(
        "ANNOUNCE_PICKUP_INTENT",
        SpeakState(lambda bb: str(bb["pickup_announcement_text"]).strip()),
        transitions={
            SPOKEN: "MOVE_ARM_TO_DETECT_POSE",
            ABORT: ABORT,
        },
    )
    sm.add_state(
        "MOVE_ARM_TO_DETECT_POSE",
        MoveArmToDetectPoseState(),
        transitions={
            ARM_POSE_READY: "WAIT_AFTER_ARM_DETECT_POSE",
            ABORT: ABORT,
            TIMEOUT: ABORT,
        },
    )
    sm.add_state(
        "WAIT_AFTER_ARM_DETECT_POSE",
        DelayState(lambda bb: bb["post_arm_detect_delay_sec"]),
        transitions={
            DELAY_DONE: "DETECT_BAG_WITH_GRIPPER_CAMERA",
        },
    )
    sm.add_state(
        "DETECT_BAG_WITH_GRIPPER_CAMERA",
        DetectBagWithGripperCameraState(),
        transitions={
            GRIPPER_BAG_DETECTED: "PICK_SELECTED_BAG",
            ABORT: "ANNOUNCE_MANUAL_BAG_HANDOFF",
            TIMEOUT: "ANNOUNCE_MANUAL_BAG_HANDOFF",
        },
    )
    sm.add_state(
        "PICK_SELECTED_BAG",
        PickBagState(),
        transitions={
            BAG_PICKED: "ANNOUNCE_START_FOLLOWING",
            ABORT: "ANNOUNCE_MANUAL_BAG_HANDOFF",
            TIMEOUT: "ANNOUNCE_MANUAL_BAG_HANDOFF",
        },
    )
    sm.add_state(
        "ANNOUNCE_MANUAL_BAG_HANDOFF",
        SpeakState(lambda bb: str(bb["manual_handoff_request_text"]).strip()),
        transitions={
            SPOKEN: "WAIT_FOR_MANUAL_BAG_HANDOFF",
            ABORT: ABORT,
        },
    )
    sm.add_state(
        "WAIT_FOR_MANUAL_BAG_HANDOFF",
        DelayState(lambda bb: bb["manual_handoff_wait_sec"]),
        transitions={
            DELAY_DONE: "CLOSE_GRIPPER_FOR_MANUAL_HANDOFF",
        },
    )
    sm.add_state(
        "CLOSE_GRIPPER_FOR_MANUAL_HANDOFF",
        GripperCommandState(
            success_outcome=GRIPPER_CLOSED,
            command_factory=lambda bb: str(bb["manual_handoff_gripper_command"]).strip() or "close",
            call_attempts_key="close_gripper_call_attempts",
            call_timeout_key="close_gripper_call_timeout_sec",
            failure_message="Failed to close the gripper for manual handoff.",
        ),
        transitions={
            GRIPPER_CLOSED: "ANNOUNCE_START_FOLLOWING",
            ABORT: ABORT,
            TIMEOUT: ABORT,
        },
    )
    sm.add_state(
        "ANNOUNCE_START_FOLLOWING",
        SpeakState(lambda bb: str(bb["start_following_text"]).strip()),
        transitions={
            SPOKEN: "ENABLE_FOLLOW_MODE",
            ABORT: ABORT,
        },
    )
    sm.add_state(
        "ENABLE_FOLLOW_MODE",
        SetFollowModeState(enable=True),
        transitions={
            FOLLOWING_ENABLED: "WAIT_FOR_STOP_COMMAND",
            ABORT: ABORT,
            TIMEOUT: ABORT,
        },
    )
    sm.add_state(
        "WAIT_FOR_STOP_COMMAND",
        WaitForStopCommandState(),
        transitions={
            STOP_COMMAND_DETECTED: "DISABLE_FOLLOW_MODE",
            ABORT: ABORT,
            TIMEOUT: ABORT,
        },
    )
    sm.add_state(
        "DISABLE_FOLLOW_MODE",
        SetFollowModeState(enable=False),
        transitions={
            FOLLOWING_ENABLED: "ANNOUNCE_STOP_FOLLOWING",
            ABORT: ABORT,
            TIMEOUT: ABORT,
        },
    )
    sm.add_state(
        "ANNOUNCE_STOP_FOLLOWING",
        SpeakState(lambda bb: str(bb["stop_following_text"]).strip()),
        transitions={
            SPOKEN: "WAIT_BEFORE_RELEASE_GRIPPER",
            ABORT: ABORT,
        },
    )
    sm.add_state(
        "WAIT_BEFORE_RELEASE_GRIPPER",
        DelayState(lambda bb: bb["pre_release_delay_sec"]),
        transitions={
            DELAY_DONE: "OPEN_GRIPPER",
        },
    )
    sm.add_state(
        "OPEN_GRIPPER",
        GripperCommandState(
            success_outcome=GRIPPER_OPENED,
            command_factory=lambda bb: str(bb["release_gripper_command"]).strip() or "open",
            call_attempts_key="open_gripper_call_attempts",
            call_timeout_key="open_gripper_call_timeout_sec",
            failure_message="Failed to open the gripper.",
        ),
        transitions={
            GRIPPER_OPENED: "WAIT_AFTER_RELEASE_GRIPPER",
            ABORT: ABORT,
            TIMEOUT: ABORT,
        },
    )
    sm.add_state(
        "WAIT_AFTER_RELEASE_GRIPPER",
        DelayState(lambda bb: bb["post_release_delay_sec"]),
        transitions={
            DELAY_DONE: "MOVE_ARM_TO_ZERO_POSE",
        },
    )
    sm.add_state(
        "MOVE_ARM_TO_ZERO_POSE",
        MoveArmToZeroPoseState(),
        transitions={
            ARM_POSE_READY: "RETURN_TO_INITIAL_POSE",
            ABORT: ABORT,
            TIMEOUT: ABORT,
        },
    )
    sm.add_state(
        "RETURN_TO_INITIAL_POSE",
        NavigateToInitialPoseState(),
        transitions={
            NAVIGATION_DONE: "ANNOUNCE_TASK_COMPLETE",
            ABORT: ABORT,
            TIMEOUT: ABORT,
        },
    )
    sm.add_state(
        "ANNOUNCE_TASK_COMPLETE",
        SpeakState(lambda bb: str(bb["task_complete_text"]).strip()),
        transitions={
            SPOKEN: FINAL_OUTCOME,
            ABORT: ABORT,
        },
    )

    return sm


def create_blackboard() -> Blackboard:
    blackboard = Blackboard()
    blackboard["sim"] = get_sim_mode()
    blackboard["initial_pose"] = dict(initial_pose)

    blackboard["retry_delay_sec"] = 1.0

    blackboard["service_wait_attempts"] = 5
    blackboard["service_wait_timeout_sec"] = 2.0

    blackboard["best_effort_wait_attempts"] = 2
    blackboard["best_effort_wait_timeout_sec"] = 0.5
    blackboard["best_effort_call_attempts"] = 2
    blackboard["best_effort_call_timeout_sec"] = 2.0

    blackboard["navigation_server_wait_attempts"] = 5
    blackboard["navigation_server_wait_timeout_sec"] = 2.0
    blackboard["navigation_goal_attempts"] = 3
    blackboard["navigation_send_goal_timeout_sec"] = 10.0
    blackboard["navigation_timeout_sec"] = 180.0

    blackboard["speak_server_wait_attempts"] = 5
    blackboard["speak_server_wait_timeout_sec"] = 2.0
    blackboard["speak_send_goal_attempts"] = 3
    blackboard["speak_send_goal_timeout_sec"] = 10.0
    blackboard["speak_result_timeout_sec"] = 30.0

    blackboard["robot_status_call_attempts"] = 3
    blackboard["robot_status_call_timeout_sec"] = 5.0
    blackboard["ready_to_start_text"] = "I am ready to start the task"
    blackboard["awake_wait_timeout_sec"] = 0.0
    blackboard["awake_poll_delay_sec"] = 0.05

    blackboard["pointed_bag_prompt"] = "brown paper bag"
    blackboard["pointed_detection_call_attempts"] = 3
    blackboard["pointed_detection_timeout_sec"] = 30.0
    blackboard["yolo_camera_name"] = get_yolo_detection_camera_name()
    blackboard["robot_base_frame"] = "base_footprint"
    blackboard["robot_base_frame_fallback"] = "base_link"
    blackboard["transform_timeout_sec"] = 0.5
    blackboard["bag_navigation_standoff_distance_m"] = 0.5
    blackboard["pickup_delay_sec"] = 3.0
    blackboard["pickup_announcement_text"] = "I will pick up the bag"
    blackboard["arm_detect_pose_name"] = "detect"
    blackboard["arm_zero_pose_name"] = "zero"
    blackboard["arm_pose_call_attempts"] = 3
    blackboard["arm_pose_call_timeout_sec"] = 10.0
    blackboard["post_arm_detect_delay_sec"] = 3.0
    blackboard["gripper_bag_prompt"] = "brown paper bag"
    blackboard["gripper_camera_name"] = DEFAULT_GRIPPER_CAMERA_NAME
    blackboard["gripper_detection_call_attempts"] = 3
    blackboard["gripper_detection_timeout_sec"] = 30.0
    blackboard["pick_target_tf"] = "brown_paper_bag_1"
    blackboard["pick_server_wait_attempts"] = 5
    blackboard["pick_server_wait_timeout_sec"] = 2.0
    blackboard["pick_send_goal_attempts"] = 3
    blackboard["pick_send_goal_timeout_sec"] = 10.0
    blackboard["pick_result_timeout_sec"] = 120.0
    blackboard["manual_handoff_request_text"] = "I fail to get the bag, can you pass it to me"
    blackboard["manual_handoff_wait_sec"] = 5.0
    blackboard["manual_handoff_gripper_command"] = "close"
    blackboard["close_gripper_call_attempts"] = 3
    blackboard["close_gripper_call_timeout_sec"] = 10.0
    blackboard["start_following_text"] = "I will start folowing now"
    blackboard["follow_enable_call_attempts"] = 3
    blackboard["follow_enable_call_timeout_sec"] = 5.0
    blackboard["follow_disable_call_attempts"] = 3
    blackboard["follow_disable_call_timeout_sec"] = 5.0
    blackboard["get_command_call_attempts"] = 1
    blackboard["get_command_call_timeout_sec"] = 20.0
    blackboard["stop_command_wait_timeout_sec"] = 0.0
    blackboard["stop_command_retry_delay_sec"] = 0.1
    blackboard["stop_command_keywords"] = ["stop"]
    blackboard["stop_following_text"] = (
        "OK, I will stop folllow you, remmember to take your luggage"
    )
    blackboard["pre_release_delay_sec"] = 3.0
    blackboard["release_gripper_command"] = "open"
    blackboard["open_gripper_call_attempts"] = 3
    blackboard["open_gripper_call_timeout_sec"] = 10.0
    blackboard["post_release_delay_sec"] = 3.0
    blackboard["task_complete_text"] = "I had complete my task now"

    blackboard["detected_side"] = ""
    blackboard["pointed_detection_message"] = ""
    blackboard["pointed_saved_image_path"] = ""
    blackboard["pointed_detections_in_frame"] = 0
    blackboard["selected_bag_pose"] = None
    blackboard["selected_bag_tf"] = ""
    blackboard["approach_bag_pose"] = None
    blackboard["navigation_status"] = -1
    blackboard["navigation_error_code"] = -1
    blackboard["navigation_skipped"] = False
    blackboard["bag_navigation_status"] = -1
    blackboard["bag_navigation_error_code"] = -1
    blackboard["bag_navigation_skipped"] = False
    blackboard["bag_navigation_target_distance_m"] = -1.0
    blackboard["bag_navigation_used_fallback_pose"] = False
    blackboard["robot_status_message"] = ""
    blackboard["last_spoken_text"] = ""
    blackboard["arm_pose_message"] = ""
    blackboard["gripper_detection_message"] = ""
    blackboard["gripper_saved_image_path"] = ""
    blackboard["gripper_detections_in_frame"] = 0
    blackboard["pick_action_status_code"] = -1
    blackboard["pick_action_status_text"] = ""
    blackboard["pick_action_message"] = ""
    blackboard["follow_enabled"] = False
    blackboard["follow_service_message"] = ""
    blackboard["last_heard_command"] = ""
    blackboard["stop_command_transcript"] = ""
    blackboard["gripper_command_message"] = ""
    blackboard["last_gripper_command"] = ""
    blackboard["last_error"] = ""
    return blackboard


def main() -> None:
    rclpy.init()
    set_ros_loggers()
    yasmin.YASMIN_LOG_INFO("task_state_machine_carry_my_luggage")

    state_machine = build_state_machine()
    YasminViewerPub(state_machine, "CARRY_MY_LUGGAGE_TASK")
    blackboard = create_blackboard()

    try:
        ensure_follow_mode_disabled_on_startup(blackboard)
        outcome = state_machine(blackboard)
        yasmin.YASMIN_LOG_INFO(
            "Carry-my-luggage state machine finished with outcome=%s sim=%s nav_skipped=%s bag_nav_skipped=%s follow_enabled=%s side=%s pick_target_tf=%s stop_command=%s error=%s"
            % (
                outcome,
                blackboard["sim"],
                blackboard["navigation_skipped"],
                blackboard["bag_navigation_skipped"],
                blackboard["follow_enabled"],
                blackboard["detected_side"],
                blackboard["pick_target_tf"],
                blackboard["stop_command_transcript"],
                blackboard["last_error"],
            )
        )
    except Exception as exc:
        yasmin.YASMIN_LOG_WARN(f"Carry-my-luggage state machine crashed: {exc}")
        raise
    finally:
        shutdown_yasmin_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

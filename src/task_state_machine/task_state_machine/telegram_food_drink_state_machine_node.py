#!/usr/bin/env python3
"""Telegram-driven state machine for food/drink sorting demo."""

from __future__ import annotations

import json
import math
import re
import threading
import time
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import requests
import rclpy
from action_msgs.msg import GoalStatus
from coqui_tts_interfaces.action import SpeakText
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TransformStamped
from nav2_msgs.action import NavigateToPose
from rcl_interfaces.msg import ParameterType
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from tf2_ros import TransformBroadcaster
from yasmin import Blackboard, State, StateMachine
from yasmin_ros.basic_outcomes import ABORT

from robot_arm_action.action import Pick
from robot_arm_action.srv import GripperCommand
from robot_arm_action.srv import ArmPose
from vlm_interfaces.srv import FoodDrinkSort
from yoloe_detection_interfaces.srv import DetectObjectPrompt


class WorkflowState(str, Enum):
    RECEIVED = "received"
    ARM_MOVING = "arm_moving"
    NAVIGATING = "navigating"
    YOLO_DETECTING = "yolo_detecting"
    ANALYZING = "analyzing"
    ANNOUNCING = "announcing"
    PICKING = "picking"
    PLACING = "placing"
    COMPLETED = "completed"
    FAILED = "failed"


STEP_DONE = "step_done"
WORKFLOW_DONE = "workflow_done"
WORKFLOW_FAILED = "workflow_failed"


class RetryState(State):
    def __init__(self, state_name: str, inner_state: State, retry_delay_sec: float = 1.0) -> None:
        super().__init__({STEP_DONE, ABORT})
        self._state_name = state_name
        self._inner_state = inner_state
        self._retry_delay_sec = max(0.0, float(retry_delay_sec))

    def execute(self, blackboard: Blackboard) -> str:
        node = blackboard["node"]
        chat_id = blackboard["chat_id"]
        attempt = 1
        while rclpy.ok():
            outcome = self._inner_state.execute(blackboard)
            if outcome != ABORT:
                if attempt > 1:
                    node.get_logger().info(
                        f"State '{self._state_name}' recovered after {attempt} attempt(s)."
                    )
                return outcome

            error_text = str(blackboard.get("last_error") or "").strip() or "Unknown failure."
            node.get_logger().warn(
                f"State '{self._state_name}' failed on attempt {attempt}; retrying. reason={error_text}"
            )
            node._log_state(chat_id, WorkflowState.FAILED, f"{self._state_name} retry {attempt}: {error_text}")
            attempt += 1
            if self._retry_delay_sec > 0.0:
                time.sleep(self._retry_delay_sec)

        blackboard["last_error"] = f"State '{self._state_name}' interrupted while retrying."
        return ABORT


# Configure named navigation targets here (same style as carry-my-luggage/lost-found).
location_pose_map: dict[str, dict[str, Any]] = {
    "table": {
        "frame_id": "map",
        "x": -0.31224,
        "y": -4.15069,
        "z": 0.0,
        "orientation": {
            "x": 0.0,
            "y": 0.0,
            "z": 0.716076,
            "w": 0.698022,
        },
    },
}

class TelegramClient:
    def __init__(self, token: str, timeout_sec: float = 20.0) -> None:
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.timeout_sec = max(1.0, float(timeout_sec))

    def get_updates(self, offset: int | None) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {
            "timeout": int(self.timeout_sec),
            "allowed_updates": ["message"],
        }
        if offset is not None:
            payload["offset"] = int(offset)
        response = requests.post(
            f"{self.base_url}/getUpdates",
            json=payload,
            timeout=self.timeout_sec + 5.0,
        )
        response.raise_for_status()
        data = response.json()
        if not data.get("ok"):
            raise RuntimeError(str(data))
        result = data.get("result")
        return result if isinstance(result, list) else []

    def send_message(self, chat_id: int | str, text: str, reply_to_message_id: int | None = None) -> None:
        payload: dict[str, Any] = {"chat_id": chat_id, "text": text}
        if reply_to_message_id is not None:
            payload["reply_to_message_id"] = int(reply_to_message_id)
        response = requests.post(
            f"{self.base_url}/sendMessage",
            json=payload,
            timeout=15.0,
        )
        response.raise_for_status()

    def send_photo(
        self,
        chat_id: int | str,
        image_path: str,
        caption: str,
        reply_to_message_id: int | None = None,
    ) -> bool:
        path = Path(image_path).expanduser()
        if not path.is_file():
            return False
        data: dict[str, Any] = {"chat_id": str(chat_id), "caption": caption}
        if reply_to_message_id is not None:
            data["reply_to_message_id"] = str(int(reply_to_message_id))
        with path.open("rb") as image_file:
            response = requests.post(
                f"{self.base_url}/sendPhoto",
                data=data,
                files={"photo": (path.name, image_file)},
                timeout=(10.0, 60.0),
            )
        response.raise_for_status()
        return True


class AckAndSpeakState(State):
    def __init__(self) -> None:
        super().__init__({STEP_DONE, ABORT})

    def execute(self, blackboard: Blackboard) -> str:
        node = blackboard["node"]
        chat_id = blackboard["chat_id"]
        reply_to_message_id = blackboard["reply_to_message_id"]
        try:
            node._log_state(chat_id, WorkflowState.RECEIVED, "request accepted")
            node._telegram.send_message(chat_id, "ok, please wait", reply_to_message_id)
            speak_ok, speak_message = node._speak_text("I receive a command to clean up the table")
            if not speak_ok:
                node.get_logger().warn(f"SpeakText failed; continuing workflow: {speak_message}")
            blackboard["last_error"] = ""
            return STEP_DONE
        except Exception as exc:
            blackboard["last_error"] = str(exc)
            return ABORT


class MoveArmCenterState(State):
    def __init__(self) -> None:
        super().__init__({STEP_DONE, ABORT})

    def execute(self, blackboard: Blackboard) -> str:
        node = blackboard["node"]
        chat_id = blackboard["chat_id"]
        try:
            node._log_state(chat_id, WorkflowState.ARM_MOVING, node.arm_center_pose_name)
            ok, message = node._move_arm_pose(node.arm_center_pose_name)
            if not ok:
                blackboard["last_error"] = f"I could not move the arm: {message}"
                return ABORT
            blackboard["last_error"] = ""
            return STEP_DONE
        except Exception as exc:
            blackboard["last_error"] = f"I could not move the arm: {exc}"
            return ABORT


class NavigateTableState(State):
    def __init__(self) -> None:
        super().__init__({STEP_DONE, ABORT})

    def execute(self, blackboard: Blackboard) -> str:
        node = blackboard["node"]
        chat_id = blackboard["chat_id"]
        try:
            node._log_state(chat_id, WorkflowState.NAVIGATING, "table")
            ok, message = node._navigate_to_location("table")
            if not ok:
                blackboard["last_error"] = f"I could not navigate to table: {message}"
                return ABORT
            blackboard["last_error"] = ""
            return STEP_DONE
        except Exception as exc:
            blackboard["last_error"] = f"I could not navigate to table: {exc}"
            return ABORT


class DetectPromptState(State):
    def __init__(self, prompt_text: str = "") -> None:
        super().__init__({STEP_DONE, ABORT})
        self._prompt_text = str(prompt_text).strip()

    def execute(self, blackboard: Blackboard) -> str:
        node = blackboard["node"]
        chat_id = blackboard["chat_id"]
        prompt_text = self._prompt_text or str(node.yoloe_prompt_text).strip()
        try:
            blackboard["pick_target_tf"] = ""
            node._log_state(chat_id, WorkflowState.YOLO_DETECTING, prompt_text)
            ok, message, target_tf = node._call_yoloe_detect(
                prompt_text=prompt_text,
                camera_name=node.default_camera_name,
                save_image=node.yoloe_save_image,
            )
            target_tf_clean = str(target_tf).strip()
            if ok and not target_tf_clean:
                ok = False
                message = "YOLO detect returned success but no target TF."
            if ok:
                ok, adjusted_message, target_tf_clean = node._create_grasp_target_tf(target_tf_clean)
                if not ok:
                    message = adjusted_message
            blackboard["pick_target_tf"] = target_tf_clean
            if not ok:
                strict_required = bool(node.yoloe_detection_required) and not bool(node.demo_mode)
                if strict_required:
                    blackboard["last_error"] = f"YOLO detection failed: {message}"
                    return ABORT
                node.get_logger().warn(f"YOLO detection failed; continuing: {message}")
            blackboard["last_error"] = ""
            return STEP_DONE
        except Exception as exc:
            strict_required = bool(node.yoloe_detection_required) and not bool(node.demo_mode)
            if strict_required:
                blackboard["last_error"] = f"YOLO detection failed: {exc}"
                return ABORT
            node.get_logger().warn(f"YOLO detection failed; continuing: {exc}")
            blackboard["last_error"] = ""
            return STEP_DONE


class AnalyzeFoodDrinkState(State):
    def __init__(self) -> None:
        super().__init__({STEP_DONE, ABORT})

    def execute(self, blackboard: Blackboard) -> str:
        node = blackboard["node"]
        chat_id = blackboard["chat_id"]
        try:
            node._log_state(chat_id, WorkflowState.ANALYZING, node.default_camera_name)
            result = node._call_food_drink_sort(node.default_camera_name, node.default_confidence_threshold)
            if result is None:
                if node.demo_mode:
                    blackboard["food_drink_result"] = SimpleNamespace(
                        announcement="I could not detect food and drinks clearly this time, but I will keep trying.",
                        image_path="",
                    )
                    blackboard["last_error"] = ""
                    return STEP_DONE
                blackboard["last_error"] = "I could not complete the food/drink check in time."
                return ABORT
            if not bool(result.success):
                if node.demo_mode:
                    blackboard["food_drink_result"] = SimpleNamespace(
                        announcement=f"I could not detect food and drinks clearly: {result.message}",
                        image_path=str(getattr(result, "image_path", "")).strip(),
                    )
                    blackboard["last_error"] = ""
                    return STEP_DONE
                blackboard["last_error"] = f"I could not complete the check: {result.message}"
                return ABORT
            blackboard["food_drink_result"] = result
            blackboard["last_error"] = ""
            return STEP_DONE
        except Exception as exc:
            if node.demo_mode:
                blackboard["food_drink_result"] = SimpleNamespace(
                    announcement=f"I could not detect food and drinks clearly: {exc}",
                    image_path="",
                )
                blackboard["last_error"] = ""
                return STEP_DONE
            blackboard["last_error"] = f"I could not complete the check: {exc}"
            return ABORT


class PickCoffeeState(State):
    def __init__(self) -> None:
        super().__init__({STEP_DONE, ABORT})

    def execute(self, blackboard: Blackboard) -> str:
        node = blackboard["node"]
        chat_id = blackboard["chat_id"]
        target_tf = str(blackboard.get("pick_target_tf") or "").strip()
        try:
            node._log_state(chat_id, WorkflowState.PICKING, target_tf or "<empty>")
            ok, message = node._pick_object(target_tf)
            if not ok:
                if node.demo_mode:
                    node.get_logger().warn(f"Pick action failed in demo mode; continuing: {message}")
                    blackboard["last_error"] = ""
                    return STEP_DONE
                blackboard["last_error"] = f"I could not pick the object: {message}"
                return ABORT
            blackboard["last_error"] = ""
            return STEP_DONE
        except Exception as exc:
            if node.demo_mode:
                node.get_logger().warn(f"Pick action failed in demo mode; continuing: {exc}")
                blackboard["last_error"] = ""
                return STEP_DONE
            blackboard["last_error"] = f"I could not pick the object: {exc}"
            return ABORT


class PlaceCoffeeRightState(State):
    def __init__(self) -> None:
        super().__init__({STEP_DONE, ABORT})

    def execute(self, blackboard: Blackboard) -> str:
        node = blackboard["node"]
        chat_id = blackboard["chat_id"]
        try:
            node._log_state(chat_id, WorkflowState.PLACING, "center -> right -> open -> center")
            ok, message = node._place_coffee_sequence()
            if not ok:
                blackboard["last_error"] = f"I could not place coffee: {message}"
                return ABORT
            blackboard["last_error"] = ""
            return STEP_DONE
        except Exception as exc:
            blackboard["last_error"] = f"I could not place coffee: {exc}"
            return ABORT


class PlaceCupNoodleLeftState(State):
    def __init__(self) -> None:
        super().__init__({STEP_DONE, ABORT})

    def execute(self, blackboard: Blackboard) -> str:
        node = blackboard["node"]
        chat_id = blackboard["chat_id"]
        try:
            node._log_state(chat_id, WorkflowState.PLACING, "center -> left -> open -> center")
            ok, message = node._place_left_sequence()
            if not ok:
                blackboard["last_error"] = f"I could not place cup noodle: {message}"
                return ABORT
            blackboard["last_error"] = ""
            return STEP_DONE
        except Exception as exc:
            blackboard["last_error"] = f"I could not place cup noodle: {exc}"
            return ABORT


class AnnounceResultState(State):
    def __init__(self) -> None:
        super().__init__({STEP_DONE, ABORT})

    def execute(self, blackboard: Blackboard) -> str:
        node = blackboard["node"]
        chat_id = blackboard["chat_id"]
        reply_to_message_id = blackboard["reply_to_message_id"]
        try:
            node._log_state(chat_id, WorkflowState.ANNOUNCING, "reporting result")
            result = blackboard.get("food_drink_result")
            announcement = str(result.announcement).strip() or (
                "I checked the scene and I will arrange food on the left and drinks on the right."
            )
            image_path = str(result.image_path).strip()
            sent_photo = False
            if image_path:
                try:
                    sent_photo = node._telegram.send_photo(chat_id, image_path, announcement, reply_to_message_id)
                except Exception as exc:
                    node.get_logger().warn(f"Failed to send photo to Telegram: {exc}")
            if not sent_photo:
                node._telegram.send_message(chat_id, announcement, reply_to_message_id)
            speak_ok, speak_message = node._speak_text(announcement)
            if not speak_ok:
                node.get_logger().warn(f"SpeakText failed for final announcement; continuing: {speak_message}")
            blackboard["last_error"] = ""
            return STEP_DONE
        except Exception as exc:
            blackboard["last_error"] = str(exc)
            return ABORT


class TelegramFoodDrinkStateMachineNode(Node):
    def __init__(self) -> None:
        super().__init__("telegram_food_drink_state_machine_node")

        self.declare_parameter("bot_token", "")
        self.declare_parameter("bot_token_file", "")
        self.declare_parameter("allowed_user_ids_csv", "")
        self.declare_parameter("default_camera_name", "gripper_camera")
        self.declare_parameter(
            "default_chat_id",
            "",
            descriptor=self._dynamic_parameter_descriptor(
                "Telegram chat id used when service calls leave chat_id empty."
            ),
        )
        self.declare_parameter("food_drink_sort_service", "/food_drink/sort")
        self.declare_parameter("yoloe_detect_service", "/yoloe/detect_prompt")
        self.declare_parameter("yoloe_prompt_text", "can_coffee")
        self.declare_parameter("yoloe_juice_prompt_text", "juice_bottle")
        self.declare_parameter("yoloe_cup_noodle_prompt_text", "cup_noodle")
        self.declare_parameter("yoloe_cookies_prompt_text", "smart_bear_cookies")
        self.declare_parameter("yoloe_save_image", True)
        self.declare_parameter("yoloe_timeout_sec", 35.0)
        self.declare_parameter("yoloe_detection_required", True)
        self.declare_parameter("navigation_action_name", "/navigate_to_pose")
        self.declare_parameter("arm_pose_service", "/arm_pose")
        self.declare_parameter("open_gripper_service", "/open_gripper")
        self.declare_parameter("arm_center_pose_name", "center")
        self.declare_parameter("navigation_server_wait_timeout_sec", 5.0)
        self.declare_parameter("navigation_send_goal_timeout_sec", 5.0)
        self.declare_parameter("navigation_timeout_sec", 90.0)
        self.declare_parameter("arm_pose_timeout_sec", 15.0)
        self.declare_parameter("gripper_command_timeout_sec", 10.0)
        self.declare_parameter("pick_action_name", "/pick_object")
        self.declare_parameter("pick_action_wait_timeout_sec", 5.0)
        self.declare_parameter("pick_timeout_sec", 45.0)
        self.declare_parameter("demo_mode", True)
        self.declare_parameter("sim_mode", False)
        self.declare_parameter("demo_navigation_delay_sec", 5.0)
        self.declare_parameter("location_pose_map_json", "")
        self.declare_parameter("speak_action_name", "/coqui_tts/speak")
        self.declare_parameter("speak_action_wait_timeout_sec", 1.0)
        self.declare_parameter("speak_timeout_sec", 20.0)
        self.declare_parameter("service_wait_timeout_sec", 5.0)
        self.declare_parameter("service_timeout_sec", 75.0)
        self.declare_parameter("default_confidence_threshold", 0.25)
        self.declare_parameter("telegram_poll_timeout_sec", 20.0)
        self.declare_parameter("state_retry_delay_sec", 1.0)
        self.declare_parameter("detect_retry_warmup_sec", 1.5)
        self.declare_parameter("camera_image_topic", "")
        self.declare_parameter("camera_frame_wait_timeout_sec", 3.0)
        self.declare_parameter("grasp_tf_y_offset_m", 0.015)
        self.declare_parameter("grasp_tf_suffix", "_grasp")

        token = str(self.get_parameter("bot_token").value).strip()
        token_file = str(self.get_parameter("bot_token_file").value).strip()
        if not token and token_file:
            token_path = Path(token_file).expanduser()
            if not token_path.is_file():
                raise ValueError(
                    f"Telegram bot token file not found: {token_path}. "
                    "Create it with @BotFather token, or pass bot_token:=..."
                )
            token = token_path.read_text(encoding="utf-8").strip()
        if not token:
            raise ValueError("Set bot_token or bot_token_file for Telegram polling.")

        self.allowed_user_ids = self._parse_int_set(str(self.get_parameter("allowed_user_ids_csv").value).strip())
        self.default_camera_name = str(self.get_parameter("default_camera_name").value).strip() or "gripper_camera"
        self.default_chat_id = str(self.get_parameter("default_chat_id").value).strip()
        self.food_drink_sort_service = (
            str(self.get_parameter("food_drink_sort_service").value).strip() or "/food_drink/sort"
        )
        self.yoloe_detect_service = (
            str(self.get_parameter("yoloe_detect_service").value).strip() or "/yoloe/detect_prompt"
        )
        self.yoloe_prompt_text = str(self.get_parameter("yoloe_prompt_text").value).strip() or "can_coffee"
        self.yoloe_juice_prompt_text = str(self.get_parameter("yoloe_juice_prompt_text").value).strip() or "juice_bottle"
        self.yoloe_cup_noodle_prompt_text = (
            str(self.get_parameter("yoloe_cup_noodle_prompt_text").value).strip() or "cup_noodle"
        )
        self.yoloe_cookies_prompt_text = (
            str(self.get_parameter("yoloe_cookies_prompt_text").value).strip() or "smart_bear_cookies"
        )
        self.yoloe_save_image = bool(self.get_parameter("yoloe_save_image").value)
        self.yoloe_timeout_sec = max(1.0, float(self.get_parameter("yoloe_timeout_sec").value))
        self.yoloe_detection_required = bool(self.get_parameter("yoloe_detection_required").value)
        self.navigation_action_name = (
            str(self.get_parameter("navigation_action_name").value).strip() or "/navigate_to_pose"
        )
        self.arm_pose_service = str(self.get_parameter("arm_pose_service").value).strip() or "/arm_pose"
        self.open_gripper_service = str(self.get_parameter("open_gripper_service").value).strip() or "/open_gripper"
        self.arm_center_pose_name = str(self.get_parameter("arm_center_pose_name").value).strip() or "center"
        self.navigation_server_wait_timeout_sec = max(
            0.1, float(self.get_parameter("navigation_server_wait_timeout_sec").value)
        )
        self.navigation_send_goal_timeout_sec = max(
            0.1, float(self.get_parameter("navigation_send_goal_timeout_sec").value)
        )
        self.navigation_timeout_sec = max(1.0, float(self.get_parameter("navigation_timeout_sec").value))
        self.arm_pose_timeout_sec = max(0.1, float(self.get_parameter("arm_pose_timeout_sec").value))
        self.gripper_command_timeout_sec = max(0.1, float(self.get_parameter("gripper_command_timeout_sec").value))
        self.pick_action_name = str(self.get_parameter("pick_action_name").value).strip() or "/pick_object"
        self.pick_action_wait_timeout_sec = max(
            0.1, float(self.get_parameter("pick_action_wait_timeout_sec").value)
        )
        self.pick_timeout_sec = max(0.1, float(self.get_parameter("pick_timeout_sec").value))
        self.demo_mode = bool(self.get_parameter("demo_mode").value)
        self.sim_mode = bool(self.get_parameter("sim_mode").value)
        self.demo_navigation_delay_sec = max(0.0, float(self.get_parameter("demo_navigation_delay_sec").value))
        self.location_pose_map = {
            self._normalize_location_key(key): dict(value) for key, value in location_pose_map.items()
        }
        location_pose_map_override = self._parse_location_pose_map_json(
            str(self.get_parameter("location_pose_map_json").value).strip()
        )
        if location_pose_map_override:
            self.location_pose_map.update(location_pose_map_override)
        self.speak_action_name = str(self.get_parameter("speak_action_name").value).strip() or "/coqui_tts/speak"
        self.speak_action_wait_timeout_sec = max(
            0.1, float(self.get_parameter("speak_action_wait_timeout_sec").value)
        )
        self.speak_timeout_sec = max(1.0, float(self.get_parameter("speak_timeout_sec").value))
        self.service_wait_timeout_sec = max(0.1, float(self.get_parameter("service_wait_timeout_sec").value))
        self.service_timeout_sec = max(1.0, float(self.get_parameter("service_timeout_sec").value))
        self.default_confidence_threshold = max(
            0.0,
            min(1.0, float(self.get_parameter("default_confidence_threshold").value)),
        )
        self.state_retry_delay_sec = max(0.0, float(self.get_parameter("state_retry_delay_sec").value))
        self.detect_retry_warmup_sec = max(0.0, float(self.get_parameter("detect_retry_warmup_sec").value))
        self.camera_image_topic = str(self.get_parameter("camera_image_topic").value).strip()
        if not self.camera_image_topic:
            self.camera_image_topic = f"/{self.default_camera_name}/color/image_raw"
        self.camera_frame_wait_timeout_sec = max(
            0.1,
            float(self.get_parameter("camera_frame_wait_timeout_sec").value),
        )
        self.grasp_tf_y_offset_m = float(self.get_parameter("grasp_tf_y_offset_m").value)
        self.grasp_tf_suffix = str(self.get_parameter("grasp_tf_suffix").value).strip() or "_grasp"

        self._callback_group = ReentrantCallbackGroup()
        self._food_drink_client = self.create_client(
            FoodDrinkSort,
            self.food_drink_sort_service,
            callback_group=self._callback_group,
        )
        self._yoloe_client = self.create_client(
            DetectObjectPrompt,
            self.yoloe_detect_service,
            callback_group=self._callback_group,
        )
        self._arm_pose_client = self.create_client(
            ArmPose,
            self.arm_pose_service,
            callback_group=self._callback_group,
        )
        self._open_gripper_client = self.create_client(
            GripperCommand,
            self.open_gripper_service,
            callback_group=self._callback_group,
        )
        self._navigate_client = ActionClient(
            self,
            NavigateToPose,
            self.navigation_action_name,
            callback_group=self._callback_group,
        )
        self._speak_action_client = ActionClient(
            self,
            SpeakText,
            self.speak_action_name,
            callback_group=self._callback_group,
        )
        self._pick_action_client = ActionClient(
            self,
            Pick,
            self.pick_action_name,
            callback_group=self._callback_group,
        )
        self._tf_broadcaster = TransformBroadcaster(self)
        self._camera_lock = threading.Lock()
        self._camera_frame_event = threading.Event()
        self._camera_frame_count = 0
        self._camera_last_frame_monotonic = 0.0
        self._camera_subscription = self.create_subscription(
            Image,
            self.camera_image_topic,
            self._camera_image_callback,
            qos_profile_sensor_data,
            callback_group=self._callback_group,
        )
        self._telegram = TelegramClient(
            token,
            timeout_sec=float(self.get_parameter("telegram_poll_timeout_sec").value),
        )
        self._stop_event = threading.Event()
        self._state_lock = threading.Lock()
        self._active_chats: set[int | str] = set()
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()

        self.get_logger().info(
            "Telegram food-drink state machine started. "
            "sort_service=%s yoloe_service=%s arm_service=%s open_gripper_service=%s nav_action=%s pick_action=%s demo_mode=%s sim_mode=%s default_camera=%s"
            % (
                self.food_drink_sort_service,
                self.yoloe_detect_service,
                self.arm_pose_service,
                self.open_gripper_service,
                self.navigation_action_name,
                self.pick_action_name,
                str(self.demo_mode).lower(),
                str(self.sim_mode).lower(),
                self.default_camera_name,
            )
        )

    @staticmethod
    def _dynamic_parameter_descriptor(description: str):
        from rcl_interfaces.msg import ParameterDescriptor

        descriptor = ParameterDescriptor()
        descriptor.description = description
        descriptor.dynamic_typing = True
        descriptor.type = ParameterType.PARAMETER_NOT_SET
        return descriptor

    def destroy_node(self) -> bool:
        self._stop_event.set()
        return super().destroy_node()

    @staticmethod
    def _parse_int_set(csv_text: str) -> set[int]:
        values: set[int] = set()
        for part in csv_text.split(","):
            text = part.strip()
            if not text:
                continue
            values.add(int(text))
        return values

    @staticmethod
    def _normalize_location_key(location: str) -> str:
        return re.sub(r"\s+", " ", str(location).strip().lower())

    def _parse_location_pose_map_json(self, text: str) -> dict[str, dict[str, Any]]:
        if not text:
            return {}
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            self.get_logger().warn(f"Invalid location_pose_map_json; ignoring it: {exc}")
            return {}
        if not isinstance(payload, dict):
            self.get_logger().warn("location_pose_map_json is not a JSON object; ignoring it.")
            return {}

        result: dict[str, dict[str, Any]] = {}
        for key, value in payload.items():
            if not isinstance(value, dict):
                continue
            normalized = self._normalize_location_key(key)
            if not normalized:
                continue
            result[normalized] = value
        return result

    def _camera_image_callback(self, _msg: Image) -> None:
        with self._camera_lock:
            self._camera_frame_count += 1
            self._camera_last_frame_monotonic = time.monotonic()
            self._camera_frame_event.set()

    def _camera_frame_snapshot(self) -> tuple[int, float]:
        with self._camera_lock:
            return self._camera_frame_count, self._camera_last_frame_monotonic

    def _wait_for_new_camera_frame(self, timeout_sec: float) -> bool:
        baseline_count, _ = self._camera_frame_snapshot()
        deadline = time.monotonic() + max(0.1, timeout_sec)
        while rclpy.ok() and time.monotonic() < deadline:
            remaining = max(0.05, deadline - time.monotonic())
            self._camera_frame_event.wait(timeout=min(remaining, 0.25))
            self._camera_frame_event.clear()
            current_count, _ = self._camera_frame_snapshot()
            if current_count > baseline_count:
                return True
        return False

    def _create_grasp_target_tf(self, detected_tf: str) -> tuple[bool, str, str]:
        detected_tf_clean = str(detected_tf).strip()
        if not detected_tf_clean:
            return False, "No target TF from YOLO detection.", ""

        grasp_tf = f"{detected_tf_clean}{self.grasp_tf_suffix}"
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = detected_tf_clean
        transform.child_frame_id = grasp_tf
        transform.transform.translation.x = 0.0
        transform.transform.translation.y = float(self.grasp_tf_y_offset_m)
        transform.transform.translation.z = 0.0
        transform.transform.rotation.x = 0.0
        transform.transform.rotation.y = 0.0
        transform.transform.rotation.z = 0.0
        transform.transform.rotation.w = 1.0
        self._tf_broadcaster.sendTransform(transform)
        self.get_logger().info(
            f"Published grasp TF '{grasp_tf}' from '{detected_tf_clean}' with y offset {self.grasp_tf_y_offset_m:.3f} m."
        )
        return True, "ok", grasp_tf

    def _location_pose_for(self, location: str) -> dict[str, Any] | None:
        return self.location_pose_map.get(self._normalize_location_key(location))

    def _create_pose_stamped_from_config(self, pose_config: dict[str, Any]) -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id = str(pose_config.get("frame_id") or "map").strip() or "map"
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(pose_config.get("x", 0.0))
        pose.pose.position.y = float(pose_config.get("y", 0.0))
        pose.pose.position.z = float(pose_config.get("z", 0.0))
        orientation = pose_config.get("orientation")
        if isinstance(orientation, dict):
            pose.pose.orientation.x = float(orientation.get("x", 0.0))
            pose.pose.orientation.y = float(orientation.get("y", 0.0))
            pose.pose.orientation.z = float(orientation.get("z", 0.0))
            pose.pose.orientation.w = float(orientation.get("w", 1.0))
        else:
            yaw = float(pose_config.get("yaw", 0.0))
            pose.pose.orientation.z = math.sin(yaw * 0.5)
            pose.pose.orientation.w = math.cos(yaw * 0.5)
        return pose

    def _poll_loop(self) -> None:
        offset: int | None = None
        while rclpy.ok() and not self._stop_event.is_set():
            try:
                updates = self._telegram.get_updates(offset)
                for update in updates:
                    update_id = int(update.get("update_id", 0))
                    offset = max(update_id + 1, offset or 0)
                    self._handle_update(update)
            except Exception as exc:
                self.get_logger().warn(f"Telegram polling error: {exc}")
                time.sleep(2.0)

    def _handle_update(self, update: dict[str, Any]) -> None:
        message = update.get("message")
        if not isinstance(message, dict):
            return
        chat = message.get("chat") if isinstance(message.get("chat"), dict) else {}
        sender = message.get("from") if isinstance(message.get("from"), dict) else {}
        chat_id = chat.get("id")
        user_id = sender.get("id")
        message_id = message.get("message_id")
        text = str(message.get("text") or "").strip()
        if chat_id is None or not text:
            return

        self.get_logger().info(f"Telegram message | chat_id={chat_id} user_id={user_id} text={text[:120]!r}")

        if self.allowed_user_ids and int(user_id or 0) not in self.allowed_user_ids:
            self._telegram.send_message(chat_id, "Sorry, this bot is restricted.", self._message_id_or_none(message_id))
            return

        lowered = text.lower().strip()
        if lowered.startswith("/start"):
            self._telegram.send_message(
                chat_id,
                (
                    "Hi, I can check visible items and separate food and drinks. "
                    "Try /sort or say: separate food and drinks."
                ),
                self._message_id_or_none(message_id),
            )
            return

        if not self._is_food_drink_request(lowered):
            return
        self._start_workflow(chat_id, self._message_id_or_none(message_id))

    @staticmethod
    def _is_food_drink_request(lowered: str) -> bool:
        if lowered.startswith("/sort") or lowered.startswith("/arrange"):
            return True
        normalized = re.sub(r"[^a-z0-9\s]", " ", lowered)
        normalized = re.sub(r"\s+", " ", normalized).strip()

        direct_pairs = [
            ("food", "drink"),
            ("food", "drinks"),
            ("snack", "drink"),
            ("snacks", "drinks"),
            ("beverage", "food"),
            ("beverages", "food"),
        ]
        if any(a in normalized and b in normalized for a, b in direct_pairs):
            return any(
                token in normalized
                for token in ("sort", "separate", "arrange", "organize", "check", "cleanup", "clean up", "tidy")
            )

        # Lifestyle-style phrasing for demo commands.
        party_context = any(
            token in normalized
            for token in ("party", "guests", "guest", "visitors", "prepare", "hosting")
        )
        table_context = any(
            token in normalized
            for token in ("table", "desk", "counter", "countertop")
        )
        cleanup_action = any(
            token in normalized
            for token in ("clean", "clean up", "cleanup", "tidy", "organize", "arrange", "set up", "setup")
        )
        assist_phrase = any(
            token in normalized
            for token in ("help me", "please help", "can you help", "could you help")
        )

        if (party_context and table_context and cleanup_action) or (assist_phrase and table_context and cleanup_action):
            return True

        return False

    def _start_workflow(self, chat_id: int | str, reply_to_message_id: int | None) -> None:
        with self._state_lock:
            if chat_id in self._active_chats:
                self._telegram.send_message(chat_id, "I am already running this sorting workflow.", reply_to_message_id)
                return
            self._active_chats.add(chat_id)

        worker = threading.Thread(
            target=self._run_workflow,
            args=(chat_id, reply_to_message_id),
            daemon=True,
        )
        worker.start()

    def _run_workflow(self, chat_id: int | str, reply_to_message_id: int | None) -> None:
        try:
            blackboard = Blackboard()
            blackboard["node"] = self
            blackboard["chat_id"] = chat_id
            blackboard["reply_to_message_id"] = reply_to_message_id
            blackboard["last_error"] = ""
            blackboard["pick_target_tf"] = ""

            sm = self._build_state_machine()
            outcome = sm(blackboard)
            if outcome == WORKFLOW_DONE:
                self._log_state(chat_id, WorkflowState.COMPLETED, "done")
                return
            try:
                error_text = str(blackboard["last_error"]).strip()
            except Exception:
                error_text = ""
            if not error_text:
                error_text = "Workflow failed."
            self._log_state(chat_id, WorkflowState.FAILED, error_text)
            self._telegram.send_message(chat_id, error_text, reply_to_message_id)
        except Exception as exc:
            self._log_state(chat_id, WorkflowState.FAILED, str(exc))
            try:
                self._telegram.send_message(chat_id, f"Workflow failed: {exc}", reply_to_message_id)
            except Exception:
                pass
        finally:
            with self._state_lock:
                self._active_chats.discard(chat_id)

    def _build_state_machine(self) -> StateMachine:
        sm = StateMachine(outcomes=[WORKFLOW_DONE, WORKFLOW_FAILED])
        sm.add_state(
            "ACK_AND_SPEAK",
            self._retry_state("ACK_AND_SPEAK", AckAndSpeakState()),
            transitions={STEP_DONE: "MOVE_ARM_CENTER", ABORT: WORKFLOW_FAILED},
        )
        sm.add_state(
            "MOVE_ARM_CENTER",
            self._retry_state("MOVE_ARM_CENTER", MoveArmCenterState()),
            transitions={STEP_DONE: "NAVIGATE_TABLE", ABORT: WORKFLOW_FAILED},
        )
        sm.add_state(
            "NAVIGATE_TABLE",
            self._retry_state("NAVIGATE_TABLE", NavigateTableState()),
            transitions={STEP_DONE: "ANALYZE_FOOD_DRINK", ABORT: WORKFLOW_FAILED},
        )
        sm.add_state(
            "ANALYZE_FOOD_DRINK",
            self._retry_state("ANALYZE_FOOD_DRINK", AnalyzeFoodDrinkState()),
            transitions={STEP_DONE: "ANNOUNCE_RESULT", ABORT: WORKFLOW_FAILED},
        )
        sm.add_state(
            "ANNOUNCE_RESULT",
            self._retry_state("ANNOUNCE_RESULT", AnnounceResultState()),
            transitions={STEP_DONE: "DETECT_COFFEE", ABORT: WORKFLOW_FAILED},
        )
        sm.add_state(
            "DETECT_COFFEE",
            self._retry_state("DETECT_COFFEE", DetectPromptState()),
            transitions={STEP_DONE: "PICK_COFFEE", ABORT: WORKFLOW_FAILED},
        )
        sm.add_state(
            "PICK_COFFEE",
            self._retry_state("PICK_COFFEE", PickCoffeeState()),
            transitions={STEP_DONE: "PLACE_COFFEE_RIGHT", ABORT: WORKFLOW_FAILED},
        )
        sm.add_state(
            "PLACE_COFFEE_RIGHT",
            self._retry_state("PLACE_COFFEE_RIGHT", PlaceCoffeeRightState()),
            transitions={STEP_DONE: "DETECT_JUICE", ABORT: WORKFLOW_FAILED},
        )
        sm.add_state(
            "DETECT_JUICE",
            self._retry_state("DETECT_JUICE", DetectPromptState(prompt_text=self.yoloe_juice_prompt_text)),
            transitions={STEP_DONE: "PICK_JUICE", ABORT: WORKFLOW_FAILED},
        )
        sm.add_state(
            "PICK_JUICE",
            self._retry_state("PICK_JUICE", PickCoffeeState()),
            transitions={STEP_DONE: "PLACE_JUICE_RIGHT", ABORT: WORKFLOW_FAILED},
        )
        sm.add_state(
            "PLACE_JUICE_RIGHT",
            self._retry_state("PLACE_JUICE_RIGHT", PlaceCoffeeRightState()),
            transitions={STEP_DONE: "DETECT_CUP_NOODLE", ABORT: WORKFLOW_FAILED},
        )
        sm.add_state(
            "DETECT_CUP_NOODLE",
            self._retry_state("DETECT_CUP_NOODLE", DetectPromptState(prompt_text=self.yoloe_cup_noodle_prompt_text)),
            transitions={STEP_DONE: "PICK_CUP_NOODLE", ABORT: WORKFLOW_FAILED},
        )
        sm.add_state(
            "PICK_CUP_NOODLE",
            self._retry_state("PICK_CUP_NOODLE", PickCoffeeState()),
            transitions={STEP_DONE: "PLACE_CUP_NOODLE_LEFT", ABORT: WORKFLOW_FAILED},
        )
        sm.add_state(
            "PLACE_CUP_NOODLE_LEFT",
            self._retry_state("PLACE_CUP_NOODLE_LEFT", PlaceCupNoodleLeftState()),
            transitions={STEP_DONE: "DETECT_COOKIES", ABORT: WORKFLOW_FAILED},
        )
        sm.add_state(
            "DETECT_COOKIES",
            self._retry_state("DETECT_COOKIES", DetectPromptState(prompt_text=self.yoloe_cookies_prompt_text)),
            transitions={STEP_DONE: "PICK_COOKIES", ABORT: WORKFLOW_FAILED},
        )
        sm.add_state(
            "PICK_COOKIES",
            self._retry_state("PICK_COOKIES", PickCoffeeState()),
            transitions={STEP_DONE: "PLACE_COOKIES_LEFT", ABORT: WORKFLOW_FAILED},
        )
        sm.add_state(
            "PLACE_COOKIES_LEFT",
            self._retry_state("PLACE_COOKIES_LEFT", PlaceCupNoodleLeftState()),
            transitions={STEP_DONE: WORKFLOW_DONE, ABORT: WORKFLOW_FAILED},
        )
        return sm

    def _retry_state(self, state_name: str, state: State) -> RetryState:
        return RetryState(state_name, state, retry_delay_sec=self.state_retry_delay_sec)

    def _call_food_drink_sort(self, camera_name: str, confidence_threshold: float):
        if not self._food_drink_client.wait_for_service(timeout_sec=self.service_wait_timeout_sec):
            raise RuntimeError(f"Sort service not available: {self.food_drink_sort_service}")
        request = FoodDrinkSort.Request()
        request.camera_name = camera_name
        request.confidence_threshold = float(confidence_threshold)
        future = self._food_drink_client.call_async(request)
        return self._wait_for_future(future, self.service_timeout_sec)

    def _call_yoloe_detect(self, prompt_text: str, camera_name: str, save_image: bool) -> tuple[bool, str, str]:
        if self.detect_retry_warmup_sec > 0.0:
            time.sleep(self.detect_retry_warmup_sec)
        if not self._yoloe_client.wait_for_service(timeout_sec=self.service_wait_timeout_sec):
            return False, f"YOLO detect service not available: {self.yoloe_detect_service}", ""
        request = DetectObjectPrompt.Request()
        request.prompt_text = str(prompt_text).strip()
        request.save_image = bool(save_image)
        request.camera_name = str(camera_name).strip() or self.default_camera_name
        response = self._wait_for_future(self._yoloe_client.call_async(request), self.yoloe_timeout_sec)
        if response is None:
            return False, "YOLO detect service timed out.", ""
        target_tf = ""
        for frame in list(getattr(response, "tf_child_frames", []) or []):
            frame_text = str(frame).strip()
            if frame_text:
                target_tf = frame_text
                break
        if not bool(response.success):
            return False, str(response.message).strip() or "YOLO detect service failed.", target_tf
        return True, str(response.message).strip() or "ok", target_tf

    def _move_arm_pose(self, pose_name: str) -> tuple[bool, str]:
        if self.demo_mode:
            self.get_logger().info(f"Demo mode: skipping arm pose call '{pose_name}'.")
            return True, "demo arm success"
        if not self._arm_pose_client.wait_for_service(timeout_sec=self.service_wait_timeout_sec):
            return False, f"Arm pose service is not available: {self.arm_pose_service}"
        request = ArmPose.Request()
        request.pose_name = pose_name
        response = self._wait_for_future(self._arm_pose_client.call_async(request), self.arm_pose_timeout_sec)
        if response is None:
            return False, f"Timed out waiting for arm pose '{pose_name}'."
        if not bool(response.success):
            return False, str(response.message).strip() or f"Arm pose '{pose_name}' failed."
        return True, "ok"

    def _navigate_to_location(self, location: str) -> tuple[bool, str]:
        if self.demo_mode:
            time.sleep(self.demo_navigation_delay_sec)
            self.get_logger().info(f"Demo mode: skipping Nav2 goal send for '{location}'.")
            return True, "demo navigation success"
        if self.sim_mode:
            time.sleep(self.demo_navigation_delay_sec)
            self.get_logger().info(f"Sim mode: skipping Nav2 goal send for '{location}'.")
            return True, "sim navigation success"

        pose_config = self._location_pose_for(location)
        if pose_config is None:
            return False, f"No configured navigation pose for '{location}'."
        if not self._navigate_client.wait_for_server(timeout_sec=self.navigation_server_wait_timeout_sec):
            return False, f"NavigateToPose action '{self.navigation_action_name}' is not available."

        goal = NavigateToPose.Goal()
        goal.pose = self._create_pose_stamped_from_config(pose_config)
        goal_handle = self._wait_for_future(
            self._navigate_client.send_goal_async(goal),
            self.navigation_send_goal_timeout_sec,
        )
        if goal_handle is None or not goal_handle.accepted:
            return False, f"Navigation goal to '{location}' was rejected."

        result_wrap = self._wait_for_future(goal_handle.get_result_async(), self.navigation_timeout_sec)
        if result_wrap is None:
            return False, f"Timed out waiting for navigation result to '{location}'."
        status = int(getattr(result_wrap, "status", GoalStatus.STATUS_UNKNOWN))
        result = getattr(result_wrap, "result", None)
        error_code = int(getattr(result, "error_code", 0)) if result is not None else 0
        if status == GoalStatus.STATUS_SUCCEEDED and error_code == 0:
            return True, "ok"
        return False, f"Navigation to '{location}' failed with status={status} error_code={error_code}."

    def _speak_text(self, text: str) -> tuple[bool, str]:
        cleaned = str(text).strip()
        if not cleaned:
            return True, "Nothing to speak."
        if not self._speak_action_client.wait_for_server(timeout_sec=self.speak_action_wait_timeout_sec):
            return False, f"Speak action server '{self.speak_action_name}' not ready."
        goal = SpeakText.Goal()
        goal.text = cleaned
        goal_handle = self._wait_for_future(
            self._speak_action_client.send_goal_async(goal),
            self.speak_timeout_sec,
        )
        if goal_handle is None or not goal_handle.accepted:
            return False, "SpeakText goal rejected."
        result_wrap = self._wait_for_future(goal_handle.get_result_async(), self.speak_timeout_sec)
        if result_wrap is None:
            return False, "SpeakText result timeout."
        result = result_wrap.result
        return bool(result.success), str(result.message)

    def _pick_object(self, target_tf: str) -> tuple[bool, str]:
        if self.demo_mode:
            self.get_logger().info(f"Demo mode: skipping pick action for target_tf='{target_tf}'.")
            return True, "demo pick success"
        target_tf_clean = str(target_tf).strip()
        if not target_tf_clean:
            return False, "No target TF from YOLO detection."
        if not self._pick_action_client.wait_for_server(timeout_sec=self.pick_action_wait_timeout_sec):
            return False, f"Pick action server '{self.pick_action_name}' not ready."
        goal = Pick.Goal()
        goal.target_tf = target_tf_clean
        goal_handle = self._wait_for_future(
            self._pick_action_client.send_goal_async(goal),
            self.pick_timeout_sec,
        )
        if goal_handle is None or not goal_handle.accepted:
            return False, f"Pick action goal rejected for target_tf='{target_tf_clean}'."
        result_wrap = self._wait_for_future(goal_handle.get_result_async(), self.pick_timeout_sec)
        if result_wrap is None:
            return False, f"Pick action result timeout for target_tf='{target_tf_clean}'."
        status_code = int(getattr(result_wrap, "status", GoalStatus.STATUS_UNKNOWN))
        result = getattr(result_wrap, "result", None)
        if (
            status_code == GoalStatus.STATUS_SUCCEEDED
            and result is not None
            and bool(getattr(result, "success", False))
        ):
            return True, str(getattr(result, "message", "")).strip() or "ok"
        error_msg = str(getattr(result, "message", "")).strip() if result is not None else ""
        if error_msg:
            return False, error_msg
        return False, f"Pick action failed with status_code={status_code} target_tf='{target_tf_clean}'."

    def _call_open_gripper(self, command: str) -> tuple[bool, str]:
        if self.demo_mode:
            self.get_logger().info(f"Demo mode: skipping gripper command '{command}'.")
            return True, "demo gripper success"
        if not self._open_gripper_client.wait_for_service(timeout_sec=self.service_wait_timeout_sec):
            return False, f"Gripper service is not available: {self.open_gripper_service}"
        request = GripperCommand.Request()
        request.command = str(command).strip()
        response = self._wait_for_future(
            self._open_gripper_client.call_async(request),
            self.gripper_command_timeout_sec,
        )
        if response is None:
            return False, f"Timed out waiting for gripper command '{command}'."
        if not bool(response.success):
            return False, str(response.message).strip() or f"Gripper command '{command}' failed."
        return True, "ok"

    def _place_coffee_sequence(self) -> tuple[bool, str]:
        sequence = [
            ("sleep", "1"),
            ("arm_pose", "center"),
            ("sleep", "1"),
            ("arm_pose", "right"),
            ("sleep", "1"),
            ("gripper", "open"),
            ("sleep", "1"),
            ("arm_pose", "center"),
        ]
        for action, value in sequence:
            if action == "sleep":
                time.sleep(float(value))
                continue
            if action == "arm_pose":
                ok, message = self._move_arm_pose(value)
                if not ok:
                    return False, message
                continue
            if action == "gripper":
                ok, message = self._call_open_gripper(value)
                if not ok:
                    return False, message
                continue
        return True, "ok"

    def _place_left_sequence(self) -> tuple[bool, str]:
        sequence = [
            ("sleep", "1"),
            ("arm_pose", "center"),
            ("sleep", "1"),
            ("arm_pose", "left"),
            ("sleep", "1"),
            ("gripper", "open"),
            ("sleep", "1"),
            ("arm_pose", "center"),
        ]
        for action, value in sequence:
            if action == "sleep":
                time.sleep(float(value))
                continue
            if action == "arm_pose":
                ok, message = self._move_arm_pose(value)
                if not ok:
                    return False, message
                continue
            if action == "gripper":
                ok, message = self._call_open_gripper(value)
                if not ok:
                    return False, message
                continue
        return True, "ok"

    def _wait_for_future(self, future, timeout_sec: float):
        event = threading.Event()
        holder: dict[str, Any] = {}

        def _done(fut) -> None:
            holder["future"] = fut
            event.set()

        future.add_done_callback(_done)
        if not event.wait(timeout=max(0.1, timeout_sec)):
            return None
        fut = holder.get("future", future)
        exc = fut.exception()
        if exc is not None:
            raise RuntimeError(str(exc))
        return fut.result()

    def _log_state(self, chat_id: int | str, state: WorkflowState, detail: str = "") -> None:
        detail_text = f" | detail={detail}" if detail else ""
        self.get_logger().info(f"FoodDrink workflow [{chat_id}] -> {state.value}{detail_text}")

    @staticmethod
    def _message_id_or_none(value: Any) -> int | None:
        try:
            parsed = int(value)
        except Exception:
            return None
        return parsed if parsed > 0 else None


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = TelegramFoodDrinkStateMachineNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        executor.shutdown()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""YASMIN ROS 2 receptionist state machine."""

from __future__ import annotations

import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Any

import rclpy
import requests
from action_msgs.msg import GoalStatus
from coqui_tts_interfaces.action import SpeakText
from geometry_msgs.msg import PoseStamped, Quaternion
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.task import Future
from robot_arm_action.action import Point
from robot_arm_action.srv import ArmPose

import yasmin
from yasmin import Blackboard, State, StateMachine
from yasmin_ros import set_ros_loggers
from yasmin_ros.action_state import ActionState
from yasmin_ros.basic_outcomes import ABORT, SUCCEED, TIMEOUT
from yasmin_ros.service_state import ServiceState
from yasmin_ros.yasmin_node import YasminNode
from yasmin_viewer import YasminViewerPub

from vlm_interfaces.action import AskNameAndDrink, DescribeHuman
from yoloe_detection_interfaces.srv import DetectObjectPrompt


HOST = "Chris"
DEFAULT_YOLO_DETECTION_CAMERA_NAME = "camera0"
DEFAULT_NAVIGATION_ACTION = "/navigate_to_pose"
DEFAULT_ARM_POSE_SERVICE = "/arm_pose"
OPENAI_API_URL = os.environ.get("OPENAI_API_URL", "https://api.openai.com/v1/responses")
OPENAI_API_KEY_ENV = os.environ.get("OPENAI_API_KEY_ENV", "OPENAI_API_KEY")
# Change this line to switch the default OpenAI model for receptionist intro text.
OPENAI_MODEL = "gpt-5.5"
OPENAI_SPEECH_RESPONSE_FORMAT = {
    "type": "json_schema",
    "name": "receptionist_speech_text",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "speech_text": {"type": "string"},
        },
        "required": ["speech_text"],
    },
}

FINAL_OUTCOME = "task_finished"
RETRY = "retry"
DETECT_GUEST_OUTCOME = "guest_detected"
EMPTY_CHAIR_DETECTED = "empty_chair_detected"
NAME_DRINK_CAPTURED = "name_drink_captured"
CHARACTERISTICS_CAPTURED = "characteristics_captured"
DELAY_DONE = "delay_done"
INTRO_READY = "intro_ready"
GUEST_SLOT_PREPARED = "guest_slot_prepared"
NEXT_GUEST_READY = "next_guest_ready"
DUPLICATE_GUEST_NAME = "duplicate_guest_name"
NAVIGATION_DONE = "navigation_done"
POINTING_DONE = "pointing_done"
# Edit this global navigation target as needed for the receptionist task.
#field1
initial_pose: dict[str, Any] = {
    "frame_id": "map",
    "x": 1.34102,
    "y": -5.45996,
    "z": 0.0,
    "orientation": {
        "x": 0.0,
        "y": 0.0,
        "z": 0.709082,
        "w": 0.705126,
    },
}

initial_host:  dict[str, Any] = {
    "frame_id": "map",
    "x": -1.10943,
    "y": -4.52295,
    "z": 0.0,
    "orientation": {
        "x": 0.0,
        "y": 0.0,
        "z": -0.710914,
        "w": 0.703279,
    },
}

chair2: dict[str, Any] = {
    "frame_id": "map",
    "x": -1.59198,
    "y": -4.61911,
    "z": 0.0,
    "orientation": {
        "x": 0.0,
        "y": 0.0,
        "z": 1.0,
        "w": 0.00539813,
    },
}

approach_guest: dict[str, Any] = {
    "frame_id": "map",
    "x": 1.34102,
    "y": -4.500,
    "z": 0.0,
    "orientation": {
        "x": 0.0,
        "y": 0.0,
        "z": 0.709082,
        "w": 0.705126,
    },
}

#field2
# initial_pose: dict[str, Any] = {
#     "frame_id": "map",
#     "x": 1.34102,
#     "y": -1.99943,
#     "z": 0.0,
#     "orientation": {
#         "x": 0.0,
#         "y": 0.0,
#         "z": 0.709082,
#         "w": 0.705126,
#     },
# }

# initial_host:  dict[str, Any] = {
#     "frame_id": "map",
#     "x": -1.47946,
#     "y": -1.41678,
#     "z": 0.0,
#     "orientation": {
#         "x": 0.0,
#         "y": 0.0,
#         "z": -0.710914,
#         "w": 0.703279,
#     },
# }

# chair2: dict[str, Any] = {
#     "frame_id": "map",
#     "x": -0.892807,
#     "y": -0.888255,
#     "z": 0.0,
#     "orientation": {
#         "x": 0.0,
#         "y": 0.0,
#         "z": 1.0,
#         "w": 0.00539813,
#     },
# }

# Guest intro positions are fixed to the host position for this setup.
# Supported values are "initial_host" and "chair2".
GUEST1_INTRO_NAVIGATION_TARGET = "initial_host"

# Guest 2 also uses the fixed host intro position.
GUEST2_INTRO_NAVIGATION_TARGET = "initial_host"


def create_guest_memory() -> dict[str, Any]:
    return {
        "name": "",
        "drink": "",
        "characteristics": "",
        "characteristics_summary": "",
        "characteristics_list_text": "",
        "intro_to_host_text": "",
        "intro_guest1_to_guest2_text": "",
    }


def ensure_guest_memory(blackboard: Blackboard, guest_key: str) -> dict[str, Any]:
    if guest_key not in blackboard:
        blackboard[guest_key] = create_guest_memory()
    return blackboard[guest_key]


def get_current_guest_key(blackboard: Blackboard) -> str:
    return str(blackboard["current_guest_key"])


def get_current_guest_memory(blackboard: Blackboard) -> dict[str, Any]:
    return ensure_guest_memory(blackboard, get_current_guest_key(blackboard))


def _parse_json_dict(text: str) -> dict[str, Any]:
    if not text.strip():
        return {}
    try:
        parsed = json.loads(text)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _humanize_key(key: str) -> str:
    return key.replace("_", " ").replace(".", " ").strip()


def _flatten_characteristics(prefix: str, value: Any) -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        flattened: list[tuple[str, Any]] = []
        for child_key, child_value in value.items():
            child_prefix = f"{prefix}.{child_key}" if prefix else str(child_key)
            flattened.extend(_flatten_characteristics(child_prefix, child_value))
        return flattened

    if isinstance(value, list):
        flattened: list[tuple[str, Any]] = []
        for index, item in enumerate(value):
            item_prefix = f"{prefix}[{index}]"
            flattened.extend(_flatten_characteristics(item_prefix, item))
        return flattened

    return [(prefix, value)]


def _format_characteristic_entry(key: str, value: Any) -> str:
    normalized_key = key.split(".")[-1]
    if "[" in normalized_key:
        normalized_key = normalized_key.split("[", 1)[0]

    if isinstance(value, bool):
        if normalized_key == "wearing_glasses":
            return "wearing glasses" if value else "not wearing glasses"
        return _humanize_key(key) if value else ""

    text_value = str(value).strip()
    if not text_value:
        return ""

    if normalized_key == "gender":
        return text_value
    if normalized_key == "cloth_color":
        return f"{text_value} shirt"
    if normalized_key == "pant_color":
        return f"{text_value} pants"
    if normalized_key == "hair_color":
        return f"{text_value} hair"
    if normalized_key.endswith("_color"):
        base = normalized_key[: -len("_color")]
        return f"{text_value} {_humanize_key(base)}"
    return f"{_humanize_key(key)}: {text_value}"


def build_characteristics_list_text(raw_text: str, summary_text: str) -> str:
    parsed = _parse_json_dict(raw_text)
    entities = parsed.get("entities", parsed)
    if not isinstance(entities, dict):
        entities = {}

    characteristics: list[str] = []
    seen: set[str] = set()
    ignored_keys = {"human_present", "face_visible", "complete", "task", "reason"}

    for key, value in _flatten_characteristics("", entities):
        normalized_key = key.split(".")[-1]
        if "[" in normalized_key:
            normalized_key = normalized_key.split("[", 1)[0]
        if normalized_key in ignored_keys:
            continue
        if value in (None, "", [], {}):
            continue

        formatted = _format_characteristic_entry(key, value)
        if not formatted or formatted in seen:
            continue
        seen.add(formatted)
        characteristics.append(formatted)

    if characteristics:
        return ", ".join(characteristics)
    return summary_text.strip()


def split_characteristics_text(text: str, max_items: int | None = None) -> list[str]:
    items = [item.strip() for item in text.split(",") if item.strip()]
    return items if max_items is None else items[:max_items]


def join_spoken_list(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def build_guest_appearance_description(text: str, max_items: int = 5) -> str:
    items = split_characteristics_text(text, max_items=max_items)
    if not items:
        return ""

    gender = ""
    details: list[str] = []

    for item in items:
        normalized = item.lower()
        if normalized in {"male", "female"}:
            gender = normalized
            continue
        if normalized == "wearing glasses":
            details.append("glasses")
            continue
        details.append(item)

    details_text = join_spoken_list(details)
    if gender and details_text:
        return f"a {gender} guest with {details_text}"
    if gender:
        return f"a {gender} guest"
    if details_text:
        return f"a guest with {details_text}"
    return ""


def build_guest1_to_guest2_intro_text(blackboard: Blackboard) -> str:
    guest1 = ensure_guest_memory(blackboard, "guest1")
    guest2 = ensure_guest_memory(blackboard, "guest2")

    guest1_name = guest1["name"].strip() or "the first guest"
    guest2_name = guest2["name"].strip() or "guest"
    drink = guest1["drink"].strip()
    appearance_source = guest1["characteristics_list_text"] or guest1["characteristics_summary"]
    appearance_description = build_guest_appearance_description(appearance_source)

    if drink and appearance_description:
        return (
            f"{guest2_name}, this is {guest1_name}, whose favourite drink is {drink}, "
            f"and who is {appearance_description}."
        )
    if drink:
        return f"{guest2_name}, this is {guest1_name}, whose favourite drink is {drink}."
    if appearance_description:
        return f"{guest2_name}, this is {guest1_name}, who is {appearance_description}."
    return f"{guest2_name}, this is {guest1_name}."


def wait_for_future(_node, future: Future, timeout_sec: float) -> Any:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if future.done():
            return future.result()
        time.sleep(0.05)
    raise TimeoutError("Timed out waiting for ROS response.")


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


def normalize_intro_navigation_target(target: Any, default: str = "initial_host") -> str:
    text = str(target).strip().lower()
    if text in {"initial_host", "chair2"}:
        return text
    return default


def get_sim_mode() -> bool:
    if "RECEPTIONIST_SIM" in os.environ:
        return parse_bool(os.environ.get("RECEPTIONIST_SIM"), default=False)
    if "SIM" in os.environ:
        return parse_bool(os.environ.get("SIM"), default=False)
    return False


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


def get_env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def get_openai_api_key() -> str:
    api_key = os.environ.get(OPENAI_API_KEY_ENV, "").strip()
    if not api_key:
        raise RuntimeError(f"OpenAI API key is not set. Export {OPENAI_API_KEY_ENV} before starting this node.")
    return api_key


def extract_openai_output_text(payload: dict[str, Any]) -> str:
    direct_text = str(payload.get("output_text") or "").strip()
    if direct_text:
        return direct_text

    parts: list[str] = []
    for item in payload.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if not isinstance(content, dict):
                continue
            if content.get("type") in {"output_text", "text"}:
                text = str(content.get("text") or "")
                if text:
                    parts.append(text)
    return "\n".join(parts).strip()


def query_openai_speech_text(prompt: str, timeout_sec: float) -> str:
    payload = {
        "model": OPENAI_MODEL,
        "instructions": "Return only JSON that matches the requested schema.",
        "input": [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            }
        ],
        "text": {"format": OPENAI_SPEECH_RESPONSE_FORMAT},
        "max_output_tokens": max(64, get_env_int("OPENAI_MAX_OUTPUT_TOKENS", 256)),
        "store": False,
    }
    response = requests.post(
        OPENAI_API_URL,
        headers={
            "Authorization": f"Bearer {get_openai_api_key()}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=max(1.0, timeout_sec),
    )
    if not response.ok:
        raise RuntimeError(f"OpenAI request failed: HTTP {response.status_code} {response.text[:500]}")

    output_text = extract_openai_output_text(response.json())
    if not output_text:
        raise RuntimeError("OpenAI response did not contain output text.")
    parsed = json.loads(output_text)
    if not isinstance(parsed, dict):
        raise RuntimeError("OpenAI response JSON is not an object.")
    speech_text = str(parsed.get("speech_text") or "").strip()
    if not speech_text:
        raise RuntimeError("OpenAI response contained empty speech_text.")
    return speech_text


def get_yolo_detection_camera_name() -> str:
    camera_name = os.environ.get(
        "YOLO_DETECTION_CAMERA_NAME", DEFAULT_YOLO_DETECTION_CAMERA_NAME
    ).strip()
    return camera_name or DEFAULT_YOLO_DETECTION_CAMERA_NAME


def call_speak_text(node, action_name: str, text: str, timeout_sec: float = 30.0) -> None:
    if not text.strip():
        return

    client = ActionClient(node, SpeakText, action_name)
    if not client.wait_for_server(timeout_sec=5.0):
        raise RuntimeError(f"Speak action '{action_name}' is not available.")

    goal = SpeakText.Goal()
    goal.text = text
    goal_handle = wait_for_future(node, client.send_goal_async(goal), timeout_sec)
    if goal_handle is None or not goal_handle.accepted:
        raise RuntimeError("SpeakText goal was rejected.")

    result_wrapper = wait_for_future(node, goal_handle.get_result_async(), timeout_sec)
    result = result_wrapper.result
    if not result.success:
        raise RuntimeError(result.message or "SpeakText action failed.")


class SpeakState(State):
    def __init__(self, text_factory, outcome_on_success: str) -> None:
        super().__init__({outcome_on_success, ABORT})
        self._text_factory = text_factory
        self._outcome_on_success = outcome_on_success
        self._node = YasminNode.get_instance()
        self._action_name = "/coqui_tts/speak"

    def execute(self, blackboard: Blackboard) -> str:
        try:
            text = str(self._text_factory(blackboard)).strip()
            blackboard["last_spoken_text"] = text
            call_speak_text(self._node, self._action_name, text)
            return self._outcome_on_success
        except Exception as exc:
            blackboard["last_error"] = str(exc)
            return ABORT


class MockDelayState(State):
    def __init__(
        self,
        delay_key: str,
        outcome_on_success: str = DELAY_DONE,
        *,
        skip_in_sim: bool = False,
    ) -> None:
        super().__init__({outcome_on_success})
        self._delay_key = delay_key
        self._outcome_on_success = outcome_on_success
        self._skip_in_sim = skip_in_sim

    def execute(self, blackboard: Blackboard) -> str:
        if self._skip_in_sim and parse_bool(blackboard["sim"], default=False):
            return self._outcome_on_success
        time.sleep(float(blackboard[self._delay_key]))
        return self._outcome_on_success


class AnnounceEmptyChairFoundState(State):
    def __init__(self) -> None:
        super().__init__({"spoken", ABORT, TIMEOUT})
        self._node = YasminNode.get_instance()
        self._action_name = "/coqui_tts/speak"
        self._service_name = DEFAULT_ARM_POSE_SERVICE
        self._arm_pose_client = self._node.create_client(ArmPose, self._service_name)

    def execute(self, blackboard: Blackboard) -> str:
        guest_name = get_current_guest_memory(blackboard)["name"]
        text = f"Hi {guest_name}, you may have your seat here".strip()

        try:
            blackboard["last_spoken_text"] = text
            call_speak_text(self._node, self._action_name, text)

            time.sleep(float(blackboard["empty_chair_announce_arm_reset_delay_sec"]))

            if not self._arm_pose_client.wait_for_service(
                timeout_sec=float(blackboard["arm_pose_service_wait_timeout_sec"])
            ):
                blackboard["last_error"] = (
                    f"Service '{self._service_name}' is not available."
                )
                return TIMEOUT

            request = ArmPose.Request()
            request.pose_name = str(blackboard["arm_zero_pose_name"]).strip() or "zero"

            response = wait_for_future(
                self._node,
                self._arm_pose_client.call_async(request),
                float(blackboard["arm_pose_call_timeout_sec"]),
            )
            if response is None:
                blackboard["last_error"] = "Arm pose service returned no response."
                return ABORT
            if not response.success:
                blackboard["last_error"] = (
                    str(response.message).strip() or "Failed to move the arm to the zero pose."
                )
                return ABORT

            time.sleep(float(blackboard["empty_chair_announce_complete_delay_sec"]))
            blackboard["last_error"] = ""
            return "spoken"
        except TimeoutError as exc:
            blackboard["last_error"] = str(exc)
            return TIMEOUT
        except Exception as exc:
            blackboard["last_error"] = str(exc)
            return ABORT


class NavigateToInitialPoseState(State):
    def __init__(self) -> None:
        super().__init__({NAVIGATION_DONE, ABORT, TIMEOUT})
        self._node = YasminNode.get_instance()
        self._action_name = DEFAULT_NAVIGATION_ACTION
        self._client = ActionClient(self._node, NavigateToPose, self._action_name)

    def execute(self, blackboard: Blackboard) -> str:
        if parse_bool(blackboard["sim"], default=False):
            blackboard["navigation_status"] = GoalStatus.STATUS_SUCCEEDED
            blackboard["navigation_error_code"] = 0
            blackboard["last_error"] = ""
            return NAVIGATION_DONE

        if not self._client.wait_for_server(timeout_sec=float(blackboard["navigation_server_wait_timeout_sec"])):
            blackboard["last_error"] = (
                f"NavigateToPose action '{self._action_name}' is not available."
            )
            return TIMEOUT

        goal = NavigateToPose.Goal()
        goal.pose = create_pose_stamped(blackboard["initial_pose"])

        try:
            goal_handle = wait_for_future(
                self._node,
                self._client.send_goal_async(goal),
                float(blackboard["navigation_send_goal_timeout_sec"]),
            )
            if goal_handle is None or not goal_handle.accepted:
                blackboard["last_error"] = "Receptionist initial navigation goal was rejected."
                return ABORT

            result_wrapper = wait_for_future(
                self._node,
                goal_handle.get_result_async(),
                float(blackboard["navigation_timeout_sec"]),
            )
            result = None if result_wrapper is None else result_wrapper.result
            status = None if result_wrapper is None else int(result_wrapper.status)
            error_code = 0 if result is None else int(getattr(result, "error_code", 0))

            blackboard["navigation_status"] = -1 if status is None else status
            blackboard["navigation_error_code"] = error_code

            if status == GoalStatus.STATUS_SUCCEEDED and error_code == 0:
                blackboard["last_navigation_target"] = "initial_pose"
                blackboard["last_error"] = ""
                return NAVIGATION_DONE

            blackboard["last_error"] = (
                f"Receptionist initial navigation failed with status={status} error_code={error_code}."
            )
            return ABORT
        except TimeoutError as exc:
            blackboard["last_error"] = str(exc)
            return TIMEOUT
        except Exception as exc:
            blackboard["last_error"] = str(exc)
            return ABORT


class NavigateAndPrepareIntroState(State):
    def __init__(self, prompt_factory, result_field: str, destination_key_field: str) -> None:
        super().__init__({INTRO_READY, ABORT, TIMEOUT})
        self._prompt_factory = prompt_factory
        self._result_field = result_field
        self._destination_key_field = destination_key_field
        self._node = YasminNode.get_instance()
        self._action_name = DEFAULT_NAVIGATION_ACTION
        self._nav_client = ActionClient(self._node, NavigateToPose, self._action_name)
        self._openai_executor = ThreadPoolExecutor(max_workers=1)

    def execute(self, blackboard: Blackboard) -> str:
        if self._destination_key_field not in blackboard:
            blackboard["last_error"] = (
                f"Blackboard field '{self._destination_key_field}' is not defined."
            )
            return ABORT

        destination_key = str(blackboard[self._destination_key_field]).strip()
        if not destination_key:
            blackboard["last_error"] = (
                f"Blackboard field '{self._destination_key_field}' does not contain a destination key."
            )
            return ABORT
        if destination_key not in blackboard:
            blackboard["last_error"] = (
                f"Navigation destination '{destination_key}' is not defined on the blackboard."
            )
            return ABORT

        openai_timeout_sec = float(blackboard["vlm_intro_timeout_sec"])
        query_wait_deadline = time.monotonic() + openai_timeout_sec
        future = self._openai_executor.submit(
            query_openai_speech_text,
            self._prompt_factory(blackboard),
            openai_timeout_sec,
        )

        if not parse_bool(blackboard["sim"], default=False):
            if not self._nav_client.wait_for_server(
                timeout_sec=float(blackboard["navigation_server_wait_timeout_sec"])
            ):
                blackboard["last_error"] = (
                    f"NavigateToPose action '{self._action_name}' is not available."
                )
                return TIMEOUT

            goal = NavigateToPose.Goal()
            goal.pose = create_pose_stamped(blackboard[destination_key])

            try:
                goal_handle = wait_for_future(
                    self._node,
                    self._nav_client.send_goal_async(goal),
                    float(blackboard["navigation_send_goal_timeout_sec"]),
                )
                if goal_handle is None or not goal_handle.accepted:
                    blackboard["last_error"] = (
                        f"Receptionist navigation goal to '{destination_key}' was rejected."
                    )
                    return ABORT

                result_wrapper = wait_for_future(
                    self._node,
                    goal_handle.get_result_async(),
                    float(blackboard["navigation_timeout_sec"]),
                )
                result = None if result_wrapper is None else result_wrapper.result
                status = None if result_wrapper is None else int(result_wrapper.status)
                error_code = 0 if result is None else int(getattr(result, "error_code", 0))

                blackboard["navigation_status"] = -1 if status is None else status
                blackboard["navigation_error_code"] = error_code

                if status != GoalStatus.STATUS_SUCCEEDED or error_code != 0:
                    blackboard["last_error"] = (
                        f"Receptionist navigation to '{destination_key}' failed "
                        f"with status={status} error_code={error_code}."
                    )
                    return ABORT
            except TimeoutError as exc:
                blackboard["last_error"] = str(exc)
                return TIMEOUT
            except Exception as exc:
                blackboard["last_error"] = str(exc)
                return ABORT
        else:
            blackboard["navigation_status"] = GoalStatus.STATUS_SUCCEEDED
            blackboard["navigation_error_code"] = 0
        blackboard["last_navigation_target"] = destination_key

        try:
            remaining_timeout_sec = max(0.1, query_wait_deadline - time.monotonic())
            blackboard[self._result_field] = future.result(timeout=remaining_timeout_sec)
            blackboard["last_error"] = ""
            return INTRO_READY
        except FutureTimeoutError:
            blackboard["last_error"] = "Timed out waiting for OpenAI speech generation."
            return TIMEOUT
        except Exception as exc:
            blackboard["last_error"] = f"OpenAI speech generation failed: {exc}"
            return ABORT


class EnsureChair2NavigationState(State):
    def __init__(self) -> None:
        super().__init__({NAVIGATION_DONE, ABORT, TIMEOUT})
        self._node = YasminNode.get_instance()
        self._action_name = DEFAULT_NAVIGATION_ACTION
        self._client = ActionClient(self._node, NavigateToPose, self._action_name)

    def execute(self, blackboard: Blackboard) -> str:
        last_navigation_target = ""
        if "last_navigation_target" in blackboard:
            last_navigation_target = str(blackboard["last_navigation_target"]).strip()

        if last_navigation_target == "chair2":
            blackboard["last_error"] = ""
            return NAVIGATION_DONE

        if "chair2" not in blackboard:
            blackboard["last_error"] = "Navigation destination 'chair2' is not defined on the blackboard."
            return ABORT

        if parse_bool(blackboard["sim"], default=False):
            blackboard["navigation_status"] = GoalStatus.STATUS_SUCCEEDED
            blackboard["navigation_error_code"] = 0
            blackboard["last_navigation_target"] = "chair2"
            blackboard["last_error"] = ""
            return NAVIGATION_DONE

        if not self._client.wait_for_server(timeout_sec=float(blackboard["navigation_server_wait_timeout_sec"])):
            blackboard["last_error"] = (
                f"NavigateToPose action '{self._action_name}' is not available."
            )
            return TIMEOUT

        goal = NavigateToPose.Goal()
        goal.pose = create_pose_stamped(blackboard["chair2"])

        try:
            goal_handle = wait_for_future(
                self._node,
                self._client.send_goal_async(goal),
                float(blackboard["navigation_send_goal_timeout_sec"]),
            )
            if goal_handle is None or not goal_handle.accepted:
                blackboard["last_error"] = "Receptionist navigation goal to 'chair2' was rejected."
                return ABORT

            result_wrapper = wait_for_future(
                self._node,
                goal_handle.get_result_async(),
                float(blackboard["navigation_timeout_sec"]),
            )
            result = None if result_wrapper is None else result_wrapper.result
            status = None if result_wrapper is None else int(result_wrapper.status)
            error_code = 0 if result is None else int(getattr(result, "error_code", 0))

            blackboard["navigation_status"] = -1 if status is None else status
            blackboard["navigation_error_code"] = error_code

            if status == GoalStatus.STATUS_SUCCEEDED and error_code == 0:
                blackboard["last_navigation_target"] = "chair2"
                blackboard["last_error"] = ""
                return NAVIGATION_DONE

            blackboard["last_error"] = (
                f"Receptionist navigation to 'chair2' failed "
                f"with status={status} error_code={error_code}."
            )
            return ABORT
        except TimeoutError as exc:
            blackboard["last_error"] = str(exc)
            return TIMEOUT
        except Exception as exc:
            blackboard["last_error"] = str(exc)
            return ABORT


class PrepareGuestSlotState(State):
    def __init__(self, guest_key: str, guest_number: int) -> None:
        super().__init__({GUEST_SLOT_PREPARED})
        self._guest_key = guest_key
        self._guest_number = guest_number

    def execute(self, blackboard: Blackboard) -> str:
        blackboard["current_guest_key"] = self._guest_key
        blackboard["current_guest_number"] = self._guest_number
        blackboard["detect_guest_attempt_count"] = 0
        ensure_guest_memory(blackboard, self._guest_key)
        return GUEST_SLOT_PREPARED


class AdvanceToGuest2State(State):
    def __init__(self) -> None:
        super().__init__({NEXT_GUEST_READY})

    def execute(self, blackboard: Blackboard) -> str:
        blackboard["current_guest_key"] = "guest2"
        blackboard["current_guest_number"] = 2
        blackboard["detect_guest_attempt_count"] = 0
        ensure_guest_memory(blackboard, "guest2")
        return NEXT_GUEST_READY


class VlmSpeechState(State):
    def __init__(self, prompt_factory, result_field: str, delay_key: str | None = None) -> None:
        super().__init__({INTRO_READY, ABORT, TIMEOUT})
        self._prompt_factory = prompt_factory
        self._result_field = result_field
        self._delay_key = delay_key

    def execute(self, blackboard: Blackboard) -> str:
        if self._delay_key is not None and not parse_bool(blackboard["sim"], default=False):
            time.sleep(float(blackboard[self._delay_key]))

        try:
            blackboard[self._result_field] = query_openai_speech_text(
                self._prompt_factory(blackboard),
                float(blackboard["vlm_intro_timeout_sec"]),
            )
            blackboard["last_error"] = ""
            return INTRO_READY
        except requests.Timeout:
            blackboard["last_error"] = "Timed out waiting for OpenAI speech generation."
            return TIMEOUT
        except Exception as exc:
            blackboard["last_error"] = f"OpenAI speech generation failed: {exc}"
            return ABORT


class PrepareGuest1ToGuest2IntroState(State):
    def __init__(self) -> None:
        super().__init__({INTRO_READY})

    def execute(self, blackboard: Blackboard) -> str:
        blackboard["guest1_to_guest2_intro_text"] = build_guest1_to_guest2_intro_text(blackboard)
        return INTRO_READY


class StoreAskNameAndDrinkState(ActionState):
    def __init__(self) -> None:
        super().__init__(
            AskNameAndDrink,
            "/ask_name_and_drink",
            self.create_goal_handler,
            {NAME_DRINK_CAPTURED, DUPLICATE_GUEST_NAME},
            self.result_handler,
            wait_timeout=5.0,
            response_timeout=180.0,
            maximum_retry=2,
        )

    def create_goal_handler(self, blackboard: Blackboard) -> AskNameAndDrink.Goal:
        goal = AskNameAndDrink.Goal()
        goal.think = False
        goal.reset_session = True
        goal.max_attempts = 3
        return goal

    def result_handler(self, blackboard: Blackboard, result: AskNameAndDrink.Result) -> str:
        if not result.success or not result.completed:
            blackboard["last_error"] = result.message or "Failed to capture guest name and drink."
            return ABORT

        guest = get_current_guest_memory(blackboard)
        guest["name"] = result.name.strip()
        guest["drink"] = result.drink.strip()
        blackboard[get_current_guest_key(blackboard)] = guest

        if get_current_guest_key(blackboard) == "guest2":
            guest1 = ensure_guest_memory(blackboard, "guest1")
            if guest1["name"].strip() and guest1["name"].strip().lower() == guest["name"].strip().lower():
                blackboard["duplicate_guest_name"] = guest["name"]
                blackboard["last_error"] = ""
                return DUPLICATE_GUEST_NAME

        return NAME_DRINK_CAPTURED


class StoreDescribeHumanState(ActionState):
    def __init__(self) -> None:
        super().__init__(
            DescribeHuman,
            "/describe_human",
            self.create_goal_handler,
            {CHARACTERISTICS_CAPTURED},
            self.result_handler,
            wait_timeout=5.0,
            response_timeout=120.0,
            maximum_retry=2,
        )

    def create_goal_handler(self, blackboard: Blackboard) -> DescribeHuman.Goal:
        return DescribeHuman.Goal()

    def result_handler(self, blackboard: Blackboard, result: DescribeHuman.Result) -> str:
        if not result.success:
            blackboard["last_error"] = result.message or "DescribeHuman action failed."
            return ABORT

        guest = get_current_guest_memory(blackboard)
        guest["characteristics"] = (result.data_text or "").strip() or (result.speech_text or "").strip()
        guest["characteristics_summary"] = (result.speech_text or "").strip()
        guest["characteristics_list_text"] = build_characteristics_list_text(
            guest["characteristics"],
            guest["characteristics_summary"],
        )
        blackboard[get_current_guest_key(blackboard)] = guest
        return CHARACTERISTICS_CAPTURED


class PointObjectState(ActionState):
    def __init__(self, target_frame: str) -> None:
        super().__init__(
            Point,
            "/point_object",
            self.create_goal_handler,
            {POINTING_DONE},
            self.result_handler,
            wait_timeout=5.0,
            response_timeout=60.0,
            maximum_retry=2,
        )
        self._target_frame = target_frame

    def create_goal_handler(self, blackboard: Blackboard) -> Point.Goal:
        del blackboard
        goal = Point.Goal()
        goal.target_frame = self._target_frame
        return goal

    def result_handler(self, blackboard: Blackboard, result: Point.Result) -> str:
        if not result.success:
            blackboard["last_error"] = (
                result.message or f"Failed to point at '{self._target_frame}'."
            )
            return ABORT
        blackboard["last_error"] = ""
        return POINTING_DONE


class DetectGuestState(ServiceState):
    def __init__(self) -> None:
        super().__init__(
            DetectObjectPrompt,
            "/yoloe/detect_prompt",
            self.create_request_handler,
            {DETECT_GUEST_OUTCOME, RETRY},
            self.response_handler,
            wait_timeout=5.0,
            response_timeout=20.0,
            maximum_retry=2,
        )

    def create_request_handler(self, blackboard: Blackboard) -> DetectObjectPrompt.Request:
        request = DetectObjectPrompt.Request()
        request.prompt_text = "person"
        request.save_image = False
        request.camera_name = get_yolo_detection_camera_name()
        return request

    def response_handler(self, blackboard: Blackboard, response: DetectObjectPrompt.Response) -> str:
        max_distance_m = float(blackboard["detect_guest_max_distance_m"])
        blackboard["detect_guest_attempt_count"] = int(blackboard["detect_guest_attempt_count"]) + 1
        blackboard["detected_guest_pose"] = None
        blackboard["detected_guest_distance_m"] = 0.0

        if response is None or not response.success:
            blackboard["last_detect_message"] = (
                "" if response is None else response.message
            ) or "Detection service reported failure."
            return self._retry_or_abort(blackboard)

        for pose_stamped in response.poses_camera_link:
            position = pose_stamped.pose.position
            distance = math.sqrt(
                position.x * position.x + position.y * position.y + position.z * position.z
            )
            if distance <= max_distance_m:
                blackboard["detected_guest_pose"] = pose_stamped
                blackboard["detected_guest_distance_m"] = distance
                blackboard["last_detect_message"] = f"Detected guest within {distance:.2f}m."
                return DETECT_GUEST_OUTCOME

        blackboard["last_detect_message"] = (
            f"No human detected within {max_distance_m:.2f}m. "
            f"Frame detections={response.detections_in_frame}."
        )
        return self._retry_or_abort(blackboard)

    @staticmethod
    def _retry_or_abort(blackboard: Blackboard) -> str:
        max_attempts = int(blackboard["detect_guest_max_attempts"])
        if max_attempts <= 0:
            return RETRY
        if int(blackboard["detect_guest_attempt_count"]) < max_attempts:
            return RETRY
        blackboard["last_error"] = blackboard["last_detect_message"]
        return ABORT


class NavigateToDetectedGuestState(State):
    def __init__(self) -> None:
        super().__init__({NAVIGATION_DONE, ABORT, TIMEOUT})
        self._node = YasminNode.get_instance()
        self._action_name = DEFAULT_NAVIGATION_ACTION
        self._client = ActionClient(self._node, NavigateToPose, self._action_name)

    def execute(self, blackboard: Blackboard) -> str:
        if "approach_guest" not in blackboard:
            blackboard["last_error"] = "Approach guest pose is not available for navigation."
            return ABORT

        goal_pose = blackboard["approach_guest"]

        if parse_bool(blackboard["sim"], default=False):
            blackboard["navigation_status"] = GoalStatus.STATUS_SUCCEEDED
            blackboard["navigation_error_code"] = 0
            blackboard["last_navigation_target"] = "approach_guest"
            blackboard["last_error"] = ""
            return NAVIGATION_DONE

        if not self._client.wait_for_server(timeout_sec=float(blackboard["navigation_server_wait_timeout_sec"])):
            blackboard["last_error"] = (
                f"NavigateToPose action '{self._action_name}' is not available."
            )
            return TIMEOUT

        goal = NavigateToPose.Goal()
        goal.pose = create_pose_stamped(goal_pose)

        try:
            goal_handle = wait_for_future(
                self._node,
                self._client.send_goal_async(goal),
                float(blackboard["navigation_send_goal_timeout_sec"]),
            )
            if goal_handle is None or not goal_handle.accepted:
                blackboard["last_error"] = "Navigation goal to approach_guest was rejected."
                return ABORT

            result_wrapper = wait_for_future(
                self._node,
                goal_handle.get_result_async(),
                float(blackboard["navigation_timeout_sec"]),
            )
            result = None if result_wrapper is None else result_wrapper.result
            status = None if result_wrapper is None else int(result_wrapper.status)
            error_code = 0 if result is None else int(getattr(result, "error_code", 0))

            blackboard["navigation_status"] = -1 if status is None else status
            blackboard["navigation_error_code"] = error_code

            if status == GoalStatus.STATUS_SUCCEEDED and error_code == 0:
                blackboard["last_navigation_target"] = "approach_guest"
                blackboard["last_error"] = ""
                return NAVIGATION_DONE

            blackboard["last_error"] = (
                f"Navigation to approach_guest failed with status={status} error_code={error_code}."
            )
            return ABORT
        except TimeoutError as exc:
            blackboard["last_error"] = str(exc)
            return TIMEOUT
        except Exception as exc:
            blackboard["last_error"] = str(exc)
            return ABORT


class DetectEmptyChairState(State):
    def __init__(self) -> None:
        super().__init__({EMPTY_CHAIR_DETECTED, RETRY})
        self._node = YasminNode.get_instance()
        self._service_name = "/yoloe/detect_prompt"
        self._client = self._node.create_client(DetectObjectPrompt, self._service_name)

    def execute(self, blackboard: Blackboard) -> str:
        request = DetectObjectPrompt.Request()
        request.prompt_text = "a empty chair"
        request.save_image = True
        request.camera_name = get_yolo_detection_camera_name()

        try:
            if not self._client.wait_for_service(timeout_sec=5.0):
                blackboard["last_empty_chair_detect_message"] = (
                    f"Service '{self._service_name}' is not available."
                )
                return RETRY

            response = wait_for_future(self._node, self._client.call_async(request), 20.0)
            if response is None or not response.success:
                blackboard["last_empty_chair_detect_message"] = (
                    "" if response is None else response.message
                ) or "Empty chair detection service reported failure."
                return RETRY

            blackboard["last_empty_chair_detect_message"] = response.message
            blackboard["last_empty_chair_image_path"] = response.saved_image_path
            return EMPTY_CHAIR_DETECTED
        except Exception as exc:
            blackboard["last_empty_chair_detect_message"] = str(exc)
            return RETRY


def host_intro_prompt(blackboard: Blackboard) -> str:
    guest = get_current_guest_memory(blackboard)
    return (
        "Generate exactly one short spoken sentence to introduce a guest to the host. "
        f"The sentence must start exactly with: Hi {HOST}, "
        "Address the host directly in the sentence. "
        "Only include the guest name and favourite drink in the spoken sentence. "
        f"Host name: {HOST}. "
        f"Guest name: {guest['name']}. "
        f"Favourite drink: {guest['drink']}. "
        "Do not mention appearance or any extra facts."
    )

def build_state_machine() -> StateMachine:
    sm = StateMachine(outcomes=[FINAL_OUTCOME, ABORT])

    sm.add_state(
        "NAVIGATE_TO_INITIAL_POSE",
        NavigateToInitialPoseState(),
        transitions={
            NAVIGATION_DONE: "ANNOUNCE_READY_FOR_NEW_GUEST_AT_START",
            ABORT: "NAVIGATE_TO_INITIAL_POSE",
            TIMEOUT: "NAVIGATE_TO_INITIAL_POSE",
        },
    )
    sm.add_state(
        "ANNOUNCE_READY_FOR_NEW_GUEST_AT_START",
        SpeakState(lambda _: "I am ready to accept new guest", "spoken"),
        transitions={"spoken": "STARTUP_DELAY", ABORT: "ANNOUNCE_READY_FOR_NEW_GUEST_AT_START"},
    )
    sm.add_state(
        "STARTUP_DELAY",
        MockDelayState("startup_delay_sec"),
        transitions={DELAY_DONE: "PREPARE_GUEST1"},
    )
    sm.add_state(
        "PREPARE_GUEST1",
        PrepareGuestSlotState("guest1", 1),
        transitions={GUEST_SLOT_PREPARED: "DETECT_GUEST1"},
    )
    sm.add_state(
        "DETECT_GUEST1",
        DetectGuestState(),
        transitions={
            DETECT_GUEST_OUTCOME: "NAVIGATE_TO_DETECTED_GUEST1",
            RETRY: "DETECT_GUEST1",
            SUCCEED: "NAVIGATE_TO_DETECTED_GUEST1",
            ABORT: "DETECT_GUEST1",
            TIMEOUT: "DETECT_GUEST1",
        },
    )
    sm.add_state(
        "NAVIGATE_TO_DETECTED_GUEST1",
        NavigateToDetectedGuestState(),
        transitions={
            NAVIGATION_DONE: "GREET_GUEST1",
            ABORT: "NAVIGATE_TO_DETECTED_GUEST1",
            TIMEOUT: "NAVIGATE_TO_DETECTED_GUEST1",
        },
    )
    sm.add_state(
        "GREET_GUEST1",
        SpeakState(lambda _: f"Hello, Welcome to {HOST} house", "spoken"),
        transitions={"spoken": "ASK_NAME_AND_DRINK_GUEST1", ABORT: "GREET_GUEST1"},
    )
    sm.add_state(
        "ASK_NAME_AND_DRINK_GUEST1",
        StoreAskNameAndDrinkState(),
        transitions={
            NAME_DRINK_CAPTURED: "DESCRIBE_GUEST1",
            SUCCEED: "DESCRIBE_GUEST1",
            ABORT: "ASK_NAME_AND_DRINK_GUEST1",
            TIMEOUT: "ASK_NAME_AND_DRINK_GUEST1",
        },
    )
    sm.add_state(
        "DESCRIBE_GUEST1",
        StoreDescribeHumanState(),
        transitions={
            CHARACTERISTICS_CAPTURED: "ASK_FOLLOW_GUEST1",
            SUCCEED: "ASK_FOLLOW_GUEST1",
            ABORT: "DESCRIBE_GUEST1",
            TIMEOUT: "DESCRIBE_GUEST1",
        },
    )
    sm.add_state(
        "ASK_FOLLOW_GUEST1",
        SpeakState(lambda bb: f"Hi {get_current_guest_memory(bb)['name']}, please follow me", "spoken"),
        transitions={"spoken": "NAVIGATE_AND_PREPARE_INTRO_GUEST1", ABORT: "ASK_FOLLOW_GUEST1"},
    )
    sm.add_state(
        "NAVIGATE_AND_PREPARE_INTRO_GUEST1",
        NavigateAndPrepareIntroState(
            host_intro_prompt,
            "current_host_intro_text",
            "guest1_intro_navigation_target",
        ),
        transitions={
            INTRO_READY: "INTRODUCE_GUEST1_TO_HOST",
            ABORT: "NAVIGATE_AND_PREPARE_INTRO_GUEST1",
            TIMEOUT: "NAVIGATE_AND_PREPARE_INTRO_GUEST1",
        },
    )
    sm.add_state(
        "INTRODUCE_GUEST1_TO_HOST",
        SpeakState(lambda bb: str(bb["current_host_intro_text"]), "spoken"),
        transitions={
            "spoken": "ENSURE_CHAIR2_BEFORE_EMPTY_CHAIR_DETECTION_GUEST1",
            ABORT: "INTRODUCE_GUEST1_TO_HOST",
        },
    )
    sm.add_state(
        "ENSURE_CHAIR2_BEFORE_EMPTY_CHAIR_DETECTION_GUEST1",
        EnsureChair2NavigationState(),
        transitions={
            NAVIGATION_DONE: "ANNOUNCE_EMPTY_CHAIR_DETECTION_GUEST1",
            ABORT: "ENSURE_CHAIR2_BEFORE_EMPTY_CHAIR_DETECTION_GUEST1",
            TIMEOUT: "ENSURE_CHAIR2_BEFORE_EMPTY_CHAIR_DETECTION_GUEST1",
        },
    )
    sm.add_state(
        "ANNOUNCE_EMPTY_CHAIR_DETECTION_GUEST1",
        SpeakState(
            lambda bb: f"Hi {get_current_guest_memory(bb)['name']}, I will find you an empty seat",
            "spoken",
        ),
        transitions={"spoken": "DETECT_EMPTY_CHAIR_GUEST1", ABORT: "ANNOUNCE_EMPTY_CHAIR_DETECTION_GUEST1"},
    )
    sm.add_state(
        "DETECT_EMPTY_CHAIR_GUEST1",
        DetectEmptyChairState(),
        transitions={
            EMPTY_CHAIR_DETECTED: "POINT_EMPTY_CHAIR_GUEST1",
            RETRY: "EMPTY_CHAIR_DETECTION_RETRY_WARNING_GUEST1",
        },
    )
    sm.add_state(
        "POINT_EMPTY_CHAIR_GUEST1",
        PointObjectState("a_empty_chair_1"),
        transitions={
            POINTING_DONE: "ANNOUNCE_EMPTY_CHAIR_FOUND_GUEST1",
            ABORT: "POINT_EMPTY_CHAIR_GUEST1",
            TIMEOUT: "POINT_EMPTY_CHAIR_GUEST1",
        },
    )
    sm.add_state(
        "ANNOUNCE_EMPTY_CHAIR_FOUND_GUEST1",
        AnnounceEmptyChairFoundState(),
        transitions={
            "spoken": "NAVIGATE_BACK_TO_START",
            ABORT: "ANNOUNCE_EMPTY_CHAIR_FOUND_GUEST1",
            TIMEOUT: "ANNOUNCE_EMPTY_CHAIR_FOUND_GUEST1",
        },
    )
    sm.add_state(
        "EMPTY_CHAIR_DETECTION_RETRY_WARNING_GUEST1",
        SpeakState(lambda _: "Sorry, I cannot find any empty seat, let me try again", "spoken"),
        transitions={"spoken": "DETECT_EMPTY_CHAIR_GUEST1", ABORT: "EMPTY_CHAIR_DETECTION_RETRY_WARNING_GUEST1"},
    )
    sm.add_state(
        "NAVIGATE_BACK_TO_START",
        NavigateToInitialPoseState(),
        transitions={
            NAVIGATION_DONE: "ANNOUNCE_READY_FOR_NEW_GUEST_AFTER_GUEST1",
            ABORT: "NAVIGATE_BACK_TO_START",
            TIMEOUT: "NAVIGATE_BACK_TO_START",
        },
    )
    sm.add_state(
        "ANNOUNCE_READY_FOR_NEW_GUEST_AFTER_GUEST1",
        SpeakState(lambda _: "I am ready to accept new guest", "spoken"),
        transitions={"spoken": "PREPARE_GUEST2", ABORT: "ANNOUNCE_READY_FOR_NEW_GUEST_AFTER_GUEST1"},
    )
    sm.add_state(
        "PREPARE_GUEST2",
        AdvanceToGuest2State(),
        transitions={NEXT_GUEST_READY: "DETECT_GUEST2"},
    )
    sm.add_state(
        "DETECT_GUEST2",
        DetectGuestState(),
        transitions={
            DETECT_GUEST_OUTCOME: "NAVIGATE_TO_DETECTED_GUEST2",
            RETRY: "DETECT_GUEST2",
            SUCCEED: "NAVIGATE_TO_DETECTED_GUEST2",
            ABORT: "DETECT_GUEST2",
            TIMEOUT: "DETECT_GUEST2",
        },
    )
    sm.add_state(
        "NAVIGATE_TO_DETECTED_GUEST2",
        NavigateToDetectedGuestState(),
        transitions={
            NAVIGATION_DONE: "GREET_GUEST2",
            ABORT: "NAVIGATE_TO_DETECTED_GUEST2",
            TIMEOUT: "NAVIGATE_TO_DETECTED_GUEST2",
        },
    )
    sm.add_state(
        "GREET_GUEST2",
        SpeakState(lambda _: f"Hello, Welcome to {HOST} house", "spoken"),
        transitions={"spoken": "ASK_NAME_AND_DRINK_GUEST2", ABORT: "GREET_GUEST2"},
    )
    sm.add_state(
        "ASK_NAME_AND_DRINK_GUEST2",
        StoreAskNameAndDrinkState(),
        transitions={
            NAME_DRINK_CAPTURED: "DESCRIBE_GUEST2",
            DUPLICATE_GUEST_NAME: "DUPLICATE_GUEST2_WARNING",
            SUCCEED: "DESCRIBE_GUEST2",
            ABORT: "ASK_NAME_AND_DRINK_GUEST2",
            TIMEOUT: "ASK_NAME_AND_DRINK_GUEST2",
        },
    )
    sm.add_state(
        "DUPLICATE_GUEST2_WARNING",
        SpeakState(
            lambda bb: (
                f"I already registered {bb['duplicate_guest_name']}. "
                "I will wait for a different guest."
            ),
            "spoken",
        ),
        transitions={"spoken": "DETECT_GUEST2", ABORT: "DUPLICATE_GUEST2_WARNING"},
    )
    sm.add_state(
        "DESCRIBE_GUEST2",
        StoreDescribeHumanState(),
        transitions={
            CHARACTERISTICS_CAPTURED: "ASK_FOLLOW_GUEST2",
            SUCCEED: "ASK_FOLLOW_GUEST2",
            ABORT: "DESCRIBE_GUEST2",
            TIMEOUT: "DESCRIBE_GUEST2",
        },
    )
    sm.add_state(
        "ASK_FOLLOW_GUEST2",
        SpeakState(lambda bb: f"Hi {get_current_guest_memory(bb)['name']}, please follow me", "spoken"),
        transitions={"spoken": "NAVIGATE_AND_PREPARE_INTRO_GUEST2", ABORT: "ASK_FOLLOW_GUEST2"},
    )
    sm.add_state(
        "NAVIGATE_AND_PREPARE_INTRO_GUEST2",
        NavigateAndPrepareIntroState(
            host_intro_prompt,
            "current_host_intro_text",
            "guest2_intro_navigation_target",
        ),
        transitions={
            INTRO_READY: "INTRODUCE_GUEST2_TO_HOST",
            ABORT: "NAVIGATE_AND_PREPARE_INTRO_GUEST2",
            TIMEOUT: "NAVIGATE_AND_PREPARE_INTRO_GUEST2",
        },
    )
    sm.add_state(
        "INTRODUCE_GUEST2_TO_HOST",
        SpeakState(lambda bb: str(bb["current_host_intro_text"]), "spoken"),
        transitions={"spoken": "PREPARE_GUEST1_TO_GUEST2_INTRO", ABORT: "INTRODUCE_GUEST2_TO_HOST"},
    )
    sm.add_state(
        "PREPARE_GUEST1_TO_GUEST2_INTRO",
        PrepareGuest1ToGuest2IntroState(),
        transitions={INTRO_READY: "NAVIGATE_TO_CHAIR2_BEFORE_INTRODUCE_GUEST1_TO_GUEST2"},
    )
    sm.add_state(
        "NAVIGATE_TO_CHAIR2_BEFORE_INTRODUCE_GUEST1_TO_GUEST2",
        EnsureChair2NavigationState(),
        transitions={
            NAVIGATION_DONE: "INTRODUCE_GUEST1_TO_GUEST2",
            ABORT: "NAVIGATE_TO_CHAIR2_BEFORE_INTRODUCE_GUEST1_TO_GUEST2",
            TIMEOUT: "NAVIGATE_TO_CHAIR2_BEFORE_INTRODUCE_GUEST1_TO_GUEST2",
        },
    )
    sm.add_state(
        "INTRODUCE_GUEST1_TO_GUEST2",
        SpeakState(lambda bb: str(bb["guest1_to_guest2_intro_text"]), "spoken"),
        transitions={"spoken": "ANNOUNCE_EMPTY_CHAIR_DETECTION_GUEST2", ABORT: "INTRODUCE_GUEST1_TO_GUEST2"},
    )
    sm.add_state(
        "ANNOUNCE_EMPTY_CHAIR_DETECTION_GUEST2",
        SpeakState(
            lambda bb: f"Hi {get_current_guest_memory(bb)['name']}, I will find you an empty seat",
            "spoken",
        ),
        transitions={"spoken": "DETECT_EMPTY_CHAIR_GUEST2", ABORT: "ANNOUNCE_EMPTY_CHAIR_DETECTION_GUEST2"},
    )
    sm.add_state(
        "DETECT_EMPTY_CHAIR_GUEST2",
        DetectEmptyChairState(),
        transitions={
            EMPTY_CHAIR_DETECTED: "POINT_EMPTY_CHAIR_GUEST2",
            RETRY: "EMPTY_CHAIR_DETECTION_RETRY_WARNING_GUEST2",
        },
    )
    sm.add_state(
        "POINT_EMPTY_CHAIR_GUEST2",
        PointObjectState("a_empty_chair_1"),
        transitions={
            POINTING_DONE: "ANNOUNCE_EMPTY_CHAIR_FOUND_GUEST2",
            ABORT: "POINT_EMPTY_CHAIR_GUEST2",
            TIMEOUT: "POINT_EMPTY_CHAIR_GUEST2",
        },
    )
    sm.add_state(
        "ANNOUNCE_EMPTY_CHAIR_FOUND_GUEST2",
        AnnounceEmptyChairFoundState(),
        transitions={
            "spoken": "NAVIGATE_TO_INITIAL_POSE_AFTER_GUEST2",
            ABORT: "ANNOUNCE_EMPTY_CHAIR_FOUND_GUEST2",
            TIMEOUT: "ANNOUNCE_EMPTY_CHAIR_FOUND_GUEST2",
        },
    )
    sm.add_state(
        "NAVIGATE_TO_INITIAL_POSE_AFTER_GUEST2",
        NavigateToInitialPoseState(),
        transitions={
            NAVIGATION_DONE: "ANNOUNCE_TASK_COMPLETED",
            ABORT: "NAVIGATE_TO_INITIAL_POSE_AFTER_GUEST2",
            TIMEOUT: "NAVIGATE_TO_INITIAL_POSE_AFTER_GUEST2",
        },
    )
    sm.add_state(
        "ANNOUNCE_TASK_COMPLETED",
        SpeakState(lambda _: "I had completed my task", "spoken"),
        transitions={"spoken": FINAL_OUTCOME, ABORT: "ANNOUNCE_TASK_COMPLETED"},
    )
    sm.add_state(
        "EMPTY_CHAIR_DETECTION_RETRY_WARNING_GUEST2",
        SpeakState(lambda _: "Sorry, I cannot find any empty seat, let me try again", "spoken"),
        transitions={"spoken": "DETECT_EMPTY_CHAIR_GUEST2", ABORT: "EMPTY_CHAIR_DETECTION_RETRY_WARNING_GUEST2"},
    )

    return sm


def create_blackboard() -> Blackboard:
    blackboard = Blackboard()
    blackboard["sim"] = get_sim_mode()
    blackboard["initial_pose"] = dict(initial_pose)
    blackboard["initial_host"] = dict(initial_host)
    blackboard["chair2"] = dict(chair2)
    blackboard["approach_guest"] = dict(approach_guest)
    blackboard["guest1_intro_navigation_target"] = normalize_intro_navigation_target(
        GUEST1_INTRO_NAVIGATION_TARGET,
        default="initial_host",
    )
    blackboard["guest2_intro_navigation_target"] = normalize_intro_navigation_target(
        GUEST2_INTRO_NAVIGATION_TARGET,
        default="initial_host",
    )
    blackboard["guest1"] = create_guest_memory()
    blackboard["guest2"] = create_guest_memory()
    blackboard["current_guest_key"] = "guest1"
    blackboard["current_guest_number"] = 1
    blackboard["detect_guest_attempt_count"] = 0
    blackboard["detect_guest_max_attempts"] = 0
    blackboard["detect_guest_max_distance_m"] = 2.5
    blackboard["detected_guest_stand_off_m"] = 2.0
    blackboard["detected_guest_min_navigation_distance_m"] = 0.05
    blackboard["detected_guest_pose"] = None
    blackboard["detected_guest_distance_m"] = 0.0
    blackboard["navigation_server_wait_timeout_sec"] = 5.0
    blackboard["navigation_send_goal_timeout_sec"] = 10.0
    blackboard["navigation_timeout_sec"] = 180.0
    blackboard["vlm_intro_timeout_sec"] = 90.0
    blackboard["arm_pose_service_wait_timeout_sec"] = 5.0
    blackboard["arm_pose_call_timeout_sec"] = 30.0
    blackboard["arm_zero_pose_name"] = "zero"
    blackboard["startup_delay_sec"] = 3.0
    blackboard["navigation_delay_sec"] = 8.0
    blackboard["return_navigation_delay_sec"] = 5.0
    blackboard["empty_chair_announce_arm_reset_delay_sec"] = 4.0
    blackboard["empty_chair_announce_complete_delay_sec"] = 2.0
    blackboard["navigation_status"] = -1
    blackboard["navigation_error_code"] = -1
    blackboard["last_navigation_target"] = ""
    blackboard["last_detect_message"] = ""
    blackboard["last_empty_chair_detect_message"] = ""
    blackboard["last_empty_chair_image_path"] = ""
    blackboard["last_error"] = ""
    blackboard["last_spoken_text"] = ""
    blackboard["duplicate_guest_name"] = ""
    blackboard["current_host_intro_text"] = ""
    blackboard["guest1_to_guest2_intro_text"] = ""
    return blackboard


def main() -> None:
    rclpy.init()
    set_ros_loggers()
    yasmin.YASMIN_LOG_INFO("task_state_machine_receptionist")

    state_machine = build_state_machine()
    YasminViewerPub(state_machine, "RECEPTIONIST_TASK")
    blackboard = create_blackboard()
    yasmin.YASMIN_LOG_INFO(f"Receptionist simulation mode: {blackboard['sim']}")

    try:
        outcome = state_machine(blackboard)
        yasmin.YASMIN_LOG_INFO(
            "Receptionist state machine finished with outcome=%s sim=%s navigation_status=%s "
            "navigation_error_code=%s last_navigation_target=%s last_error=%s guest1=%s guest2=%s"
            % (
                outcome,
                blackboard["sim"],
                blackboard["navigation_status"],
                blackboard["navigation_error_code"],
                blackboard["last_navigation_target"],
                blackboard["last_error"],
                json.dumps(ensure_guest_memory(blackboard, "guest1"), ensure_ascii=True),
                json.dumps(ensure_guest_memory(blackboard, "guest2"), ensure_ascii=True),
            )
        )
    except Exception as exc:
        yasmin.YASMIN_LOG_WARN(f"Receptionist state machine crashed: {exc}")
        raise
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

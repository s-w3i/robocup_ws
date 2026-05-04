#!/usr/bin/env python3
"""Telegram bot bridge for the lost-and-found VLM check."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import requests
import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rcl_interfaces.msg import ParameterType
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.task import Future

from robot_arm_action.srv import ArmPose
from vlm_interfaces.srv import (
    CaptureImage,
    LocationImage,
    LostFoundVlmCheck,
    TelegramReply,
    TelegramVisualSearch,
    VisualQuestion,
)


DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_OPENAI_API_URL = "https://api.openai.com/v1/responses"
DEFAULT_OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
DEFAULT_SEARCH_LOCATIONS = ["living room", "bedroom"]


# Edit these named navigation targets like the carry_my_luggage initial_pose.
location_pose_map: dict[str, dict[str, Any]] = {
    "living room": {
        "frame_id": "map",
        "x": -1.57401,
        "y": -4.54915,
        "z": 0.0,
        "orientation": {
            "x": 0.0,
            "y": 0.0,
            "z": 0.999925,
            "w": 0.012214,
        },
    },
    "bedroom": {
        "frame_id": "map",
        "x": 0.909185,
        "y": -5.19148,
        "z": 0.0,
        "orientation": {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0275075,
            "w": 0.999622,
        },
    },
}


@dataclass(frozen=True)
class SearchRequest:
    object_name: str
    locations: list[str]
    original_text: str


@dataclass
class LocationResult:
    location: str
    success: bool
    found: bool = False
    confidence: float = 0.0
    image_path: str = ""
    annotated_image_path: str = ""
    camera_name: str = ""
    reason: str = ""
    error: str = ""


class QueryParseError(ValueError):
    pass


class SearchWorkflowState(str, Enum):
    RECEIVED = "received"
    NAVIGATING = "navigating"
    ARM_MOVING = "arm_moving"
    CAPTURING_LOCATION = "capturing_location"
    QUERYING_VLM = "querying_vlm"
    FOUND = "found"
    NOT_FOUND = "not_found"
    FAILED = "failed"


def normalize_chat_text(text: str) -> str:
    cleaned = str(text).replace("\n", " ")
    return " ".join(cleaned.strip().split())


def strip_leading_article(text: str) -> str:
    return re.sub(r"^(?:the|my|a|an)\s+", "", text.strip(), flags=re.IGNORECASE).strip()


def parse_lost_found_query(text: str) -> SearchRequest:
    original = normalize_chat_text(text)
    lowered = re.sub(r"[?!]+", "", original.lower()).strip()
    lowered = re.sub(r"\b(on|in|at)\s+(?=(?:on|in|at)\b)", "", lowered)
    matches = list(re.finditer(r"\b(on|in|at|inside|near|under|by)\b", lowered))
    match = matches[-1] if matches else None
    if not match:
        raise QueryParseError("Please tell me the object and where to check, for example: check my key on the table or bed.")

    before_location = lowered[: match.start()].strip(" ,.")
    raw_locations = lowered[match.end() :].strip(" ,.")
    raw_locations = re.sub(r"\b(?:please|for me|now)\b", " ", raw_locations)
    location_parts = re.split(r"\s*(?:,|\bor\b|\band\b|\bthen\b)\s*", raw_locations)
    locations: list[str] = []
    for part in location_parts:
        location = re.sub(r"^(?:on|in|at|inside|near|under|by)\s+", "", part.strip(" ."), flags=re.IGNORECASE)
        location = strip_leading_article(location)
        if not location:
            continue
        if location not in locations:
            locations.append(location)
    if not locations:
        raise QueryParseError("Please tell me at least one place to check.")

    # Prefer explicit find/check command patterns first.
    pattern_candidates = [
        r"\b(?:help me(?: to)?\s+)?(?:find|look for|search for|check|see if)\s+(?:(?:my|the|a|an)\s+)?(?P<object>[a-z0-9][a-z0-9 _-]{0,80})$",
        r"\b(?:did i leave|have i left|i left)\s+(?:(?:my|the|a|an)\s+)?(?P<object>[a-z0-9][a-z0-9 _-]{0,80})$",
        r"\b(?:my|the|a|an)\s+(?P<object>[a-z0-9][a-z0-9 _-]{0,80})$",
    ]
    object_name = ""
    for pattern in pattern_candidates:
        matched = re.search(pattern, before_location, flags=re.IGNORECASE)
        if matched:
            object_name = str(matched.group("object") or "").strip()
            break

    if not object_name:
        object_phrase = re.sub(
            r"^(?:can you|could you|please|help me(?: to)?|can you help me|robot|check|find|look for|search for|is|if|whether|tell me|see if|can you see if|did i leave|have i left|i left)\b",
            " ",
            before_location,
            flags=re.IGNORECASE,
        )
        object_phrase = re.sub(
            r"\b(?:can you|could you|please|help me|check|find|look for|search for|is|if|whether|tell me|see if|left|leave|to)\b",
            " ",
            object_phrase,
            flags=re.IGNORECASE,
        )
        object_name = " ".join(object_phrase.split())

    object_name = strip_leading_article(object_name)
    object_name = re.sub(r"^[^a-z0-9]+|[^a-z0-9]+$", "", object_name, flags=re.IGNORECASE).strip()
    object_name = object_name.strip(" .")
    if not object_name:
        raise QueryParseError("Please tell me what object to look for.")

    return SearchRequest(object_name=object_name, locations=locations, original_text=original)


class TelegramClient:
    def __init__(self, token: str, timeout_sec: float = 20.0) -> None:
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.timeout_sec = max(1.0, float(timeout_sec))
        self.max_send_attempts = 3

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
        updates = data.get("result", [])
        return updates if isinstance(updates, list) else []

    def send_message(self, chat_id: int | str, text: str, reply_to_message_id: int | None = None) -> None:
        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "text": text,
        }
        if reply_to_message_id is not None:
            payload["reply_to_message_id"] = int(reply_to_message_id)
        response = self._post_with_retries(
            "sendMessage",
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
        data: dict[str, Any] = {
            "chat_id": str(chat_id),
            "caption": caption,
        }
        if reply_to_message_id is not None:
            data["reply_to_message_id"] = str(int(reply_to_message_id))
        response = None
        last_exc: Exception | None = None
        for attempt in range(1, self.max_send_attempts + 1):
            try:
                with path.open("rb") as image_file:
                    response = requests.post(
                        f"{self.base_url}/sendPhoto",
                        data=data,
                        files={"photo": (path.name, image_file)},
                        timeout=(10.0, 60.0),
                    )
                break
            except requests.RequestException as exc:
                last_exc = exc
                if attempt >= self.max_send_attempts:
                    raise
                time.sleep(0.75 * attempt)
        if response is None:
            raise RuntimeError(str(last_exc) if last_exc is not None else "sendPhoto failed")
        response.raise_for_status()
        return True

    def _post_with_retries(self, method: str, **kwargs: Any) -> requests.Response:
        last_exc: Exception | None = None
        for attempt in range(1, self.max_send_attempts + 1):
            try:
                return requests.post(f"{self.base_url}/{method}", **kwargs)
            except requests.RequestException as exc:
                last_exc = exc
                if attempt >= self.max_send_attempts:
                    raise
                time.sleep(0.75 * attempt)
        raise RuntimeError(str(last_exc) if last_exc is not None else f"{method} failed")


class TelegramLostFoundNode(Node):
    def __init__(self) -> None:
        super().__init__("telegram_lost_found_node")

        self.declare_parameter("bot_token", "")
        self.declare_parameter("bot_token_file", "")
        self.declare_parameter("allowed_user_ids", [])
        self.declare_parameter("allowed_user_ids_csv", "")
        self.declare_parameter("lost_found_vlm_service", "/lost_found/vlm_check")
        self.declare_parameter("visual_question_service", "/lost_found/visual_question")
        self.declare_parameter("telegram_reply_service", "/telegram/reply")
        self.declare_parameter("telegram_visual_search_service", "/telegram/visual_search")
        self.declare_parameter("location_image_service", "/lost_found/location_image")
        self.declare_parameter("capture_image_service", "/camera/capture")
        self.declare_parameter(
            "default_chat_id",
            "",
            descriptor=self._dynamic_parameter_descriptor("Telegram chat id used when service calls leave chat_id empty."),
        )
        self.declare_parameter("default_camera_name", "gripper_camera")
        self.declare_parameter("navigation_action_name", "/navigate_to_pose")
        self.declare_parameter("arm_pose_service", "/arm_pose")
        self.declare_parameter("arm_detect_pose_name", "detect")
        self.declare_parameter("arm_zero_pose_name", "zero")
        self.declare_parameter("navigation_server_wait_timeout_sec", 5.0)
        self.declare_parameter("navigation_send_goal_timeout_sec", 5.0)
        self.declare_parameter("navigation_timeout_sec", 90.0)
        self.declare_parameter("arm_pose_timeout_sec", 15.0)
        self.declare_parameter("demo_mode", True)
        self.declare_parameter("sim_mode", False)
        self.declare_parameter("demo_navigation_delay_sec", 5.0)
        self.declare_parameter("location_pose_map_json", "")
        self.declare_parameter("service_wait_timeout_sec", 5.0)
        self.declare_parameter("location_timeout_sec", 120.0)
        self.declare_parameter("vlm_timeout_sec", 75.0)
        self.declare_parameter("found_confidence_threshold", 0.65)
        self.declare_parameter("telegram_poll_timeout_sec", 20.0)
        self.declare_parameter("ollama_base_url", os.environ.get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL))
        self.declare_parameter("openai_api_url", os.environ.get("OPENAI_API_URL", DEFAULT_OPENAI_API_URL))
        self.declare_parameter("openai_api_key_env", os.environ.get("OPENAI_API_KEY_ENV", DEFAULT_OPENAI_API_KEY_ENV))
        self.declare_parameter("use_openai_chat", False)
        self.declare_parameter("chat_model", os.environ.get("TEXT_MODEL", "qwen3.5:9b"))
        self.declare_parameter("chat_timeout_sec", 60.0)
        self.declare_parameter("chat_num_predict", 160)
        self.declare_parameter("max_history_messages", 10)
        self.declare_parameter("auto_start_ollama", True)
        self.declare_parameter(
            "chat_system_prompt",
            (
                "You are Ava, a concise helpful robot assistant on Telegram. "
                "Answer normal chat messages briefly in English. "
                "If the user wants the robot to look through its camera, tell them to use /search followed by the question."
            ),
        )

        token = str(self.get_parameter("bot_token").value).strip()
        token_file = str(self.get_parameter("bot_token_file").value).strip()
        if not token and token_file:
            token_path = Path(token_file).expanduser()
            if not token_path.is_file():
                raise ValueError(
                    f"Telegram bot token file not found: {token_path}. "
                    "Create it with the token from @BotFather, or pass bot_token:=..."
                )
            token = token_path.read_text(encoding="utf-8").strip()
        if not token:
            raise ValueError("Set bot_token or bot_token_file for Telegram polling.")

        self.lost_found_vlm_service = (
            str(self.get_parameter("lost_found_vlm_service").value).strip() or "/lost_found/vlm_check"
        )
        self.visual_question_service = (
            str(self.get_parameter("visual_question_service").value).strip() or "/lost_found/visual_question"
        )
        self.telegram_reply_service = (
            str(self.get_parameter("telegram_reply_service").value).strip() or "/telegram/reply"
        )
        self.telegram_visual_search_service = (
            str(self.get_parameter("telegram_visual_search_service").value).strip()
            or "/telegram/visual_search"
        )
        self.location_image_service = (
            str(self.get_parameter("location_image_service").value).strip() or "/lost_found/location_image"
        )
        self.capture_image_service = (
            str(self.get_parameter("capture_image_service").value).strip() or "/camera/capture"
        )
        self.default_chat_id = str(self.get_parameter("default_chat_id").value).strip()
        self.default_camera_name = str(self.get_parameter("default_camera_name").value).strip() or "gripper_camera"
        self.navigation_action_name = (
            str(self.get_parameter("navigation_action_name").value).strip() or "/navigate_to_pose"
        )
        self.arm_pose_service = str(self.get_parameter("arm_pose_service").value).strip() or "/arm_pose"
        self.arm_detect_pose_name = str(self.get_parameter("arm_detect_pose_name").value).strip() or "detect"
        self.arm_zero_pose_name = str(self.get_parameter("arm_zero_pose_name").value).strip() or "zero"
        self.navigation_server_wait_timeout_sec = max(
            0.1, float(self.get_parameter("navigation_server_wait_timeout_sec").value)
        )
        self.navigation_send_goal_timeout_sec = max(
            0.1, float(self.get_parameter("navigation_send_goal_timeout_sec").value)
        )
        self.navigation_timeout_sec = max(1.0, float(self.get_parameter("navigation_timeout_sec").value))
        self.arm_pose_timeout_sec = max(0.1, float(self.get_parameter("arm_pose_timeout_sec").value))
        self.demo_mode = bool(self.get_parameter("demo_mode").value)
        self.sim_mode = bool(self.get_parameter("sim_mode").value)
        self.demo_navigation_delay_sec = max(0.0, float(self.get_parameter("demo_navigation_delay_sec").value))
        self.location_pose_map = {self._normalize_location_key(key): dict(value) for key, value in location_pose_map.items()}
        location_pose_map_override = self._parse_location_pose_map_json(
            str(self.get_parameter("location_pose_map_json").value).strip()
        )
        if location_pose_map_override:
            self.location_pose_map.update(location_pose_map_override)
        self.service_wait_timeout_sec = max(0.1, float(self.get_parameter("service_wait_timeout_sec").value))
        self.location_timeout_sec = max(1.0, float(self.get_parameter("location_timeout_sec").value))
        self.vlm_timeout_sec = max(1.0, float(self.get_parameter("vlm_timeout_sec").value))
        self.found_confidence_threshold = max(
            0.0,
            min(1.0, float(self.get_parameter("found_confidence_threshold").value)),
        )
        self.allowed_user_ids = self._parse_int_set(
            self.get_parameter("allowed_user_ids").value,
            str(self.get_parameter("allowed_user_ids_csv").value).strip(),
        )
        self.ollama_base_url = str(self.get_parameter("ollama_base_url").value).strip().rstrip("/")
        self.openai_api_url = str(self.get_parameter("openai_api_url").value).strip() or DEFAULT_OPENAI_API_URL
        self.openai_api_key_env = (
            str(self.get_parameter("openai_api_key_env").value).strip() or DEFAULT_OPENAI_API_KEY_ENV
        )
        self.use_openai_chat = bool(self.get_parameter("use_openai_chat").value)
        self.chat_model = str(self.get_parameter("chat_model").value).strip()
        self.chat_timeout_sec = max(1.0, float(self.get_parameter("chat_timeout_sec").value))
        self.chat_num_predict = max(32, int(self.get_parameter("chat_num_predict").value))
        self.max_history_messages = max(2, int(self.get_parameter("max_history_messages").value))
        self.auto_start_ollama = bool(self.get_parameter("auto_start_ollama").value)
        self.chat_system_prompt = str(self.get_parameter("chat_system_prompt").value).strip()

        self._callback_group = ReentrantCallbackGroup()
        self._location_client = self.create_client(
            LocationImage,
            self.location_image_service,
            callback_group=self._callback_group,
        )
        self._capture_image_client = self.create_client(
            CaptureImage,
            self.capture_image_service,
            callback_group=self._callback_group,
        )
        self._lost_found_vlm_client = self.create_client(
            LostFoundVlmCheck,
            self.lost_found_vlm_service,
            callback_group=self._callback_group,
        )
        self._visual_question_client = self.create_client(
            VisualQuestion,
            self.visual_question_service,
            callback_group=self._callback_group,
        )
        self._arm_pose_client = self.create_client(
            ArmPose,
            self.arm_pose_service,
            callback_group=self._callback_group,
        )
        self._navigate_client = ActionClient(
            self,
            NavigateToPose,
            self.navigation_action_name,
            callback_group=self._callback_group,
        )
        self.create_service(
            TelegramReply,
            self.telegram_reply_service,
            self._handle_telegram_reply_service,
            callback_group=self._callback_group,
        )
        self.create_service(
            TelegramVisualSearch,
            self.telegram_visual_search_service,
            self._handle_telegram_visual_search_service,
            callback_group=self._callback_group,
        )
        self._telegram = TelegramClient(
            token,
            timeout_sec=float(self.get_parameter("telegram_poll_timeout_sec").value),
        )
        self._active_chats: set[int | str] = set()
        self._chat_histories: dict[int | str, list[dict[str, str]]] = {}
        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()
        if self.auto_start_ollama and not self.use_openai_chat:
            self._ensure_ollama_running()
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()
        self.get_logger().info(
            "Telegram robot bot polling started. "
            f"demo_mode={str(self.demo_mode).lower()} "
            f"sim_mode={str(self.sim_mode).lower()} "
            f"chat_backend={'openai' if self.use_openai_chat else 'ollama'} "
            f"chat_model={self.chat_model} visual_question_service={self.visual_question_service} "
            f"reply_service={self.telegram_reply_service} "
            f"visual_search_service={self.telegram_visual_search_service}"
        )

    @staticmethod
    def _dynamic_parameter_descriptor(description: str):
        from rcl_interfaces.msg import ParameterDescriptor

        descriptor = ParameterDescriptor()
        descriptor.description = description
        descriptor.dynamic_typing = True
        descriptor.type = ParameterType.PARAMETER_NOT_SET
        return descriptor

    def _handle_telegram_reply_service(
        self,
        request: TelegramReply.Request,
        response: TelegramReply.Response,
    ) -> TelegramReply.Response:
        chat_id = str(request.chat_id).strip()
        text = str(request.text).strip()
        image_path = str(request.image_path).strip()
        reply_to_message_id = int(request.reply_to_message_id)
        if reply_to_message_id <= 0:
            reply_to_message_id_or_none = None
        else:
            reply_to_message_id_or_none = reply_to_message_id

        response.success = False
        response.message = ""
        if not chat_id:
            response.message = "chat_id is required."
            return response
        if not text and not image_path:
            response.message = "text or image_path is required."
            return response

        try:
            sent_photo = False
            if image_path:
                sent_photo = self._telegram.send_photo(
                    chat_id,
                    image_path,
                    text,
                    reply_to_message_id_or_none,
                )
            if not sent_photo and text:
                self._telegram.send_message(chat_id, text, reply_to_message_id_or_none)
            response.success = True
            response.message = "ok"
            return response
        except Exception as exc:
            response.message = str(exc)
            self.get_logger().error(f"Telegram reply service failed: {exc}")
            return response

    def _handle_telegram_visual_search_service(
        self,
        request: TelegramVisualSearch.Request,
        response: TelegramVisualSearch.Response,
    ) -> TelegramVisualSearch.Response:
        response.accepted = False
        response.message = ""

        chat_id = str(request.chat_id).strip() or self.default_chat_id
        question = str(request.question).strip()
        object_name = str(request.object_name).strip()
        camera_name = str(request.camera_name).strip() or self.default_camera_name
        image_path = str(request.image_path).strip()
        confidence_threshold = float(request.confidence_threshold)
        if confidence_threshold <= 0.0:
            confidence_threshold = self.found_confidence_threshold

        if not chat_id:
            response.message = "chat_id is required, or set default_chat_id."
            return response
        if not question:
            if object_name:
                question = f"Can you see {object_name}?"
            else:
                response.message = "question or object_name is required."
                return response

        with self._state_lock:
            if chat_id in self._active_chats:
                response.message = "A robot search is already active for this chat."
                return response
            self._active_chats.add(chat_id)

        worker = threading.Thread(
            target=self._run_visual_question,
            args=(question, chat_id, None, object_name, camera_name, image_path, confidence_threshold),
            daemon=True,
        )
        worker.start()
        response.accepted = True
        response.message = "accepted"
        return response

    @staticmethod
    def _parse_int_set(values: Any, csv_text: str) -> set[int]:
        raw_values: list[Any]
        if csv_text:
            raw_values = [part.strip() for part in csv_text.split(",")]
        elif isinstance(values, (list, tuple)):
            raw_values = list(values)
        else:
            raw_values = []
        parsed: set[int] = set()
        for value in raw_values:
            text = str(value).strip()
            if not text:
                continue
            parsed.add(int(text))
        return parsed

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

        result_wrapper = self._wait_for_future(goal_handle.get_result_async(), self.navigation_timeout_sec)
        if result_wrapper is None:
            return False, f"Timed out waiting for navigation result to '{location}'."
        status = int(getattr(result_wrapper, "status", GoalStatus.STATUS_UNKNOWN))
        result = getattr(result_wrapper, "result", None)
        error_code = int(getattr(result, "error_code", 0)) if result is not None else 0
        if status == GoalStatus.STATUS_SUCCEEDED and error_code == 0:
            return True, "ok"
        return False, f"Navigation to '{location}' failed with status={status} error_code={error_code}."

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

    def _capture_image(self, camera_name: str, file_prefix: str) -> str:
        if not self._capture_image_client.wait_for_service(timeout_sec=self.service_wait_timeout_sec):
            self.get_logger().warn(f"Capture image service is not available: {self.capture_image_service}")
            return ""
        request = CaptureImage.Request()
        request.camera_name = camera_name or self.default_camera_name
        request.save_dir = "/home/usern/robocup_ws/captures"
        request.file_prefix = file_prefix
        response = self._wait_for_future(self._capture_image_client.call_async(request), min(self.vlm_timeout_sec, 15.0))
        if response is None or not bool(response.success):
            return ""
        return str(response.saved_image_path).strip()

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
        self.get_logger().info(f"Telegram message received | chat_id={chat_id} user_id={user_id} text={text[:80]!r}")

        if text.startswith("/start"):
            self._telegram.send_message(
                chat_id,
                (
                    "Hi, I am Ava. Chat with me normally. "
                    "You can ask naturally, for example: can you help me see if I left my bottle in the bedroom or living room? "
                    "Use /search only for direct camera questions."
                ),
                self._message_id_or_none(message_id),
            )
            return

        if self.allowed_user_ids and int(user_id or 0) not in self.allowed_user_ids:
            self._telegram.send_message(chat_id, "Sorry, this robot bot is restricted.", self._message_id_or_none(message_id))
            return

        lowered = text.lower().strip()
        if lowered.startswith("/search") or lowered.startswith("/look"):
            question = self._strip_command(text)
            if not question:
                self._telegram.send_message(
                    chat_id,
                    "Please send a visual question after /search, for example: /search is my key on the table?",
                    self._message_id_or_none(message_id),
                )
                return
            self._start_visual_question(question, chat_id, self._message_id_or_none(message_id))
            return

        if lowered.startswith("/lost"):
            text = self._strip_command(text)
            if not text:
                self._telegram.send_message(
                    chat_id,
                    "Please send the lost item request after /lost, for example: /lost check my key on the table or bed",
                    self._message_id_or_none(message_id),
                )
                return
            search = self._maybe_parse_search_request(text)
            if search is None:
                self._telegram.send_message(
                    chat_id,
                    "Please tell me the object and places to check, for example: /lost my key in the bedroom or living room",
                    self._message_id_or_none(message_id),
                )
                return
            self._start_search_request(search, chat_id, self._message_id_or_none(message_id))
            return

        search = self._maybe_parse_search_request(text)
        if search is not None:
            self._start_search_request(search, chat_id, self._message_id_or_none(message_id))
            return

        self._start_chat_reply(text, chat_id, self._message_id_or_none(message_id))

    @staticmethod
    def _strip_command(text: str) -> str:
        return re.sub(r"^/\w+(?:@\w+)?\s*", "", str(text).strip(), count=1).strip()

    def _maybe_parse_search_request(self, text: str) -> SearchRequest | None:
        normalized = normalize_chat_text(text).lower()
        intent_hint = re.search(
            r"\b(find|look for|search for|check|see if|did i leave|have i left|i left|lost)\b",
            normalized,
        )
        if intent_hint is None:
            return None
        try:
            parsed = parse_lost_found_query(text)
            return SearchRequest(
                object_name=parsed.object_name,
                locations=list(DEFAULT_SEARCH_LOCATIONS),
                original_text=parsed.original_text,
            )
        except QueryParseError:
            return None

    def _start_visual_question(self, question: str, chat_id: int | str, reply_to_message_id: int | None) -> None:
        with self._state_lock:
            if chat_id in self._active_chats:
                self._telegram.send_message(chat_id, "I am already working on a robot request for this chat.", reply_to_message_id)
                return
            self._active_chats.add(chat_id)

        self._telegram.send_message(chat_id, "Ok, please wait a moment.", reply_to_message_id)
        worker = threading.Thread(
            target=self._run_visual_question,
            args=(question, chat_id, reply_to_message_id, "", self.default_camera_name, "", self.found_confidence_threshold),
            daemon=True,
        )
        worker.start()

    def _start_chat_reply(self, text: str, chat_id: int | str, reply_to_message_id: int | None) -> None:
        worker = threading.Thread(
            target=self._run_chat_reply,
            args=(text, chat_id, reply_to_message_id),
            daemon=True,
        )
        worker.start()

    def _start_legacy_lost_found(self, text: str, chat_id: int | str, reply_to_message_id: int | None) -> None:
        search = self._maybe_parse_search_request(text)
        if search is None:
            self._telegram.send_message(chat_id, "Please tell me what object to find and where to check.", reply_to_message_id)
            return
        self._start_search_request(search, chat_id, reply_to_message_id)

    def _start_search_request(self, search: SearchRequest, chat_id: int | str, reply_to_message_id: int | None) -> None:
        try:
            self._log_search_state(chat_id, SearchWorkflowState.RECEIVED, search.original_text)
            self.get_logger().info(
                "Parsed search request | "
                f"object={search.object_name!r} locations={search.locations!r}"
            )
            with self._state_lock:
                if chat_id in self._active_chats:
                    self._telegram.send_message(chat_id, "I am already working on a robot request for this chat.", reply_to_message_id)
                    return
                self._active_chats.add(chat_id)

            self._telegram.send_message(chat_id, self._build_ack(search), reply_to_message_id)
            worker = threading.Thread(
                target=self._run_search_session,
                args=(search, chat_id, reply_to_message_id),
                daemon=True,
            )
            worker.start()
        except QueryParseError as exc:
            self._telegram.send_message(chat_id, str(exc), reply_to_message_id)

    def _run_visual_question(
        self,
        question: str,
        chat_id: int | str,
        reply_to_message_id: int | None,
        object_name: str = "",
        camera_name: str = "",
        image_path: str = "",
        confidence_threshold: float | None = None,
    ) -> None:
        try:
            response = self._request_visual_question(
                question,
                object_name=object_name,
                camera_name=camera_name,
                image_path=image_path,
                confidence_threshold=confidence_threshold,
            )
            if response is None:
                self._send_telegram_message(chat_id, "I could not get a visual answer in time.", reply_to_message_id)
                return
            if not bool(response.success):
                self._send_telegram_message(
                    chat_id,
                    f"I could not check the camera: {str(response.message).strip()}",
                    reply_to_message_id,
                )
                return
            reply_text = str(response.reply_text).strip() or str(response.reason).strip() or "I checked the camera."
            image_path = str(response.image_path).strip()
            if bool(response.found):
                image_path = self._annotate_image_from_data_text(
                    image_path,
                    str(response.data_text).strip(),
                    object_name=str(response.object_name).strip() if hasattr(response, "object_name") else object_name,
                )
            sent_photo = False
            if image_path:
                try:
                    sent_photo = self._telegram.send_photo(chat_id, image_path, reply_text, reply_to_message_id)
                except Exception as exc:
                    self.get_logger().warn(f"Failed to send Telegram photo; sending text only: {exc}")
            if not sent_photo:
                self._send_telegram_message(chat_id, reply_text, reply_to_message_id)
            self.get_logger().info(
                "Visual question Telegram reply sent | "
                f"chat_id={chat_id} found={bool(response.found)} confidence={float(response.confidence):.2f}"
            )
        except Exception as exc:
            self.get_logger().error(f"Visual question worker failed before reply: {exc}")
            self._send_telegram_message(
                chat_id,
                "Sorry, I checked the camera but could not send the result correctly.",
                reply_to_message_id,
            )
        finally:
            with self._state_lock:
                self._active_chats.discard(chat_id)

    def _send_telegram_message(
        self,
        chat_id: int | str,
        text: str,
        reply_to_message_id: int | None = None,
    ) -> bool:
        try:
            self._telegram.send_message(chat_id, text, reply_to_message_id)
            return True
        except Exception as exc:
            self.get_logger().error(f"Failed to send Telegram message: {exc}")
            return False

    def _request_visual_question(
        self,
        question: str,
        *,
        object_name: str = "",
        camera_name: str = "",
        image_path: str = "",
        confidence_threshold: float | None = None,
    ) -> Any | None:
        if not self._visual_question_client.wait_for_service(timeout_sec=self.service_wait_timeout_sec):
            self.get_logger().warn(f"Visual question service is not available: {self.visual_question_service}")
            return None

        request = VisualQuestion.Request()
        request.question = question
        request.object_name = object_name.strip() or self._extract_object_hint(question)
        request.camera_name = camera_name.strip() or self.default_camera_name
        request.image_path = image_path.strip()
        request.confidence_threshold = float(
            self.found_confidence_threshold if confidence_threshold is None else confidence_threshold
        )
        future = self._visual_question_client.call_async(request)
        return self._wait_for_future(future, self.vlm_timeout_sec)

    @staticmethod
    def _extract_object_hint(question: str) -> str:
        text = str(question).strip().lower()
        patterns = [
            r"\b(?:my|the|a|an)\s+([a-z0-9][a-z0-9 _-]{0,40}?)(?:\s+(?:on|in|at|near|under|over)\b|[?.!,]|$)",
            r"\b(?:look for|search for|find|see|check)\s+(?:my|the|a|an)?\s*([a-z0-9][a-z0-9 _-]{0,40}?)(?:\s+(?:on|in|at|near|under|over)\b|[?.!,]|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return strip_leading_article(match.group(1)).strip()
        return ""

    def _run_chat_reply(self, text: str, chat_id: int | str, reply_to_message_id: int | None) -> None:
        try:
            reply = self._chat_with_model(chat_id, text)
        except Exception as exc:
            self.get_logger().warn(f"Chat reply failed: {exc}")
            reply = "Sorry, I am having trouble answering right now."
        self._send_telegram_message(chat_id, reply, reply_to_message_id)

    def _chat_with_model(self, chat_id: int | str, text: str) -> str:
        if self.use_openai_chat:
            return self._chat_with_openai(chat_id, text)
        return self._chat_with_ollama(chat_id, text)

    def _chat_with_ollama(self, chat_id: int | str, text: str) -> str:
        with self._state_lock:
            history = list(self._chat_histories.get(chat_id, []))
        messages = [{"role": "system", "content": self.chat_system_prompt}] + history
        messages.append({"role": "user", "content": text})
        payload: dict[str, Any] = {
            "model": self.chat_model,
            "messages": messages,
            "stream": False,
            "keep_alive": "30m",
            "think": False,
            "options": {
                "temperature": 0.2,
                "num_predict": self.chat_num_predict,
            },
        }
        result = requests.post(
            f"{self.ollama_base_url}/api/chat",
            json=payload,
            timeout=self.chat_timeout_sec,
        )
        result.raise_for_status()
        reply = str(result.json().get("message", {}).get("content", "") or "").strip()
        reply = self._clean_chat_reply(reply) or "I am not sure how to answer that."
        with self._state_lock:
            updated = history + [
                {"role": "user", "content": text},
                {"role": "assistant", "content": reply},
            ]
            self._chat_histories[chat_id] = updated[-self.max_history_messages :]
        return reply

    def _chat_with_openai(self, chat_id: int | str, text: str) -> str:
        with self._state_lock:
            history = list(self._chat_histories.get(chat_id, []))

        input_items: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": self.chat_system_prompt}],
            }
        ]
        for message in history:
            role = str(message.get("role", "")).strip()
            content = str(message.get("content", "")).strip()
            if role in {"user", "assistant"} and content:
                input_items.append(
                    {
                        "role": role,
                        "content": [{"type": "input_text", "text": content}],
                    }
                )
        input_items.append(
            {
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
            }
        )

        payload: dict[str, Any] = {
            "model": self.chat_model,
            "input": input_items,
            "max_output_tokens": self.chat_num_predict,
            "store": False,
        }
        result = requests.post(
            self.openai_api_url,
            headers={
                "Authorization": f"Bearer {self._get_openai_api_key()}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.chat_timeout_sec,
        )
        if not result.ok:
            raise RuntimeError(f"OpenAI request failed: HTTP {result.status_code} {result.text[:500]}")
        reply = self._clean_chat_reply(self._extract_openai_output_text(result.json()))
        reply = reply or "I am not sure how to answer that."
        with self._state_lock:
            updated = history + [
                {"role": "user", "content": text},
                {"role": "assistant", "content": reply},
            ]
            self._chat_histories[chat_id] = updated[-self.max_history_messages :]
        return reply

    def _get_openai_api_key(self) -> str:
        api_key = os.environ.get(self.openai_api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(
                f"OpenAI API key is not set. Export {self.openai_api_key_env} before starting this node."
            )
        return api_key

    @staticmethod
    def _extract_openai_output_text(payload: dict[str, Any]) -> str:
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
                    text = str(content.get("text") or "").strip()
                    if text:
                        parts.append(text)
        return "\n".join(parts).strip()

    @staticmethod
    def _clean_chat_reply(reply: str) -> str:
        cleaned = re.sub(r"<think>.*?</think>", "", str(reply), flags=re.DOTALL | re.IGNORECASE).strip()
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                for key in ("assistant_reply", "reply", "text", "speech_text"):
                    value = str(parsed.get(key) or "").strip()
                    if value:
                        return value
        except Exception:
            pass
        return cleaned

    def _ollama_ready(self, timeout: float = 1.0) -> bool:
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=timeout)
            return response.ok
        except requests.RequestException:
            return False

    def _ensure_ollama_running(self) -> None:
        if self._ollama_ready(timeout=1.0):
            return
        self.get_logger().info("Ollama is not reachable; starting 'ollama serve'.")
        try:
            subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except Exception as exc:
            self.get_logger().warn(f"Unable to start ollama serve: {exc}")

    @staticmethod
    def _message_id_or_none(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _build_ack(search: SearchRequest) -> str:
        if len(search.locations) == 1:
            return f"I'll check the {search.locations[0]} for your {search.object_name}."
        return f"I'll check the {search.locations[0]} first, then the {', then '.join(search.locations[1:])}."

    def _log_search_state(self, key: int | str, state: SearchWorkflowState, detail: str = "") -> None:
        suffix = f" | detail={detail}" if detail else ""
        self.get_logger().info(f"Search workflow [{key}] -> {state.value}{suffix}")

    def _run_search_session(self, search: SearchRequest, chat_id: int | str, reply_to_message_id: int | None) -> None:
        session_id = uuid.uuid4().hex
        results: list[LocationResult] = []
        try:
            for index, location in enumerate(search.locations):
                self._send_telegram_message(
                    chat_id,
                    self._build_room_progress_text(search, location, index, len(search.locations)),
                    reply_to_message_id,
                )
                result = self._check_location(session_id, chat_id, search, location, index)
                results.append(result)
                self._send_telegram_message(
                    chat_id,
                    self._build_room_result_text(search, result),
                    reply_to_message_id,
                )
                if result.success and result.found:
                    self._log_search_state(chat_id, SearchWorkflowState.FOUND, location)
                    break
                if not result.success:
                    self._log_search_state(chat_id, SearchWorkflowState.FAILED, result.error or location)
            final_text, image_path = self._build_final_reply(search, results)
            if not any(result.success and result.found for result in results):
                self._log_search_state(chat_id, SearchWorkflowState.NOT_FOUND, search.object_name)
            sent_photo = False
            if image_path:
                try:
                    sent_photo = self._telegram.send_photo(chat_id, image_path, final_text, reply_to_message_id)
                except Exception as exc:
                    self.get_logger().warn(f"Failed to send Telegram photo; sending text only: {exc}")
            if not sent_photo:
                self._send_telegram_message(chat_id, final_text, reply_to_message_id)
        finally:
            with self._state_lock:
                self._active_chats.discard(chat_id)

    def _check_location(
        self,
        session_id: str,
        chat_id: int | str,
        search: SearchRequest,
        location: str,
        index: int,
    ) -> LocationResult:
        self._log_search_state(chat_id, SearchWorkflowState.NAVIGATING, location)
        nav_success, nav_message = self._navigate_to_location(location)
        if not nav_success:
            return LocationResult(location=location, success=False, error=nav_message)

        self._log_search_state(chat_id, SearchWorkflowState.ARM_MOVING, self.arm_detect_pose_name)
        arm_success, arm_message = self._move_arm_pose(self.arm_detect_pose_name)
        if not arm_success:
            return LocationResult(location=location, success=False, error=arm_message)

        location_response = self._request_location_image(session_id, search, location, index)
        if location_response is None:
            image_path = ""
            camera_name = self.default_camera_name
        else:
            image_path = str(location_response.image_path).strip()
            camera_name = str(location_response.camera_name).strip() or self.default_camera_name
            if not bool(location_response.success):
                self.get_logger().warn(
                    f"Location image request for '{location}' failed, falling back to direct capture: "
                    f"{str(location_response.message).strip() or 'location image request failed.'}"
                )

        self._log_search_state(chat_id, SearchWorkflowState.CAPTURING_LOCATION, location)
        if not image_path:
            image_path = self._capture_image(camera_name, f"search_{self._normalize_location_key(location).replace(' ', '_')}")

        self._log_search_state(chat_id, SearchWorkflowState.QUERYING_VLM, location)
        vlm_response = self._request_lost_found_vlm(search, location, image_path, camera_name)
        if vlm_response is None:
            return LocationResult(
                location=location,
                success=False,
                image_path=image_path,
                camera_name=camera_name,
                error="Timed out waiting for VLM response.",
            )
        if not bool(vlm_response.success):
            return LocationResult(
                location=location,
                success=False,
                image_path=image_path,
                camera_name=camera_name,
                error=str(vlm_response.message).strip() or "VLM request failed.",
            )

        try:
            found = bool(vlm_response.found)
            confidence = float(vlm_response.confidence)
            reason = str(vlm_response.reason).strip()
        except Exception as exc:
            return LocationResult(
                location=location,
                success=False,
                image_path=image_path,
                camera_name=camera_name,
                error=str(exc),
            )

        return LocationResult(
            location=location,
            success=True,
            found=found,
            confidence=confidence,
            image_path=image_path,
            annotated_image_path=self._annotate_image_from_data_text(
                image_path,
                str(vlm_response.data_text).strip(),
                object_name=search.object_name,
            )
            if found
            else "",
            camera_name=camera_name,
            reason=reason,
        )

    def _request_location_image(
        self,
        session_id: str,
        search: SearchRequest,
        location: str,
        index: int,
    ) -> Any | None:
        if not self._location_client.wait_for_service(timeout_sec=self.service_wait_timeout_sec):
            self.get_logger().warn(f"Location image service is not available: {self.location_image_service}")
            return None

        request = LocationImage.Request()
        request.session_id = session_id
        request.object_name = search.object_name
        request.location_name = location
        request.location_index = int(index)
        request.original_query = search.original_text
        future = self._location_client.call_async(request)
        return self._wait_for_future(future, self.location_timeout_sec)

    def _request_lost_found_vlm(
        self,
        search: SearchRequest,
        location: str,
        image_path: str,
        camera_name: str,
    ) -> Any | None:
        if not self._lost_found_vlm_client.wait_for_service(timeout_sec=self.service_wait_timeout_sec):
            self.get_logger().warn(f"Lost-found VLM service is not available: {self.lost_found_vlm_service}")
            return None

        request = LostFoundVlmCheck.Request()
        request.object_name = search.object_name
        request.location_name = location
        request.image_path = image_path
        request.camera_name = camera_name or self.default_camera_name
        request.confidence_threshold = float(self.found_confidence_threshold)
        future = self._lost_found_vlm_client.call_async(request)
        return self._wait_for_future(future, self.vlm_timeout_sec)

    @staticmethod
    def _wait_for_future(future: Future, timeout_sec: float) -> Any | None:
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            if future.done():
                if future.exception() is not None:
                    raise future.exception()
                return future.result()
            time.sleep(0.05)
        return None

    def _build_final_reply(self, search: SearchRequest, results: list[LocationResult]) -> tuple[str, str]:
        found_result = next((result for result in results if result.success and result.found), None)
        last_image = ""
        for result in results:
            if result.image_path:
                last_image = result.image_path

        if found_result is not None:
            result_image_path = found_result.annotated_image_path or found_result.image_path
            if results.index(found_result) == 0:
                return f"I found your {search.object_name} on the {found_result.location}.", result_image_path
            previous = [result.location for result in results[: results.index(found_result)] if result.success]
            if previous:
                return (
                    f"I didn't see your {search.object_name} on the {self._join_locations(previous)}, "
                    f"but I found it on the {found_result.location}.",
                    result_image_path,
                )
            return f"I found your {search.object_name} on the {found_result.location}.", result_image_path

        checked = [result.location for result in results if result.success]
        failed = [result.location for result in results if not result.success]
        if checked and failed:
            return (
                f"I couldn't find your {search.object_name} on the {self._join_locations(checked)}. "
                f"I could not complete the check for the {self._join_locations(failed)}.",
                last_image,
            )
        if checked:
            return f"I couldn't find your {search.object_name} on the {self._join_locations(checked)}.", last_image
        if failed:
            return (
                f"I could not complete the search for your {search.object_name} on the "
                f"{self._join_locations(failed)}.",
                last_image,
            )
        return f"I could not complete the search for your {search.object_name}.", last_image

    @staticmethod
    def _join_locations(locations: list[str]) -> str:
        if not locations:
            return ""
        if len(locations) == 1:
            return locations[0]
        return " or ".join([", ".join(locations[:-1]), locations[-1]]) if len(locations) > 2 else " or ".join(locations)

    @staticmethod
    def _build_room_progress_text(search: SearchRequest, location: str, index: int, total: int) -> str:
        if total <= 1:
            return f"I am going to the {location} now to look for your {search.object_name}."
        if index == 0:
            return f"I am going to the {location} first to look for your {search.object_name}."
        return f"I am going to the {location} next to keep looking for your {search.object_name}."

    @staticmethod
    def _build_room_result_text(search: SearchRequest, result: LocationResult) -> str:
        if not result.success:
            return (
                f"I could not complete the check in the {result.location}: "
                f"{result.error or 'robot step failed.'}"
            )
        if result.found:
            return f"I found your {search.object_name} in the {result.location}."
        return f"I checked the {result.location}. I did not find your {search.object_name} there."

    @staticmethod
    def _extract_bbox_2d(data_text: str) -> tuple[float, float, float, float] | None:
        if not data_text.strip():
            return None
        try:
            parsed = json.loads(data_text)
        except Exception:
            return None
        if not isinstance(parsed, dict):
            return None
        entities = parsed.get("entities")
        if not isinstance(entities, dict):
            return None
        bbox = entities.get("bbox_2d")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return None
        try:
            x1, y1, x2, y2 = [float(value) for value in bbox]
        except (TypeError, ValueError):
            return None
        if not (0.0 <= x1 <= 1000.0 and 0.0 <= y1 <= 1000.0 and 0.0 <= x2 <= 1000.0 and 0.0 <= y2 <= 1000.0):
            return None
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def _annotate_image_from_data_text(self, image_path: str, data_text: str, object_name: str) -> str:
        bbox = self._extract_bbox_2d(data_text)
        if bbox is None or not image_path.strip():
            return image_path

        path = Path(image_path).expanduser()
        if not path.is_file():
            return image_path

        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            return image_path

        image_h, image_w = image.shape[:2]
        x1n, y1n, x2n, y2n = bbox
        x1 = int(round((x1n / 1000.0) * (image_w - 1)))
        y1 = int(round((y1n / 1000.0) * (image_h - 1)))
        x2 = int(round((x2n / 1000.0) * (image_w - 1)))
        y2 = int(round((y2n / 1000.0) * (image_h - 1)))
        x1 = max(0, min(image_w - 1, x1))
        y1 = max(0, min(image_h - 1, y1))
        x2 = max(0, min(image_w - 1, x2))
        y2 = max(0, min(image_h - 1, y2))
        if x2 <= x1 or y2 <= y1:
            return image_path

        color = (40, 220, 40)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        label = (object_name or "object").strip() or "object"
        cv2.putText(
            image,
            label,
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )

        annotated_path = path.with_name(f"{path.stem}_bbox{path.suffix}")
        if cv2.imwrite(str(annotated_path), image):
            return str(annotated_path)
        return image_path

    def destroy_node(self) -> bool:
        self._stop_event.set()
        if self._poll_thread.is_alive():
            self._poll_thread.join(timeout=2.0)
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = TelegramLostFoundNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()

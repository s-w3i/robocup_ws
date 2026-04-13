#!/usr/bin/env python3
"""Dual-mode home assistant orchestrator for chat and simple robot actions."""

from __future__ import annotations

import json
import os
import re
import select
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any

import requests
import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import Bool
from std_srvs.srv import Trigger

from coqui_tts_interfaces.action import SpeakText
from coqui_tts_interfaces.srv import RobotStatus
from vlm_interfaces.srv import CaptureImage, VlmQuery
from yoloe_detection_interfaces.srv import DetectObjectPrompt


DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
VALID_STATUSES = ("sleep", "listening", "idle", "thinking", "operating")
VALID_INTENTS = {
    "chat",
    "take_picture",
    "describe_scene",
    "detect_object",
    "end_session",
    "unknown",
}
INTENT_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {
            "type": "string",
            "enum": sorted(VALID_INTENTS),
        },
        "reply_text": {"type": "string"},
        "camera_name": {"type": "string"},
        "object_prompt": {"type": "string"},
    },
    "required": ["intent", "reply_text", "camera_name", "object_prompt"],
}
CHAT_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "assistant_reply": {"type": "string"},
        "end_session": {"type": "boolean"},
        "end_reason": {"type": ["string", "null"]},
    },
    "required": ["assistant_reply", "end_session", "end_reason"],
}
DEFAULT_CHAT_SYSTEM_PROMPT = (
    "You are EVA, a helpful home service robot. Keep answers short and clear for speech output. "
    "Always respond in English. Set end_session=true when the user clearly wants to end the conversation "
    "(for example: bye, goodbye, that's all, no more, stop chatting, go to sleep). "
    "When end_session=true, provide a short polite closing sentence in assistant_reply."
)
DEFAULT_INTENT_SYSTEM_PROMPT = (
    "You are the intent router for a home service robot. Output JSON only.\n"
    "Choose exactly one intent from: chat, take_picture, describe_scene, detect_object, end_session, unknown.\n"
    "- chat: greetings, small talk, general spoken questions, or anything answerable without using a robot action.\n"
    "- take_picture: the user asks to take, capture, or save a photo, image, or snapshot.\n"
    "- describe_scene: the user asks what the robot sees or asks for a scene description.\n"
    "- detect_object: the user asks the robot to find, detect, or look for a specific object.\n"
    "- end_session: the user wants to stop the conversation.\n"
    "- unknown: only when the command is too ambiguous to classify.\n"
    "Set reply_text to a short clarification only when detect_object is missing the target object, "
    "or when end_session needs a short goodbye. Otherwise leave reply_text empty.\n"
    "Never claim that an action already succeeded.\n"
    "If the user mentions a camera like camera0 or camera, put it in camera_name. Otherwise leave camera_name empty.\n"
    "Set object_prompt to the object name for detect_object, otherwise leave it empty."
)
SCENE_DESCRIPTION_PROMPT = """You are a home service robot describing the current camera view.
Return JSON only.

Set speech_text to one short sentence the robot should say aloud.

Set data_text to a JSON object with exactly these top-level keys:
- task
- reason
- complete
- entities

Set task to "describe_scene".
Set reason to a short explanation.
Set complete to true when the image can be described.

Set entities to a JSON object with these keys:
- summary
- visible_people
- main_objects
- notable_details

Use short factual values.
summary must be a short string.
visible_people should be a number or null.
main_objects should be a short list of visible object names.
notable_details should be a short list of notable details.
Describe only what is visible and do not guess beyond the image.
"""


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class IntentDecision:
    intent: str
    reply_text: str = ""
    camera_name: str = ""
    object_prompt: str = ""


class HomeAssistantOrchestratorNode(Node):
    def __init__(self) -> None:
        super().__init__("home_assistant_orchestrator_node")

        self.declare_parameter("awake_topic", "/awake")
        self.declare_parameter("awake_greeting_done_topic", "/awake_greeting_done")
        self.declare_parameter("get_command_service", "/get_command")
        self.declare_parameter("robot_status_service", "/robot_status")
        self.declare_parameter("speak_action_name", "/coqui_tts/speak")
        self.declare_parameter("vlm_query_service", "/vlm/query")
        self.declare_parameter("detect_object_service", "/yoloe/detect_prompt")
        self.declare_parameter("capture_image_service", "/camera/capture")
        self.declare_parameter("default_camera_name", "camera0")
        self.declare_parameter("camera_names", ["camera0"])
        self.declare_parameter("camera_names_csv", "")
        self.declare_parameter("get_command_fail_window_sec", 10.0)
        self.declare_parameter("get_command_retry_delay_sec", 0.2)
        self.declare_parameter("debug_text_input_mode", False)
        self.declare_parameter("debug_text_input_prompt", "You")
        self.declare_parameter("service_response_timeout_sec", 30.0)
        self.declare_parameter("ollama_base_url", os.environ.get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL))
        self.declare_parameter("chat_model", os.environ.get("TEXT_MODEL", "qwen3.5:9b"))
        self.declare_parameter("ollama_keep_alive", os.environ.get("OLLAMA_KEEP_ALIVE", "30m"))
        self.declare_parameter("ollama_timeout_sec", 180.0)
        self.declare_parameter("ollama_temperature", 0.2)
        self.declare_parameter("ollama_num_ctx", 2048)
        self.declare_parameter("ollama_num_batch", 128)
        self.declare_parameter("auto_start_ollama", True)
        self.declare_parameter("ollama_start_timeout_sec", 20.0)
        self.declare_parameter("max_history_messages", 12)
        self.declare_parameter("chat_system_prompt", DEFAULT_CHAT_SYSTEM_PROMPT)
        self.declare_parameter("intent_system_prompt", DEFAULT_INTENT_SYSTEM_PROMPT)
        self.declare_parameter("fallback_error_reply", "Sorry, I am having trouble right now.")
        self.declare_parameter("trace_logging", True)

        self.awake_topic = str(self.get_parameter("awake_topic").value)
        self.awake_greeting_done_topic = str(self.get_parameter("awake_greeting_done_topic").value)
        self.get_command_service = str(self.get_parameter("get_command_service").value)
        self.robot_status_service = str(self.get_parameter("robot_status_service").value)
        self.speak_action_name = str(self.get_parameter("speak_action_name").value)
        self.vlm_query_service = str(self.get_parameter("vlm_query_service").value)
        self.detect_object_service = str(self.get_parameter("detect_object_service").value)
        self.capture_image_service = str(self.get_parameter("capture_image_service").value)
        self.default_camera_name = (
            str(self.get_parameter("default_camera_name").value).strip() or "camera0"
        )
        self.camera_names = [str(name).strip() for name in self.get_parameter("camera_names").value]
        camera_names_csv = str(self.get_parameter("camera_names_csv").value).strip()
        self.get_command_fail_window_sec = float(self.get_parameter("get_command_fail_window_sec").value)
        self.get_command_retry_delay_sec = float(self.get_parameter("get_command_retry_delay_sec").value)
        self.debug_text_input_mode = bool(self.get_parameter("debug_text_input_mode").value)
        self.debug_text_input_prompt = str(self.get_parameter("debug_text_input_prompt").value).strip() or "You"
        self.service_response_timeout_sec = float(self.get_parameter("service_response_timeout_sec").value)
        self.ollama_base_url = str(self.get_parameter("ollama_base_url").value).rstrip("/")
        self.chat_model = str(self.get_parameter("chat_model").value)
        self.ollama_keep_alive = str(self.get_parameter("ollama_keep_alive").value)
        self.ollama_timeout_sec = float(self.get_parameter("ollama_timeout_sec").value)
        self.ollama_temperature = float(self.get_parameter("ollama_temperature").value)
        self.ollama_num_ctx = int(self.get_parameter("ollama_num_ctx").value)
        self.ollama_num_batch = int(self.get_parameter("ollama_num_batch").value)
        self.auto_start_ollama = bool(self.get_parameter("auto_start_ollama").value)
        self.ollama_start_timeout_sec = float(self.get_parameter("ollama_start_timeout_sec").value)
        self.max_history_messages = max(2, int(self.get_parameter("max_history_messages").value))
        self.chat_system_prompt = str(self.get_parameter("chat_system_prompt").value).strip()
        self.intent_system_prompt = str(self.get_parameter("intent_system_prompt").value).strip()
        self.fallback_error_reply = str(self.get_parameter("fallback_error_reply").value).strip()
        self.trace_logging = bool(self.get_parameter("trace_logging").value)

        if camera_names_csv:
            self.camera_names = [part.strip() for part in camera_names_csv.split(",") if part.strip()]
        if self.default_camera_name not in self.camera_names:
            self.camera_names.append(self.default_camera_name)
        self._camera_names_by_key = {
            self._normalize_camera_name(name): name for name in self.camera_names if name
        }

        self._callback_group = ReentrantCallbackGroup()
        self._get_command_client = self.create_client(
            Trigger,
            self.get_command_service,
            callback_group=self._callback_group,
        )
        self._robot_status_client = self.create_client(
            RobotStatus,
            self.robot_status_service,
            callback_group=self._callback_group,
        )
        self._speak_action_client = ActionClient(
            self,
            SpeakText,
            self.speak_action_name,
            callback_group=self._callback_group,
        )
        self._vlm_client = self.create_client(
            VlmQuery,
            self.vlm_query_service,
            callback_group=self._callback_group,
        )
        self._detect_object_client = self.create_client(
            DetectObjectPrompt,
            self.detect_object_service,
            callback_group=self._callback_group,
        )
        self._capture_image_client = self.create_client(
            CaptureImage,
            self.capture_image_service,
            callback_group=self._callback_group,
        )

        self._shutdown_event = threading.Event()
        self._session_cancel_event = threading.Event()
        self._state_lock = threading.Lock()
        self._session_active = False
        self._pending_awake = False
        self._session_thread: threading.Thread | None = None
        self._debug_tty_path = "/dev/tty"

        status_qos = QoSProfile(depth=1)
        status_qos.reliability = QoSReliabilityPolicy.RELIABLE
        status_qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        self._awake_sub = self.create_subscription(
            Bool,
            self.awake_topic,
            self._awake_callback,
            status_qos,
        )
        self._awake_greeting_done_sub = self.create_subscription(
            Bool,
            self.awake_greeting_done_topic,
            self._awake_greeting_done_callback,
            status_qos,
        )

        if self.auto_start_ollama:
            try:
                self._ensure_ollama_running()
            except Exception as exc:
                self.get_logger().error(f"Failed to start/reach Ollama: {exc}")
        elif not self._ollama_ready(timeout=1.0):
            self.get_logger().warn(
                f"Ollama is not reachable at {self.ollama_base_url}. "
                "Assistant requests will fail until it is available."
            )

        self.get_logger().info(
            "Home assistant orchestrator ready. "
            f"chat_model='{self.chat_model}' get_command={self.get_command_service} "
            f"vlm_service={self.vlm_query_service} detect_service={self.detect_object_service} "
            f"capture_service={self.capture_image_service}"
        )
        self.get_logger().info(
            f"debug_text_input_mode={self.debug_text_input_mode} prompt='{self.debug_text_input_prompt}'"
        )

        if self.debug_text_input_mode:
            self.get_logger().info("Debug text input mode enabled; starting a session immediately.")
            self._start_session_thread(trigger_reason="debug")
        else:
            self.get_logger().info(
                f"Session start trigger: {self.awake_topic}=true + {self.awake_greeting_done_topic}=true"
            )

    def destroy_node(self) -> bool:
        self._shutdown_event.set()
        self._session_cancel_event.set()
        return super().destroy_node()

    def _trace(self, message: str) -> None:
        if self.trace_logging:
            self.get_logger().info(f"[trace] {message}")

    def _awake_callback(self, msg: Bool) -> None:
        if self.debug_text_input_mode:
            return

        value = bool(msg.data)
        with self._state_lock:
            if not value:
                self._pending_awake = False
                if self._session_active:
                    self._session_cancel_event.set()
                return
            self._pending_awake = True
        self.get_logger().info(
            "Received /awake=true. Waiting for awake greeting completion to start assistant session."
        )

    def _awake_greeting_done_callback(self, msg: Bool) -> None:
        if self.debug_text_input_mode or not bool(msg.data):
            return

        with self._state_lock:
            if not self._pending_awake or self._session_active:
                return
        self._start_session_thread(trigger_reason="wake")

    def _start_session_thread(self, trigger_reason: str) -> None:
        with self._state_lock:
            if self._session_active:
                return
            self._pending_awake = False
            self._session_active = True
            self._session_cancel_event.clear()
            self._session_thread = threading.Thread(
                target=self._session_loop,
                args=(trigger_reason,),
                daemon=True,
            )
            self._session_thread.start()

    def _session_loop(self, trigger_reason: str) -> None:
        end_reason = "session complete"
        history: list[dict[str, str]] = []
        if self.chat_system_prompt:
            history.append({"role": "system", "content": self.chat_system_prompt})

        fail_window_start: float | None = None
        self.get_logger().info(f"Assistant session started. trigger={trigger_reason}")
        self._set_robot_status("idle")
        try:
            while not self._shutdown_event.is_set():
                if self._session_cancel_event.is_set():
                    end_reason = "Canceled by /awake=false."
                    break

                if not self.debug_text_input_mode:
                    self._set_robot_status("listening")
                call_start_mono = time.monotonic()
                ok, user_text, fail_message = self._call_get_command()
                now_mono = time.monotonic()
                if not ok:
                    if self.debug_text_input_mode:
                        end_reason = fail_message or "Debug input ended."
                        break
                    if fail_window_start is None:
                        fail_window_start = call_start_mono
                    silent_for = now_mono - fail_window_start
                    self.get_logger().info(
                        f"get_command failed: {fail_message} "
                        f"(silent_for={silent_for:.1f}/{self.get_command_fail_window_sec:.1f}s)"
                    )
                    if silent_for >= self.get_command_fail_window_sec:
                        end_reason = (
                            f"No user command received for {self.get_command_fail_window_sec:.1f}s."
                        )
                        break
                    time.sleep(max(0.0, self.get_command_retry_delay_sec))
                    continue

                fail_window_start = None
                cleaned_user = user_text.strip()
                if not cleaned_user:
                    continue

                self.get_logger().info(f"User: {cleaned_user}")
                history.append({"role": "user", "content": cleaned_user})
                self._trim_history(history)

                spoken_reply = ""
                should_end = False
                self._set_robot_status("thinking")
                try:
                    decision = self._classify_intent(cleaned_user)
                    self._trace(
                        "intent classified | "
                        f"intent={decision.intent!r} camera={decision.camera_name!r} "
                        f"object_prompt={decision.object_prompt!r} reply_text={decision.reply_text!r}"
                    )

                    if decision.intent in {"chat", "unknown"}:
                        spoken_reply, should_end, llm_reason = self._chat_with_ollama(history)
                        if should_end:
                            end_reason = llm_reason or "Chat model ended the session."
                    elif decision.intent == "end_session":
                        spoken_reply = decision.reply_text or "Goodbye."
                        should_end = True
                        end_reason = "User ended the session."
                    elif decision.intent == "take_picture":
                        self._set_robot_status("operating")
                        spoken_reply = self._handle_take_picture(decision)
                    elif decision.intent == "describe_scene":
                        self._set_robot_status("operating")
                        spoken_reply = self._handle_describe_scene(cleaned_user, decision)
                    elif decision.intent == "detect_object":
                        if not decision.object_prompt.strip():
                            spoken_reply = decision.reply_text or "What object should I look for?"
                        else:
                            self._set_robot_status("operating")
                            spoken_reply = self._handle_detect_object(decision)
                    else:
                        spoken_reply, should_end, llm_reason = self._chat_with_ollama(history)
                        if should_end:
                            end_reason = llm_reason or "Chat model ended the session."
                except Exception as exc:
                    self.get_logger().error(f"Assistant handling failed: {exc}")
                    spoken_reply = self.fallback_error_reply or "Sorry, I am having trouble right now."
                finally:
                    if not self._shutdown_event.is_set() and not self._session_cancel_event.is_set():
                        self._set_robot_status("idle")

                if spoken_reply:
                    history.append({"role": "assistant", "content": spoken_reply})
                    self._trim_history(history)
                    speak_ok, speak_message = self._speak_text(spoken_reply)
                    if not speak_ok:
                        end_reason = f"SpeakText action failed: {speak_message}"
                        break

                if should_end:
                    break
        finally:
            self.get_logger().info(f"Assistant session ending. reason={end_reason}")
            self._set_robot_status("sleep")
            with self._state_lock:
                self._session_active = False
                self._pending_awake = False
                self._session_cancel_event.clear()
            self.get_logger().info("Assistant session ended.")

    def _trim_history(self, history: list[dict[str, str]]) -> None:
        if not history:
            return
        if history[0].get("role") == "system":
            system = history[0]
            tail = history[1:]
            if len(tail) <= self.max_history_messages:
                return
            history[:] = [system] + tail[-self.max_history_messages :]
            return
        if len(history) > self.max_history_messages:
            history[:] = history[-self.max_history_messages :]

    def _classify_intent(self, user_text: str) -> IntentDecision:
        payload: dict[str, Any] = {
            "model": self.chat_model,
            "messages": [
                {"role": "system", "content": self.intent_system_prompt},
                {"role": "user", "content": user_text},
            ],
            "stream": False,
            "keep_alive": self.ollama_keep_alive,
            "format": INTENT_RESPONSE_SCHEMA,
            "options": {
                "temperature": 0.0,
                "num_ctx": min(1024, self.ollama_num_ctx),
                "num_batch": self.ollama_num_batch,
                "num_predict": 128,
            },
            "think": False,
        }

        result = requests.post(
            f"{self.ollama_base_url}/api/chat",
            json=payload,
            timeout=max(1.0, self.ollama_timeout_sec),
        )
        result.raise_for_status()
        content = (result.json().get("message", {}).get("content") or "").strip()
        if not content:
            return IntentDecision(intent="chat")

        try:
            parsed = self._parse_json_relaxed(content)
        except Exception as exc:
            self.get_logger().warn(f"Intent parse failed, falling back to chat: {exc}")
            return IntentDecision(intent="chat")

        intent = self._normalize_intent(parsed.get("intent", "unknown"))
        return IntentDecision(
            intent=intent,
            reply_text=self._sanitize_spoken_reply(parsed.get("reply_text", "")),
            camera_name=str(parsed.get("camera_name", "") or "").strip(),
            object_prompt=str(parsed.get("object_prompt", "") or "").strip(),
        )

    def _chat_with_ollama(
        self, history: list[dict[str, str]]
    ) -> tuple[str, bool, str]:
        payload: dict[str, Any] = {
            "model": self.chat_model,
            "messages": history,
            "stream": False,
            "keep_alive": self.ollama_keep_alive,
            "format": CHAT_RESPONSE_SCHEMA,
            "options": {
                "temperature": self.ollama_temperature,
                "num_ctx": self.ollama_num_ctx,
                "num_batch": self.ollama_num_batch,
            },
            "think": False,
        }

        result = requests.post(
            f"{self.ollama_base_url}/api/chat",
            json=payload,
            timeout=max(1.0, self.ollama_timeout_sec),
        )
        result.raise_for_status()
        content = (result.json().get("message", {}).get("content") or "").strip()
        if not content:
            return "Could you repeat that, please?", False, ""

        try:
            parsed = self._parse_json_relaxed(content)
            reply = self._sanitize_spoken_reply(parsed.get("assistant_reply", ""))
            should_end = _to_bool(parsed.get("end_session"))
            end_reason_raw = parsed.get("end_reason")
            end_reason = "" if end_reason_raw is None else str(end_reason_raw).strip()
            if not reply and not should_end:
                reply = "Could you repeat that, please?"
            return reply, should_end, end_reason
        except Exception:
            should_end = bool(
                re.search(r"(?im)\bend_session\s*[:=]\s*(true|1|yes|on)\b", content)
            )
            reply = self._sanitize_spoken_reply(content)
            if not reply and not should_end:
                reply = "Could you repeat that, please?"
            return reply, should_end, ""

    def _handle_take_picture(self, decision: IntentDecision) -> str:
        camera_name = self._resolve_camera_name(decision.camera_name)
        if not self._capture_image_client.wait_for_service(timeout_sec=1.0):
            return "I cannot take a picture right now because the camera service is not ready."

        request = CaptureImage.Request()
        request.camera_name = camera_name
        request.save_dir = ""
        request.file_prefix = "snapshot"

        future = self._capture_image_client.call_async(request)
        ok, response, error_text = self._wait_for_future(future, self.service_response_timeout_sec)
        if not ok:
            return f"I couldn't take a picture because the camera service failed: {error_text}"
        if response is None:
            return "I couldn't take a picture because the camera service returned no response."
        if not response.success:
            return f"I couldn't take a picture: {response.message}"

        basename = os.path.basename(str(response.saved_image_path).strip())
        camera_used = str(response.camera_used).strip() or camera_name
        if basename:
            return f"I saved a picture from {camera_used} as {basename}."
        return f"I saved a picture from {camera_used}."

    def _handle_describe_scene(self, user_text: str, decision: IntentDecision) -> str:
        camera_name = self._resolve_camera_name(decision.camera_name)
        if not self._vlm_client.wait_for_service(timeout_sec=1.0):
            return "I cannot describe the scene right now because the vision service is not ready."

        request = VlmQuery.Request()
        request.need_image = True
        request.camera_name = camera_name
        request.prompt = SCENE_DESCRIPTION_PROMPT
        request.reasoning_mode = "fast"
        request.user_input = user_text
        request.request_profile = "vision_strict"
        request.max_retry_count = 0
        request.json_repair_mode = 1
        request.num_predict_override = 0
        request.timeout_sec_override = 0.0

        future = self._vlm_client.call_async(request)
        ok, response, error_text = self._wait_for_future(future, self.service_response_timeout_sec)
        if not ok:
            return f"I couldn't describe the scene because the vision service failed: {error_text}"
        if response is None:
            return "I couldn't describe the scene because the vision service returned no response."
        if not response.success:
            return f"I couldn't describe the scene: {response.message}"

        speech_text = str(response.speech_text).strip()
        if speech_text:
            return speech_text
        return "I couldn't describe the scene clearly."

    def _handle_detect_object(self, decision: IntentDecision) -> str:
        camera_name = self._resolve_camera_name(decision.camera_name)
        object_prompt = decision.object_prompt.strip()
        if not self._detect_object_client.wait_for_service(timeout_sec=1.0):
            return "I cannot look for objects right now because the detection service is not ready."

        request = DetectObjectPrompt.Request()
        request.prompt_text = object_prompt
        request.save_image = True
        request.camera_name = camera_name

        future = self._detect_object_client.call_async(request)
        ok, response, error_text = self._wait_for_future(future, self.service_response_timeout_sec)
        if not ok:
            return f"I couldn't look for {object_prompt} because detection failed: {error_text}"
        if response is None:
            return f"I couldn't look for {object_prompt} because detection returned no response."
        if not response.success:
            return f"I couldn't look for {object_prompt}: {response.message}"

        saved_suffix = ""
        saved_basename = os.path.basename(str(response.saved_image_path).strip())
        if saved_basename:
            saved_suffix = f" I saved the annotated image as {saved_basename}."

        if int(response.detections_in_frame) > 0:
            count = int(response.detections_in_frame)
            match_word = "match" if count == 1 else "matches"
            return f"I found {count} {match_word} for {object_prompt}.{saved_suffix}"
        return f"I could not find {object_prompt}.{saved_suffix}"

    def _resolve_camera_name(self, requested_camera_name: str) -> str:
        stripped = str(requested_camera_name).strip()
        if not stripped:
            return self.default_camera_name

        normalized = self._normalize_camera_name(stripped)
        resolved = self._camera_names_by_key.get(normalized)
        if resolved:
            return resolved

        self.get_logger().warn(
            f"Unknown camera '{stripped}', falling back to default '{self.default_camera_name}'."
        )
        return self.default_camera_name

    def _call_get_command(self) -> tuple[bool, str, str]:
        if self.debug_text_input_mode:
            return self._call_debug_text_command()

        if not self._get_command_client.wait_for_service(timeout_sec=0.5):
            return False, "", f"Service '{self.get_command_service}' not ready."

        request = Trigger.Request()
        future = self._get_command_client.call_async(request)
        ok, response, error_text = self._wait_for_future(
            future,
            self.service_response_timeout_sec,
            cancel_on_session_stop=False,
        )
        if not ok:
            return False, "", error_text
        if response is None:
            return False, "", "No response from get_command service."

        text = str(response.message).strip()
        if response.success and text:
            return True, text, ""
        if response.success and not text:
            return False, "", "Empty speech transcription."
        return False, "", text or "get_command failed."

    def _call_debug_text_command(self) -> tuple[bool, str, str]:
        prompt = f"{self.debug_text_input_prompt}: "
        self.get_logger().info("Debug text input mode active. Waiting for terminal input.")

        read_stream = None
        write_stream = None
        close_read = False
        close_write = False
        try:
            if os.path.exists(self._debug_tty_path):
                read_stream = open(self._debug_tty_path, "r", encoding="utf-8", buffering=1)
                write_stream = open(self._debug_tty_path, "w", encoding="utf-8", buffering=1)
                close_read = True
                close_write = True
            elif sys.stdin is not None and not sys.stdin.closed:
                read_stream = sys.stdin
                write_stream = sys.stdout if sys.stdout is not None and not sys.stdout.closed else None
            else:
                return False, "", "No interactive terminal available for debug text input."

            prompt_written = False
            while not self._shutdown_event.is_set() and not self._session_cancel_event.is_set():
                try:
                    if write_stream is not None and not prompt_written:
                        write_stream.write(prompt)
                        write_stream.flush()
                        prompt_written = True
                except Exception:
                    pass

                ready, _, _ = select.select([read_stream], [], [], 0.2)
                if not ready:
                    continue

                line = read_stream.readline()
                if line == "":
                    return False, "", "Debug text input closed."

                text = line.strip()
                if not text:
                    prompt_written = False
                    continue
                return True, text, ""

            return False, "", "Canceled."
        except Exception as exc:
            return False, "", f"Debug text input failed: {exc}"
        finally:
            if close_read and read_stream is not None:
                try:
                    read_stream.close()
                except Exception:
                    pass
            if close_write and write_stream is not None:
                try:
                    write_stream.close()
                except Exception:
                    pass

    def _speak_text(self, text: str) -> tuple[bool, str]:
        cleaned = str(text).strip()
        if not cleaned:
            return True, "Nothing to speak."

        if not self._speak_action_client.wait_for_server(timeout_sec=0.8):
            return False, f"Speak action server '{self.speak_action_name}' not ready."

        goal = SpeakText.Goal()
        goal.text = cleaned
        send_goal_future = self._speak_action_client.send_goal_async(goal)
        ok, goal_handle, error_text = self._wait_for_future(send_goal_future, self.service_response_timeout_sec)
        if not ok:
            return False, f"Failed to send SpeakText goal: {error_text}"
        if goal_handle is None or not goal_handle.accepted:
            return False, "SpeakText goal rejected."

        result_future = goal_handle.get_result_async()
        ok, result_wrap, error_text = self._wait_for_future(result_future, self.service_response_timeout_sec)
        if not ok:
            return False, f"SpeakText result wait failed: {error_text}"
        if result_wrap is None:
            return False, "SpeakText returned no result."

        result = result_wrap.result
        if result.success:
            return True, result.message
        return False, result.message

    def _set_robot_status(self, target: str) -> bool:
        normalized = str(target).strip().lower()
        if normalized not in VALID_STATUSES:
            return False
        if not self._robot_status_client.wait_for_service(timeout_sec=0.8):
            self.get_logger().warn(
                f"Robot status service '{self.robot_status_service}' not ready; cannot set '{normalized}'."
            )
            return False

        request = RobotStatus.Request()
        request.status = normalized
        future = self._robot_status_client.call_async(request)
        ok, response, error_text = self._wait_for_future(
            future,
            self.service_response_timeout_sec,
            cancel_on_session_stop=False,
        )
        if not ok:
            self.get_logger().warn(f"Failed to set robot status '{normalized}': {error_text}")
            return False
        if response is None:
            self.get_logger().warn(f"Failed to set robot status '{normalized}': empty response.")
            return False
        if not response.success:
            self.get_logger().warn(
                f"Robot status service rejected '{normalized}': {response.message}"
            )
            return False
        return True

    def _wait_for_future(
        self,
        future,
        timeout_sec: float,
        *,
        cancel_on_session_stop: bool = True,
    ) -> tuple[bool, Any, str]:
        event = threading.Event()
        holder: dict[str, Any] = {}

        def _done(fut) -> None:
            holder["future"] = fut
            event.set()

        future.add_done_callback(_done)
        start = time.monotonic()
        timeout_sec = max(0.1, float(timeout_sec))
        while not event.wait(timeout=0.1):
            if self._shutdown_event.is_set():
                return False, None, "Canceled."
            if cancel_on_session_stop and self._session_cancel_event.is_set():
                return False, None, "Canceled."
            if (time.monotonic() - start) >= timeout_sec:
                return False, None, "timeout"

        fut = holder.get("future", future)
        exc = fut.exception()
        if exc is not None:
            return False, None, str(exc)
        return True, fut.result(), ""

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
            subprocess.Popen(  # pylint: disable=consider-using-with
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"Unable to start ollama serve: {exc}") from exc

        deadline = time.time() + max(1.0, self.ollama_start_timeout_sec)
        while time.time() < deadline:
            if self._ollama_ready(timeout=1.0):
                return
            time.sleep(0.5)
        raise RuntimeError(f"Ollama not reachable at {self.ollama_base_url}")

    @staticmethod
    def _parse_json_relaxed(text: str) -> dict[str, Any]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            if not match:
                raise
            return json.loads(match.group(0))

    @staticmethod
    def _sanitize_spoken_reply(text: Any) -> str:
        cleaned = str(text).strip()
        if not cleaned:
            return ""

        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,:\n\t")
        return cleaned

    @staticmethod
    def _normalize_intent(value: Any) -> str:
        normalized = str(value).strip().lower()
        if normalized in VALID_INTENTS:
            return normalized
        return "unknown"

    @staticmethod
    def _normalize_camera_name(camera_name: str) -> str:
        return camera_name.strip().lower()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = HomeAssistantOrchestratorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

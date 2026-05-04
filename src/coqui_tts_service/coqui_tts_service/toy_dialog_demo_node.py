#!/usr/bin/env python3
"""Demo dialog node for child toy/animal visual Q&A."""

from __future__ import annotations

import base64
import json
import os
import re
import select
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any

import cv2
import requests
import rclpy
from coqui_tts_interfaces.srv import RobotStatus
from cv_bridge import CvBridge, CvBridgeError
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from std_srvs.srv import SetBool, Trigger

from coqui_tts_interfaces.action import SpeakText


DEFAULT_OPENAI_API_URL = "https://api.openai.com/v1/responses"
DEFAULT_OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
DEFAULT_OPENAI_MODEL = "gpt-5.5"
VALID_STATUSES = ("sleep", "listening", "idle", "thinking", "operating")
VALID_INTENTS = ("chat", "vision", "end_session")
DEFAULT_INTENT_MAX_TOKENS = 128
DEFAULT_CHAT_MAX_TOKENS = 220
DEFAULT_RETRY_MAX_TOKENS = 480

VISION_PROMPT = """You are a friendly robot answering a child's question about the current camera image.
Return JSON only.

The child may be holding a toy, toy animal, object, or picture in front of the robot.
Answer the user's question from the image. If the user asks "what is this animal", identify the animal or toy animal if visible.
Keep speech_text short, warm, and easy for a child to understand.
If the image is unclear, say the robot cannot see it clearly and ask the child to hold it closer.

Set speech_text to one short sentence to speak aloud.
Set data_text to a JSON object with exactly these top-level keys:
- task
- reason
- complete
- entities

Set task to "visual_question".
Set reason to a short explanation based only on the image.
Set complete to true when the question was answered from the image.

Set entities to a JSON object with exactly these keys:
- answer
- subject
- visible_objects
- confidence

Use only what is visible in the image.
"""


@dataclass
class CameraFrame:
    image_bgr: Any
    stamp_ns: int
    received_monotonic: float


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class IntentDecision:
    intent: str
    reason: str = ""


class ToyDialogDemoNode(Node):
    def __init__(self) -> None:
        super().__init__("toy_dialog_demo_node")

        self.declare_parameter("enabled", True)
        self.declare_parameter("sim", False)
        self.declare_parameter("toggle_service", "/toy_dialog_demo/set_enabled")
        self.declare_parameter("awake_topic", "/awake")
        self.declare_parameter("awake_greeting_done_topic", "/awake_greeting_done")
        self.declare_parameter("get_command_service", "/get_command")
        self.declare_parameter("robot_status_service", "/robot_status")
        self.declare_parameter("speak_action_name", "/coqui_tts/speak")
        self.declare_parameter("camera_names_csv", "camera0,gripper_camera")
        self.declare_parameter("camera_topics_csv", "/camera0/color/image_raw,/gripper_camera/color/image_raw")
        self.declare_parameter("default_camera_name", "camera0")
        self.declare_parameter("image_wait_timeout_sec", 1.5)
        self.declare_parameter("lazy_subscriptions", True)
        self.declare_parameter("subscription_idle_timeout_sec", 3.0)
        self.declare_parameter("subscription_poll_period_sec", 0.5)
        self.declare_parameter("get_command_fail_window_sec", 10.0)
        self.declare_parameter("get_command_retry_delay_sec", 0.2)
        self.declare_parameter("service_response_timeout_sec", 35.0)
        self.declare_parameter("debug_text_input_mode", False)
        self.declare_parameter("debug_text_input_prompt", "You")
        self.declare_parameter("openai_api_url", os.environ.get("OPENAI_API_URL", DEFAULT_OPENAI_API_URL))
        self.declare_parameter("openai_api_key_env", os.environ.get("OPENAI_API_KEY_ENV", DEFAULT_OPENAI_API_KEY_ENV))
        self.declare_parameter("openai_model", os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL))
        self.declare_parameter("openai_image_detail", "auto")
        self.declare_parameter("openai_thinking_effort", "medium")
        self.declare_parameter("openai_timeout_sec", 45.0)
        self.declare_parameter("openai_retry_timeout_sec", 60.0)
        self.declare_parameter("openai_max_retry_count", 1)
        self.declare_parameter("intent_max_output_tokens", DEFAULT_INTENT_MAX_TOKENS)
        self.declare_parameter("chat_max_output_tokens", DEFAULT_CHAT_MAX_TOKENS)
        self.declare_parameter("vision_max_output_tokens", DEFAULT_CHAT_MAX_TOKENS)
        self.declare_parameter("retry_max_output_tokens", DEFAULT_RETRY_MAX_TOKENS)
        self.declare_parameter("vision_recapture_attempts", 2)
        self.declare_parameter("vision_recapture_delay_sec", 0.7)
        self.declare_parameter("vision_min_confidence", 0.35)
        self.declare_parameter("max_session_messages", 16)
        self.declare_parameter("max_run_memory_messages", 40)
        self.declare_parameter("fallback_error_reply", "Sorry, I am having trouble right now.")
        self.declare_parameter("trace_logging", True)

        self.enabled = bool(self.get_parameter("enabled").value)
        self.sim = bool(self.get_parameter("sim").value)
        self.toggle_service = str(self.get_parameter("toggle_service").value)
        self.awake_topic = str(self.get_parameter("awake_topic").value)
        self.awake_greeting_done_topic = str(self.get_parameter("awake_greeting_done_topic").value)
        self.get_command_service = str(self.get_parameter("get_command_service").value)
        self.robot_status_service = str(self.get_parameter("robot_status_service").value)
        self.speak_action_name = str(self.get_parameter("speak_action_name").value)
        self.default_camera_name = str(self.get_parameter("default_camera_name").value).strip() or "camera0"
        camera_names_csv = str(self.get_parameter("camera_names_csv").value).strip()
        camera_topics_csv = str(self.get_parameter("camera_topics_csv").value).strip()
        self.camera_names = [part.strip() for part in camera_names_csv.split(",") if part.strip()]
        if self.default_camera_name not in self.camera_names:
            self.camera_names.insert(0, self.default_camera_name)
        self.camera_topics = [part.strip() for part in camera_topics_csv.split(",") if part.strip()]
        if len(self.camera_topics) != len(self.camera_names):
            raise ValueError("camera_names_csv and camera_topics_csv must have same count")
        self.image_wait_timeout_sec = max(0.1, float(self.get_parameter("image_wait_timeout_sec").value))
        self.lazy_subscriptions = bool(self.get_parameter("lazy_subscriptions").value)
        self.subscription_idle_timeout_sec = max(
            0.0, float(self.get_parameter("subscription_idle_timeout_sec").value)
        )
        self.subscription_poll_period_sec = max(
            0.1, float(self.get_parameter("subscription_poll_period_sec").value)
        )
        self.get_command_fail_window_sec = float(self.get_parameter("get_command_fail_window_sec").value)
        self.get_command_retry_delay_sec = float(self.get_parameter("get_command_retry_delay_sec").value)
        self.service_response_timeout_sec = float(self.get_parameter("service_response_timeout_sec").value)
        self.debug_text_input_mode = bool(self.get_parameter("debug_text_input_mode").value) or self.sim
        self.debug_text_input_prompt = str(self.get_parameter("debug_text_input_prompt").value).strip() or "You"
        self.openai_api_url = str(self.get_parameter("openai_api_url").value).strip() or DEFAULT_OPENAI_API_URL
        self.openai_api_key_env = (
            str(self.get_parameter("openai_api_key_env").value).strip() or DEFAULT_OPENAI_API_KEY_ENV
        )
        self.openai_model = str(self.get_parameter("openai_model").value).strip() or DEFAULT_OPENAI_MODEL
        self.openai_image_detail = str(self.get_parameter("openai_image_detail").value).strip() or "auto"
        self.openai_thinking_effort = str(self.get_parameter("openai_thinking_effort").value).strip() or "medium"
        self.openai_timeout_sec = max(3.0, float(self.get_parameter("openai_timeout_sec").value))
        self.openai_retry_timeout_sec = max(
            self.openai_timeout_sec, float(self.get_parameter("openai_retry_timeout_sec").value)
        )
        self.openai_max_retry_count = max(0, int(self.get_parameter("openai_max_retry_count").value))
        self.intent_max_output_tokens = max(64, int(self.get_parameter("intent_max_output_tokens").value))
        self.chat_max_output_tokens = max(96, int(self.get_parameter("chat_max_output_tokens").value))
        self.vision_max_output_tokens = max(96, int(self.get_parameter("vision_max_output_tokens").value))
        self.retry_max_output_tokens = max(128, int(self.get_parameter("retry_max_output_tokens").value))
        self.vision_recapture_attempts = max(0, int(self.get_parameter("vision_recapture_attempts").value))
        self.vision_recapture_delay_sec = max(0.0, float(self.get_parameter("vision_recapture_delay_sec").value))
        self.vision_min_confidence = max(0.0, min(1.0, float(self.get_parameter("vision_min_confidence").value)))
        self.max_session_messages = max(2, int(self.get_parameter("max_session_messages").value))
        self.max_run_memory_messages = max(2, int(self.get_parameter("max_run_memory_messages").value))
        self.fallback_error_reply = str(self.get_parameter("fallback_error_reply").value).strip()
        self.trace_logging = bool(self.get_parameter("trace_logging").value)

        self._bridge = CvBridge()
        self._callback_group = ReentrantCallbackGroup()
        self._lock = threading.Lock()
        self._shutdown_event = threading.Event()
        self._session_cancel_event = threading.Event()
        self._state_lock = threading.Lock()
        self._session_active = False
        self._pending_awake = False
        self._session_thread: threading.Thread | None = None
        self._debug_tty_path = "/dev/tty"
        self._run_memory: list[dict[str, str]] = []
        self._camera_topics_by_name = dict(zip(self.camera_names, self.camera_topics))
        self._camera_subscriptions: dict[str, Any] = {}
        self._camera_subscription_deadlines: dict[str, float] = {}
        self._frames: dict[str, CameraFrame] = {}

        status_qos = QoSProfile(depth=1)
        status_qos.reliability = QoSReliabilityPolicy.RELIABLE
        status_qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        self.create_subscription(Bool, self.awake_topic, self._awake_callback, status_qos)
        self.create_subscription(
            Bool,
            self.awake_greeting_done_topic,
            self._awake_greeting_done_callback,
            status_qos,
        )
        self.create_timer(self.subscription_poll_period_sec, self._cleanup_idle_subscriptions)
        self.create_service(SetBool, self.toggle_service, self._handle_set_enabled)
        self._get_command_client = self.create_client(Trigger, self.get_command_service, callback_group=self._callback_group)
        self._robot_status_client = self.create_client(RobotStatus, self.robot_status_service, callback_group=self._callback_group)
        self._speak_action_client = ActionClient(self, SpeakText, self.speak_action_name, callback_group=self._callback_group)

        self.get_logger().info(
            "Toy dialog demo ready. "
            f"enabled={self.enabled} sim={self.sim} cameras={self.camera_names} "
            f"toggle_service={self.toggle_service}"
        )

        if self.debug_text_input_mode and self.enabled:
            self._start_session_thread("debug")

    def destroy_node(self) -> bool:
        self._shutdown_event.set()
        self._session_cancel_event.set()
        return super().destroy_node()

    def _trace(self, message: str) -> None:
        if self.trace_logging:
            self.get_logger().info(f"[trace] {message}")

    def _handle_set_enabled(self, request: SetBool.Request, response: SetBool.Response) -> SetBool.Response:
        enabled = bool(request.data)
        with self._state_lock:
            self.enabled = enabled
            self._pending_awake = False
            if not enabled and self._session_active:
                self._session_cancel_event.set()
        response.success = True
        response.message = f"toy_dialog_demo enabled={str(enabled).lower()}"
        self.get_logger().info(response.message)
        return response

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
            if not self.enabled:
                self.get_logger().info("Ignoring /awake=true because toy dialog demo is disabled.")
                return
            self._pending_awake = True
        self.get_logger().info("Received /awake=true. Waiting for greeting completion.")

    def _awake_greeting_done_callback(self, msg: Bool) -> None:
        if self.debug_text_input_mode or not bool(msg.data):
            return
        with self._state_lock:
            if not self.enabled or not self._pending_awake or self._session_active:
                return
        self._start_session_thread("wake")

    def _start_session_thread(self, trigger_reason: str) -> None:
        with self._state_lock:
            if not self.enabled or self._session_active:
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
        history = self._new_session_history()
        self.get_logger().info(f"Toy dialog session started. trigger={trigger_reason}")
        self._set_robot_status("idle")
        fail_window_start: float | None = None
        try:
            while not self._shutdown_event.is_set():
                if self._session_cancel_event.is_set() or not self.enabled:
                    end_reason = "Session canceled or disabled."
                    break

                if not self.debug_text_input_mode:
                    self._set_robot_status("listening")
                call_start = time.monotonic()
                ok, user_text, fail_message = self._call_get_command()
                now = time.monotonic()
                if not ok:
                    if self.debug_text_input_mode:
                        end_reason = fail_message or "Debug input ended."
                        break
                    if fail_window_start is None:
                        fail_window_start = call_start
                    silent_for = now - fail_window_start
                    if silent_for >= self.get_command_fail_window_sec:
                        end_reason = f"No user command received for {self.get_command_fail_window_sec:.1f}s."
                        break
                    time.sleep(max(0.0, self.get_command_retry_delay_sec))
                    continue

                fail_window_start = None
                cleaned_user = user_text.strip()
                if not cleaned_user:
                    continue

                self.get_logger().info(f"User: {cleaned_user}")
                self._append_memory(history, "user", cleaned_user)

                self._set_robot_status("thinking")
                try:
                    decision = self._classify_intent(history, cleaned_user)
                    self._trace(f"intent={decision.intent!r} reason={decision.reason!r}")
                    if decision.intent == "end_session":
                        assistant_reply = "Okay, see you next time."
                        should_end = True
                        llm_reason = decision.reason or "user ended session"
                    elif decision.intent == "vision":
                        self._set_robot_status("operating")
                        assistant_reply = self._answer_with_best_camera(cleaned_user, history)
                        should_end = False
                        llm_reason = ""
                    else:
                        assistant_reply, should_end, llm_reason = self._chat_with_openai(history)
                except Exception as exc:
                    self.get_logger().error(f"Toy dialog handling failed: {exc}")
                    assistant_reply = self.fallback_error_reply or "Sorry, I am having trouble right now."
                    should_end = False
                    llm_reason = ""
                finally:
                    if not self._shutdown_event.is_set() and not self._session_cancel_event.is_set():
                        self._set_robot_status("idle")

                if assistant_reply:
                    self._append_memory(history, "assistant", assistant_reply)
                    speak_ok, speak_message = self._speak_text(assistant_reply)
                    if not speak_ok:
                        if self.debug_text_input_mode:
                            # In debug text-input mode, keep the dialog running even
                            # without the TTS action server.
                            self.get_logger().warn(
                                f"SpeakText unavailable in debug mode; continuing without speech: {speak_message}"
                            )
                        else:
                            end_reason = f"SpeakText action failed: {speak_message}"
                            break

                if should_end:
                    end_reason = llm_reason or "chat model ended session"
                    break
        finally:
            self.get_logger().info(f"Toy dialog session ending. reason={end_reason}")
            self._set_robot_status("sleep")
            with self._state_lock:
                self._session_active = False
                self._pending_awake = False
                self._session_cancel_event.clear()
            self.get_logger().info("Toy dialog session ended.")

    def _new_session_history(self) -> list[dict[str, str]]:
        system_prompt = (
            "You are EVA, a friendly robot talking with children in a demo. "
            "Keep answers short, spoken, warm, and in English. "
            "Use the remembered conversation when the child asks follow-up questions about a toy or animal. "
            "Set end_session=true only when the user clearly says bye, stop, that's all, no more, or go to sleep."
        )
        history = [{"role": "system", "content": system_prompt}]
        if self._run_memory:
            memory_text = "\n".join(
                f"{item['role']}: {item['content']}" for item in self._run_memory[-self.max_run_memory_messages :]
            )
            history.append({"role": "system", "content": f"Memory from this program run:\n{memory_text}"})
        return history

    def _append_memory(self, history: list[dict[str, str]], role: str, content: str) -> None:
        item = {"role": role, "content": content}
        history.append(item)
        self._run_memory.append(item)
        self._trim_history(history, self.max_session_messages)
        if len(self._run_memory) > self.max_run_memory_messages:
            self._run_memory[:] = self._run_memory[-self.max_run_memory_messages :]

    def _classify_intent(self, history: list[dict[str, str]], user_text: str) -> IntentDecision:
        if self._looks_end_session(user_text):
            return IntentDecision("end_session", "keyword shortcut")
        if self._looks_visual(user_text):
            return IntentDecision("vision", "keyword shortcut")

        history_text = "\n".join(f"{item['role']}: {item['content']}" for item in history[-8:])
        prompt = (
            "Classify the latest user message for a robot toy dialog demo.\n"
            "Return JSON only with keys: intent, reason.\n"
            f"intent must be one of {list(VALID_INTENTS)}.\n"
            "Use intent=vision only when the user needs the CURRENT camera image to answer.\n"
            "Use intent=chat for follow-up questions from memory/general knowledge (example: what does this animal eat).\n"
            "Use intent=end_session only for clear stop/bye commands.\n\n"
            f"Recent conversation:\n{history_text}\n\n"
            f"Latest user message:\n{user_text}"
        )
        try:
            parsed = self._openai_json_query(
                prompt_text=prompt,
                max_output_tokens=self.intent_max_output_tokens,
            )
            intent = str(parsed.get("intent", "chat")).strip().lower()
            reason = str(parsed.get("reason", "") or "").strip()
        except Exception as exc:
            self.get_logger().warn(f"Intent classification fallback: {exc}")
            if self._looks_visual(user_text):
                return IntentDecision("vision", "keyword fallback")
            return IntentDecision("chat", "parse fallback")
        if intent not in VALID_INTENTS:
            intent = "vision" if self._looks_visual(user_text) else "chat"
        return IntentDecision(intent=intent, reason=reason)

    def _answer_with_best_camera(self, user_text: str, history: list[dict[str, str]]) -> str:
        last_error = ""
        for camera_name in self._candidate_camera_names():
            ok, answer, error = self._query_vision(user_text, camera_name, history)
            if ok and self._is_useful_visual_reply(answer):
                self._trace(f"vision answer selected camera={camera_name!r}")
                return answer
            last_error = error or answer
            self._trace(f"vision answer rejected camera={camera_name!r} reason={last_error!r}")
        return "I can't see it clearly yet. Could you hold it closer to the camera?"

    def _query_vision(self, user_text: str, camera_name: str, history: list[dict[str, str]]) -> tuple[bool, str, str]:
        context_text = self._recent_dialog_context(history, max_turns=8)
        requested_attributes = self._infer_requested_attributes(history)
        attributes_text = ", ".join(requested_attributes) if requested_attributes else "none"
        prompt = (
            f"{VISION_PROMPT.strip()}\n\n"
            "Also include these keys in entities when possible:\n"
            "- answer\n- subject\n- visible_objects\n- confidence\n\n"
            "Use the recent dialogue context to resolve follow-up intent.\n"
            "If the user says vague phrases like 'how about this animal' or 'what about this one', "
            "carry over the previously asked attribute (for example diet, habitat, behavior) and answer that for the current animal.\n"
            "When possible, include a short contrast with the previous animal in speech_text.\n"
            "If previously requested attributes include diet and habitat, provide BOTH for the current animal in the spoken reply.\n\n"
            f"Previously requested attributes: {attributes_text}\n\n"
            f"Recent dialogue:\n{context_text}\n\n"
            f"User question:\n{user_text}\n"
        )
        last_error = ""
        total_attempts = 1 + self.vision_recapture_attempts
        for capture_attempt in range(1, total_attempts + 1):
            frame = self._wait_for_frame(camera_name, min_received_monotonic=time.monotonic())
            if frame is None:
                last_error = f"No fresh image available from camera '{camera_name}'"
                continue

            image_b64 = self._encode_image(frame.image_bgr)
            try:
                parsed = self._openai_json_query(
                    prompt_text=prompt,
                    image_b64=image_b64,
                    max_output_tokens=self.vision_max_output_tokens,
                    reasoning_effort=self.openai_thinking_effort,
                )
            except Exception as exc:
                last_error = str(exc)
                if capture_attempt < total_attempts and self.vision_recapture_delay_sec > 0.0:
                    time.sleep(self.vision_recapture_delay_sec)
                continue

            reply = self._sanitize_spoken_reply(parsed.get("speech_text", ""))
            if not reply:
                entities = parsed.get("data_text", {}).get("entities", {}) if isinstance(parsed.get("data_text"), dict) else {}
                answer = entities.get("answer") if isinstance(entities, dict) else ""
                reply = self._sanitize_spoken_reply(answer) if answer else ""
            if not reply:
                last_error = "Vision response was empty"
                if capture_attempt < total_attempts and self.vision_recapture_delay_sec > 0.0:
                    time.sleep(self.vision_recapture_delay_sec)
                continue

            if self._is_confident_vision_response(parsed, reply):
                return True, reply, ""

            last_error = "Vision result unclear or low confidence"
            self.get_logger().info(
                f"[trace] vision result unclear on camera={camera_name!r} attempt={capture_attempt}/{total_attempts}; recapturing."
            )
            if capture_attempt < total_attempts and self.vision_recapture_delay_sec > 0.0:
                time.sleep(self.vision_recapture_delay_sec)

        return False, "", last_error or "Vision query failed"

    def _is_confident_vision_response(self, parsed: dict[str, Any], reply: str) -> bool:
        if not self._is_useful_visual_reply(reply):
            return False
        data_text = parsed.get("data_text", {})
        if not isinstance(data_text, dict):
            return False
        complete = data_text.get("complete")
        if isinstance(complete, bool) and not complete:
            return False
        entities = data_text.get("entities", {})
        if not isinstance(entities, dict):
            return True
        confidence = self._parse_confidence_score(entities.get("confidence"))
        return confidence >= self.vision_min_confidence

    @staticmethod
    def _parse_confidence_score(value: Any) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value or "").strip().lower()
        if not text:
            return 1.0
        if text in {"high", "strong"}:
            return 0.85
        if text in {"medium", "moderate"}:
            return 0.60
        if text in {"low", "weak"}:
            return 0.20
        try:
            return float(text)
        except ValueError:
            return 1.0

    @staticmethod
    def _recent_dialog_context(history: list[dict[str, str]], max_turns: int = 8) -> str:
        pairs = [item for item in history if item.get("role") in {"user", "assistant"}]
        if not pairs:
            return "none"
        trimmed = pairs[-max_turns:]
        return "\n".join(f"{item['role']}: {item['content']}" for item in trimmed)

    @staticmethod
    def _infer_requested_attributes(history: list[dict[str, str]]) -> list[str]:
        attributes: list[str] = []
        user_lines = [str(item.get("content", "")).lower() for item in history if item.get("role") == "user"]
        if any(token in line for line in user_lines for token in ("eat", "food", "diet", "feed")):
            attributes.append("diet")
        if any(
            token in line
            for line in user_lines
            for token in ("where", "live", "habitat", "home", "forest", "grassland", "jungle")
        ):
            attributes.append("habitat")
        return attributes

    def _chat_with_openai(self, history: list[dict[str, str]]) -> tuple[str, bool, str]:
        history_text = "\n".join(f"{item['role']}: {item['content']}" for item in history[-20:])
        prompt = (
            "You are EVA, a friendly robot talking with children in English.\n"
            "Return JSON only with keys: assistant_reply, end_session, end_reason.\n"
            "assistant_reply must be short, warm, and spoken style.\n"
            "Use remembered conversation context.\n"
            "Set end_session=true only when user clearly asks to stop/bye.\n\n"
            f"Conversation:\n{history_text}"
        )
        parsed = self._openai_json_query(
            prompt_text=prompt,
            max_output_tokens=self.chat_max_output_tokens,
        )
        reply = self._sanitize_spoken_reply(parsed.get("assistant_reply", ""))
        should_end = _to_bool(parsed.get("end_session"))
        end_reason = "" if parsed.get("end_reason") is None else str(parsed.get("end_reason")).strip()
        return reply or "Could you repeat that, please?", should_end, end_reason

    def _openai_json_query(
        self,
        prompt_text: str,
        image_b64: str | None = None,
        max_output_tokens: int = DEFAULT_CHAT_MAX_TOKENS,
        reasoning_effort: str = "",
    ) -> dict[str, Any]:
        attempt_specs = [(max_output_tokens, self.openai_timeout_sec)]
        for _ in range(self.openai_max_retry_count):
            attempt_specs.append((self.retry_max_output_tokens, self.openai_retry_timeout_sec))

        last_exc: Exception | None = None
        for attempt_index, (attempt_tokens, attempt_timeout) in enumerate(attempt_specs, start=1):
            content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt_text}]
            if image_b64:
                content.append(
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{image_b64}",
                        "detail": self.openai_image_detail,
                    }
                )
            payload: dict[str, Any] = {
                "model": self.openai_model,
                "instructions": "Return only valid JSON.",
                "input": [{"role": "user", "content": content}],
                "text": {"format": {"type": "json_object"}},
                "max_output_tokens": int(attempt_tokens),
                "store": False,
            }
            if reasoning_effort:
                payload["reasoning"] = {"effort": reasoning_effort}
            self._trace(
                "openai request start | "
                f"attempt={attempt_index}/{len(attempt_specs)} "
                f"use_vision={bool(image_b64)} "
                f"timeout_s={attempt_timeout:.1f} "
                f"max_output_tokens={attempt_tokens}"
            )
            try:
                result = requests.post(
                    self.openai_api_url,
                    headers={
                        "Authorization": f"Bearer {self._openai_api_key()}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=max(1.0, attempt_timeout),
                )
                if not result.ok:
                    raise RuntimeError(f"OpenAI request failed: HTTP {result.status_code} {result.text[:500]}")
                output_text = self._extract_openai_output_text(result.json())
                self._trace(f"openai raw reply | content={output_text[:1200]!r}")
                parsed = self._parse_json_relaxed(output_text, allow_repair=True)
                if isinstance(parsed, dict) and parsed:
                    return parsed
                raise RuntimeError("OpenAI response JSON was empty")
            except Exception as exc:
                last_exc = exc
                if attempt_index < len(attempt_specs):
                    self.get_logger().warn(
                        "OpenAI reply was incomplete/malformed; retrying once with larger output budget."
                    )
                    continue
                break
        raise RuntimeError(str(last_exc) if last_exc else "OpenAI request failed")

    def _openai_api_key(self) -> str:
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
            item_text = str(item.get("text") or item.get("output_text") or "").strip()
            if item_text:
                parts.append(item_text)
            for content in item.get("content", []):
                if not isinstance(content, dict):
                    continue
                if content.get("type") in {"output_text", "text"}:
                    text = str(content.get("text") or content.get("output_text") or "")
                    if text:
                        parts.append(text)
                elif "json" in content and isinstance(content.get("json"), (dict, list)):
                    parts.append(json.dumps(content["json"], ensure_ascii=False))
                elif "value" in content and isinstance(content.get("value"), (dict, list)):
                    parts.append(json.dumps(content["value"], ensure_ascii=False))
        return "\n".join(parts).strip()

    def _on_image(self, camera_name: str, msg: Image) -> None:
        try:
            image = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            self.get_logger().warn(f"Failed to convert image from '{camera_name}': {exc}")
            return

        stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        with self._lock:
            self._frames[camera_name] = CameraFrame(
                image_bgr=image,
                stamp_ns=stamp_ns,
                received_monotonic=time.monotonic(),
            )

    def _subscribe_camera(self, camera_name: str) -> None:
        if camera_name not in self._camera_topics_by_name:
            raise ValueError(f"Unknown camera '{camera_name}'. Available cameras: {sorted(self._camera_topics_by_name)}")
        with self._lock:
            if camera_name in self._camera_subscriptions:
                if self.lazy_subscriptions:
                    self._camera_subscription_deadlines[camera_name] = (
                        time.monotonic() + self.subscription_idle_timeout_sec
                    )
                return
        topic = self._camera_topics_by_name[camera_name]
        subscription = self.create_subscription(
            Image,
            topic,
            lambda msg, name=camera_name: self._on_image(name, msg),
            qos_profile_sensor_data,
            callback_group=self._callback_group,
        )
        with self._lock:
            self._camera_subscriptions[camera_name] = subscription
            if self.lazy_subscriptions:
                self._camera_subscription_deadlines[camera_name] = (
                    time.monotonic() + self.subscription_idle_timeout_sec
                )
        self.get_logger().info(f"[{camera_name}] Subscribed to camera stream on demand.")

    def _unsubscribe_camera(self, camera_name: str) -> None:
        with self._lock:
            subscription = self._camera_subscriptions.pop(camera_name, None)
            self._camera_subscription_deadlines.pop(camera_name, None)
            self._frames.pop(camera_name, None)
        if subscription is None:
            return
        try:
            self.destroy_subscription(subscription)
        except Exception:
            pass
        self.get_logger().info(f"[{camera_name}] Unsubscribed from idle camera stream.")

    def _touch_camera_subscription(self, camera_name: str) -> None:
        if not self.lazy_subscriptions:
            return
        with self._lock:
            if camera_name in self._camera_subscriptions:
                self._camera_subscription_deadlines[camera_name] = (
                    time.monotonic() + self.subscription_idle_timeout_sec
                )

    def _cleanup_idle_subscriptions(self) -> None:
        if not self.lazy_subscriptions or self.subscription_idle_timeout_sec <= 0.0:
            return
        now = time.monotonic()
        with self._lock:
            expired = [name for name, deadline in self._camera_subscription_deadlines.items() if deadline <= now]
        for name in expired:
            self._unsubscribe_camera(name)

    def _wait_for_frame(
        self,
        camera_name: str,
        min_received_monotonic: float | None = None,
    ) -> CameraFrame | None:
        if camera_name not in self._camera_topics_by_name:
            raise ValueError(f"Unknown camera '{camera_name}'. Available cameras: {sorted(self._camera_topics_by_name)}")
        self._subscribe_camera(camera_name)
        deadline = time.time() + self.image_wait_timeout_sec
        while time.time() < deadline:
            self._touch_camera_subscription(camera_name)
            with self._lock:
                frame = self._frames.get(camera_name)
            if frame is not None:
                if min_received_monotonic is not None and frame.received_monotonic < min_received_monotonic:
                    time.sleep(0.05)
                    continue
                return frame
            time.sleep(0.05)
        return None

    def _call_get_command(self) -> tuple[bool, str, str]:
        if self.debug_text_input_mode:
            return self._call_debug_text_command()
        if not self._get_command_client.wait_for_service(timeout_sec=0.5):
            return False, "", f"Service '{self.get_command_service}' not ready."
        future = self._get_command_client.call_async(Trigger.Request())
        ok, response, error_text = self._wait_for_future(future, self.service_response_timeout_sec)
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
                if write_stream is not None and not prompt_written:
                    write_stream.write(prompt)
                    write_stream.flush()
                    prompt_written = True
                ready, _, _ = select.select([read_stream], [], [], 0.2)
                if not ready:
                    continue
                line = read_stream.readline()
                if line == "":
                    return False, "", "Debug text input closed."
                text = line.strip()
                if text:
                    return True, text, ""
                prompt_written = False
            return False, "", "Canceled."
        except Exception as exc:
            return False, "", f"Debug text input failed: {exc}"
        finally:
            if close_read and read_stream is not None:
                read_stream.close()
            if close_write and write_stream is not None:
                write_stream.close()

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
        return bool(result.success), str(result.message)

    def _set_robot_status(self, target: str) -> bool:
        normalized = str(target).strip().lower()
        if normalized not in VALID_STATUSES:
            return False
        if not self._robot_status_client.wait_for_service(timeout_sec=0.8):
            return False
        request = RobotStatus.Request()
        request.status = normalized
        future = self._robot_status_client.call_async(request)
        ok, response, _ = self._wait_for_future(future, self.service_response_timeout_sec, cancel_on_session_stop=False)
        return bool(ok and response is not None and response.success)

    def _wait_for_future(self, future, timeout_sec: float, *, cancel_on_session_stop: bool = True) -> tuple[bool, Any, str]:
        event = threading.Event()
        holder: dict[str, Any] = {}

        def _done(fut) -> None:
            holder["future"] = fut
            event.set()

        future.add_done_callback(_done)
        start = time.monotonic()
        while not event.wait(timeout=0.1):
            if self._shutdown_event.is_set():
                return False, None, "Canceled."
            if cancel_on_session_stop and self._session_cancel_event.is_set():
                return False, None, "Canceled."
            if (time.monotonic() - start) >= max(0.1, float(timeout_sec)):
                return False, None, "timeout"
        fut = holder.get("future", future)
        exc = fut.exception()
        if exc is not None:
            return False, None, str(exc)
        return True, fut.result(), ""

    @staticmethod
    def _encode_image(image_bgr: Any) -> str:
        ok, encoded = cv2.imencode(".jpg", image_bgr)
        if not ok:
            raise RuntimeError("Failed to encode image for vision request")
        return base64.b64encode(encoded.tobytes()).decode("ascii")

    def _candidate_camera_names(self) -> list[str]:
        candidates: list[str] = []
        for name in [self.default_camera_name, "camera0", "gripper_camera", *self.camera_names]:
            if name and name not in candidates:
                candidates.append(name)
        return candidates

    @staticmethod
    def _looks_visual(text: str) -> bool:
        cleaned = text.lower()
        patterns = (
            "what is this",
            "what's this",
            "what am i holding",
            "what do you see",
            "can you see",
            "look at",
            "in my hand",
            "shown",
            "camera",
        )
        return any(pattern in cleaned for pattern in patterns)

    @staticmethod
    def _looks_end_session(text: str) -> bool:
        cleaned = re.sub(r"[^a-z0-9\s]", " ", text.lower())
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        end_phrases = (
            "bye",
            "goodbye",
            "stop",
            "stop chatting",
            "end session",
            "that is all",
            "that's all",
            "no more",
            "go to sleep",
        )
        return any(phrase == cleaned or phrase in cleaned for phrase in end_phrases)

    @staticmethod
    def _is_useful_visual_reply(text: str) -> bool:
        cleaned = str(text).strip().lower()
        if not cleaned:
            return False
        unclear = (
            "can't see",
            "cannot see",
            "couldn't see",
            "could not see",
            "not clear",
            "unclear",
            "i don't see",
            "i do not see",
            "cannot identify",
            "can't identify",
            "unable to identify",
        )
        return not any(marker in cleaned for marker in unclear)

    @staticmethod
    def _parse_json_relaxed(text: str, allow_repair: bool = True) -> dict[str, Any]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            if allow_repair:
                repaired = ToyDialogDemoNode._repair_json_tail(cleaned)
                if repaired != cleaned:
                    try:
                        return json.loads(repaired)
                    except json.JSONDecodeError:
                        pass
            match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            if not match:
                raise
            candidate = match.group(0)
            if allow_repair:
                candidate = ToyDialogDemoNode._repair_json_tail(candidate)
            return json.loads(candidate)

    @staticmethod
    def _repair_json_tail(text: str) -> str:
        repaired = text.strip()
        if not repaired:
            return repaired
        literal_repairs = {
            r":\s*nul(?=(?:\s*[}\]])*\s*$)": ": null",
            r":\s*tru(?=(?:\s*[}\]])*\s*$)": ": true",
            r":\s*fals(?=(?:\s*[}\]])*\s*$)": ": false",
        }
        for pattern, replacement in literal_repairs.items():
            repaired = re.sub(pattern, replacement, repaired)
        open_braces = repaired.count("{")
        close_braces = repaired.count("}")
        if close_braces < open_braces:
            repaired += "}" * (open_braces - close_braces)
        open_brackets = repaired.count("[")
        close_brackets = repaired.count("]")
        if close_brackets < open_brackets:
            repaired += "]" * (open_brackets - close_brackets)
        return repaired

    @staticmethod
    def _sanitize_spoken_reply(text: Any) -> str:
        cleaned = str(text).strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = re.sub(r"[{}\"]", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,:\n\t")
        return cleaned

    @staticmethod
    def _trim_history(history: list[dict[str, str]], max_messages: int) -> None:
        if not history:
            return
        system = [item for item in history if item.get("role") == "system"]
        non_system = [item for item in history if item.get("role") != "system"]
        history[:] = system + non_system[-max_messages:]


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = ToyDialogDemoNode()
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

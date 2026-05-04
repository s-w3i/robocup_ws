#!/usr/bin/env python3
"""ROS 2 service node for text or image-assisted VLM queries via Ollama."""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any

import cv2
import requests
import rclpy
from coqui_tts_interfaces.srv import RobotStatus
from cv_bridge import CvBridge, CvBridgeError
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from vlm_interfaces.srv import VlmQuery

DUAL_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "speech_text": {"type": "string"},
        "data_text": {"type": "object"},
    },
    "required": ["speech_text", "data_text"],
}


@dataclass
class CameraFrame:
    image_bgr: Any
    stamp_ns: int
    received_monotonic: float


@dataclass
class QueryPolicy:
    request_profile: str
    max_retry_count: int
    allow_json_repair: bool
    timeout_sec: float
    retry_timeout_sec: float
    num_predict: int
    retry_num_predict: int


class VlmQueryServiceNode(Node):
    def __init__(self) -> None:
        super().__init__("vlm_query_service_node")

        self.declare_parameter("service_name", "/vlm/query")
        self.declare_parameter("ollama_base_url", os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"))
        self.declare_parameter("vlm_model", os.environ.get("VISION_MODEL", "qwen3.5:9b"))
        self.declare_parameter("vlm_keep_alive", os.environ.get("OLLAMA_KEEP_ALIVE", "30m"))
        self.declare_parameter("vlm_timeout_sec", 90.0)
        self.declare_parameter("text_timeout_sec", 60.0)
        self.declare_parameter("text_retry_timeout_sec", 90.0)
        self.declare_parameter("ollama_start_timeout_sec", 20.0)
        self.declare_parameter("auto_start_ollama", True)
        self.declare_parameter("vlm_num_ctx", 4096)
        self.declare_parameter("text_num_ctx", 2048)
        self.declare_parameter("fast_num_predict", 128)
        self.declare_parameter("thinking_num_predict", 512)
        self.declare_parameter("text_fast_num_predict", 64)
        self.declare_parameter("text_thinking_num_predict", 96)
        self.declare_parameter("default_camera_name", "camera0")
        self.declare_parameter("default_camera_topic", "/camera0/color/image_raw")
        self.declare_parameter("camera_names", ["camera0"])
        self.declare_parameter("camera_topics", ["/camera0/color/image_raw"])
        self.declare_parameter("camera_names_csv", "")
        self.declare_parameter("camera_topics_csv", "")
        self.declare_parameter("image_wait_timeout_sec", 3.0)
        self.declare_parameter("lazy_subscriptions", True)
        self.declare_parameter("subscription_idle_timeout_sec", 3.0)
        self.declare_parameter("subscription_poll_period_sec", 0.5)
        self.declare_parameter("manage_robot_status", True)
        self.declare_parameter("robot_status_service", "/robot_status")
        self.declare_parameter("robot_status_timeout_sec", 2.0)
        self.declare_parameter("trace_logging", True)
        self.declare_parameter("trace_log_prompt", True)
        self.declare_parameter("trace_log_raw_reply", True)
        self.declare_parameter(
            "system_prompt",
            (
                "You are a robotics vision-language assistant. "
                "Answer clearly, directly, and based on the provided text and image when available."
            ),
        )

        self.service_name = str(self.get_parameter("service_name").value)
        self.ollama_base_url = str(self.get_parameter("ollama_base_url").value).rstrip("/")
        self.vlm_model = str(self.get_parameter("vlm_model").value)
        self.vlm_keep_alive = str(self.get_parameter("vlm_keep_alive").value)
        self.vlm_timeout_sec = max(5.0, float(self.get_parameter("vlm_timeout_sec").value))
        self.text_timeout_sec = max(3.0, float(self.get_parameter("text_timeout_sec").value))
        self.text_retry_timeout_sec = max(
            self.text_timeout_sec,
            float(self.get_parameter("text_retry_timeout_sec").value),
        )
        self.ollama_start_timeout_sec = max(1.0, float(self.get_parameter("ollama_start_timeout_sec").value))
        self.auto_start_ollama = bool(self.get_parameter("auto_start_ollama").value)
        self.vlm_num_ctx = max(512, int(self.get_parameter("vlm_num_ctx").value))
        self.text_num_ctx = max(256, int(self.get_parameter("text_num_ctx").value))
        self.fast_num_predict = max(32, int(self.get_parameter("fast_num_predict").value))
        self.thinking_num_predict = max(self.fast_num_predict, int(self.get_parameter("thinking_num_predict").value))
        self.text_fast_num_predict = max(16, int(self.get_parameter("text_fast_num_predict").value))
        self.text_thinking_num_predict = max(
            self.text_fast_num_predict,
            int(self.get_parameter("text_thinking_num_predict").value),
        )
        self.default_camera_name = str(self.get_parameter("default_camera_name").value).strip() or "camera0"
        self.default_camera_topic = str(self.get_parameter("default_camera_topic").value).strip() or "/camera0/color/image_raw"
        self.camera_names = [str(name).strip() for name in self.get_parameter("camera_names").value]
        self.camera_topics = [str(topic).strip() for topic in self.get_parameter("camera_topics").value]
        camera_names_csv = str(self.get_parameter("camera_names_csv").value).strip()
        camera_topics_csv = str(self.get_parameter("camera_topics_csv").value).strip()
        self.image_wait_timeout_sec = max(0.1, float(self.get_parameter("image_wait_timeout_sec").value))
        self.lazy_subscriptions = bool(self.get_parameter("lazy_subscriptions").value)
        self.subscription_idle_timeout_sec = max(
            0.0, float(self.get_parameter("subscription_idle_timeout_sec").value)
        )
        self.subscription_poll_period_sec = max(
            0.1, float(self.get_parameter("subscription_poll_period_sec").value)
        )
        self.manage_robot_status = bool(self.get_parameter("manage_robot_status").value)
        self.robot_status_service = str(self.get_parameter("robot_status_service").value).strip() or "/robot_status"
        self.robot_status_timeout_sec = max(0.1, float(self.get_parameter("robot_status_timeout_sec").value))
        self.trace_logging = bool(self.get_parameter("trace_logging").value)
        self.trace_log_prompt = bool(self.get_parameter("trace_log_prompt").value)
        self.trace_log_raw_reply = bool(self.get_parameter("trace_log_raw_reply").value)
        self.system_prompt = str(self.get_parameter("system_prompt").value).strip()

        if not camera_names_csv and not camera_topics_csv:
            self.camera_names = [self.default_camera_name]
            self.camera_topics = [self.default_camera_topic]
        if camera_names_csv:
            self.camera_names = [part.strip() for part in camera_names_csv.split(",") if part.strip()]
        if camera_topics_csv:
            self.camera_topics = [part.strip() for part in camera_topics_csv.split(",") if part.strip()]

        if len(self.camera_names) != len(self.camera_topics):
            raise ValueError("camera_names and camera_topics must have the same length")

        self._bridge = CvBridge()
        self._lock = threading.Lock()
        self._callback_group = ReentrantCallbackGroup()
        self._frames: dict[str, CameraFrame] = {}
        self._camera_topics_by_name = dict(zip(self.camera_names, self.camera_topics))
        self._camera_subscriptions: dict[str, Any] = {}
        self._camera_subscription_deadlines: dict[str, float] = {}
        self._robot_status_client = self.create_client(
            RobotStatus,
            self.robot_status_service,
            callback_group=self._callback_group,
        )

        self.create_timer(self.subscription_poll_period_sec, self._cleanup_idle_subscriptions)

        if not self.lazy_subscriptions:
            for camera_name in self._camera_topics_by_name:
                self._subscribe_camera(camera_name)

        self.create_service(
            VlmQuery,
            self.service_name,
            self._handle_query,
            callback_group=self._callback_group,
        )

        if self.auto_start_ollama:
            self._ensure_ollama_running()

        self.get_logger().info(
            f"VLM query service ready on {self.service_name} using model '{self.vlm_model}'"
        )
        self.get_logger().info(
            "Lazy subscriptions: %s (idle_timeout=%.2fs)"
            % (str(self.lazy_subscriptions).lower(), self.subscription_idle_timeout_sec)
        )

    def _trace(self, message: str) -> None:
        if self.trace_logging:
            self.get_logger().info(f"[trace] {message}")

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
            raise ValueError(
                f"Unknown camera '{camera_name}'. Available cameras: {sorted(self._camera_topics_by_name)}"
            )

        topic = self._camera_topics_by_name[camera_name]
        with self._lock:
            if camera_name in self._camera_subscriptions:
                if self.lazy_subscriptions:
                    self._camera_subscription_deadlines[camera_name] = (
                        time.monotonic() + self.subscription_idle_timeout_sec
                    )
                return

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
            expired_camera_names = [
                camera_name
                for camera_name, deadline in self._camera_subscription_deadlines.items()
                if deadline <= now
            ]

        for camera_name in expired_camera_names:
            self._unsubscribe_camera(camera_name)

    def _handle_query(
        self, request: VlmQuery.Request, response: VlmQuery.Response
    ) -> VlmQuery.Response:
        response.success = False
        response.message = ""
        response.speech_text = ""
        response.data_text = ""
        response.image_used = False
        response.camera_used = ""
        response.reasoning_mode_used = ""
        response.model_name = ""

        thinking_state_set = False
        request_started = time.time()
        request_started_monotonic = time.monotonic()
        try:
            reasoning_mode = self._normalize_reasoning_mode(request.reasoning_mode)
            response.reasoning_mode_used = reasoning_mode
            self._trace(
                "request received | "
                f"need_image={bool(request.need_image)} "
                f"camera={str(request.camera_name).strip() or self.default_camera_name!r} "
                f"profile={str(request.request_profile).strip() or 'default'!r} "
                f"reasoning_mode={reasoning_mode} "
                f"user_input={str(request.user_input).strip()[:300]!r}"
            )

            if self.manage_robot_status:
                thinking_state_set = self._set_robot_status("thinking")

            if self.auto_start_ollama and not self._ollama_ready(timeout=1.0):
                self._ensure_ollama_running()

            image_b64 = None
            use_vision = bool(request.need_image)
            if bool(request.need_image):
                camera_name = str(request.camera_name).strip() or self.default_camera_name
                frame = self._wait_for_frame(
                    camera_name,
                    min_received_monotonic=request_started_monotonic,
                )
                if frame is None:
                    response.message = (
                        f"No fresh image available from camera '{camera_name}'. "
                        "Check the camera topic or increase image_wait_timeout_sec."
                    )
                    return response

                image_b64 = self._encode_image(frame.image_bgr)
                response.image_used = True
                response.camera_used = camera_name

            prompt_text = self._build_prompt(
                prompt=str(request.prompt),
                user_input=str(request.user_input),
                reasoning_mode=reasoning_mode,
                need_image=bool(request.need_image),
                camera_used=response.camera_used,
            )
            if self.trace_log_prompt:
                self._trace(f"prompt built | prompt={prompt_text[:1200]!r}")
            query_policy = self._resolve_query_policy(
                request_profile=str(request.request_profile),
                reasoning_mode=reasoning_mode,
                use_vision=use_vision,
                max_retry_count=int(request.max_retry_count),
                json_repair_mode=int(request.json_repair_mode),
                num_predict_override=int(request.num_predict_override),
                timeout_sec_override=float(request.timeout_sec_override),
            )
            speech_text, data_text, model_name = self._query_ollama(
                prompt_text,
                image_b64=image_b64,
                reasoning_mode=reasoning_mode,
                query_policy=query_policy,
            )

            response.success = True
            response.message = "ok"
            response.speech_text = speech_text
            response.data_text = data_text
            response.model_name = model_name
            self._trace(
                "request complete | "
                f"elapsed_s={time.time() - request_started:.3f} "
                f"speech_text={speech_text[:200]!r} "
                f"data_text={data_text[:400]!r}"
            )
            return response
        except Exception as exc:  # pragma: no cover
            response.message = str(exc)
            self.get_logger().error(f"VLM query failed: {exc}")
            return response
        finally:
            if self.manage_robot_status and thinking_state_set:
                self._set_robot_status("idle")

    def _wait_for_frame(
        self,
        camera_name: str,
        min_received_monotonic: float | None = None,
    ) -> CameraFrame | None:
        if camera_name not in self._camera_topics_by_name:
            raise ValueError(
                f"Unknown camera '{camera_name}'. Available cameras: {sorted(self._camera_topics_by_name)}"
            )

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

    def _query_ollama(
        self, prompt_text: str, image_b64: str | None, reasoning_mode: str, query_policy: QueryPolicy
    ) -> tuple[str, str, str]:
        think = reasoning_mode == "thinking"
        # For structured robotics control replies, keeping Ollama's explicit
        # "think" mode off is more reliable than exposing chain-of-thought.
        api_think = False
        use_vision = image_b64 is not None
        model_name = self.vlm_model
        keep_alive = self.vlm_keep_alive
        timeout_sec = query_policy.timeout_sec
        num_ctx = self.vlm_num_ctx if use_vision else self.text_num_ctx
        num_predict = query_policy.num_predict

        message: dict[str, Any] = {"role": "user", "content": prompt_text}
        if image_b64 is not None:
            message["images"] = [image_b64]

        payload: dict[str, Any] = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                message,
            ],
            "stream": False,
            "keep_alive": keep_alive,
            "think": api_think,
            "format": DUAL_RESPONSE_SCHEMA,
            "options": {
                "temperature": 0.2 if think else 0.0,
                "num_ctx": num_ctx,
                "num_predict": num_predict,
            },
        }

        attempt_specs = [(num_predict, timeout_sec)]
        for _ in range(max(0, query_policy.max_retry_count)):
            attempt_specs.append((query_policy.retry_num_predict, query_policy.retry_timeout_sec))

        last_error: Exception | None = None
        final_speech_text = ""
        final_data_text = ""
        for attempt_index, (attempt_predict, attempt_timeout) in enumerate(attempt_specs, start=1):
            payload["options"]["num_predict"] = attempt_predict
            attempt_started = time.time()
            self._trace(
                "ollama request start | "
                f"attempt={attempt_index}/{len(attempt_specs)} "
                f"use_vision={use_vision} "
                f"model={model_name!r} "
                f"timeout_s={attempt_timeout:.1f} "
                f"num_ctx={num_ctx} "
                f"num_predict={attempt_predict} "
                f"reasoning_mode={reasoning_mode}"
            )
            try:
                result = requests.post(
                    f"{self.ollama_base_url}/api/chat",
                    json=payload,
                    timeout=max(1.0, attempt_timeout),
                )
                result.raise_for_status()
                self._trace(
                    "ollama request finished | "
                    f"attempt={attempt_index} "
                    f"status_code={result.status_code} "
                    f"elapsed_s={time.time() - attempt_started:.3f}"
                )
                message_obj = result.json().get("message", {})
                content = str(message_obj.get("content", "") or "").strip()
                thinking = str(message_obj.get("thinking", "") or "").strip()
                if self.trace_log_raw_reply:
                    self._trace(
                        "ollama raw reply | "
                        f"content={content[:1200]!r} "
                        f"thinking_present={bool(thinking)} "
                        f"thinking_chars={len(thinking)}"
                    )
                speech_text, data_text = self._decode_dual_response(
                    content=content,
                    thinking=thinking,
                    allow_json_repair=query_policy.allow_json_repair,
                )
                self._trace(
                    "decoded reply | "
                    f"speech_text={speech_text[:200]!r} "
                    f"data_text={data_text[:400]!r}"
                )
                validation_error = self._validate_dual_response(
                    speech_text,
                    data_text,
                    allow_json_repair=query_policy.allow_json_repair,
                    request_profile=query_policy.request_profile,
                )
                if validation_error is None:
                    final_speech_text = speech_text
                    final_data_text = data_text
                    break
                last_error = RuntimeError(validation_error)
                if attempt_index < len(attempt_specs):
                    self.get_logger().warn(
                        "Structured VLM reply was incomplete or malformed; retrying with a larger generation budget."
                    )
                    continue
                raise last_error
            except requests.ReadTimeout as exc:
                last_error = exc
                if attempt_index < len(attempt_specs):
                    self.get_logger().warn(
                        "Text query timed out waiting for Ollama; retrying once with a longer timeout."
                    )
                    continue
                raise RuntimeError(
                    "Timed out waiting for Ollama response. "
                    f"text_timeout_sec={self.text_timeout_sec:.1f}, "
                    f"text_retry_timeout_sec={self.text_retry_timeout_sec:.1f}, "
                    f"text_fast_num_predict={self.text_fast_num_predict}, "
                    f"text_thinking_num_predict={self.text_thinking_num_predict}."
                ) from exc
            except requests.RequestException as exc:
                last_error = exc
                raise

        if last_error is not None and not final_speech_text and not final_data_text:
            raise RuntimeError(str(last_error) if last_error is not None else "Ollama request failed")
        if final_speech_text or final_data_text:
            return final_speech_text, final_data_text, model_name
        raise RuntimeError("VLM returned an empty response")

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
                repaired = VlmQueryServiceNode._repair_json_tail(cleaned)
                if repaired != cleaned:
                    try:
                        return json.loads(repaired)
                    except json.JSONDecodeError:
                        pass
            match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            if not match:
                raise
            candidate = match.group(0)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                if allow_repair:
                    repaired = VlmQueryServiceNode._repair_json_tail(candidate)
                    if repaired != candidate:
                        return json.loads(repaired)
                raise

    @staticmethod
    def _repair_json_tail(text: str) -> str:
        repaired = text.strip()
        if not repaired:
            return repaired

        # Repair common truncated JSON literals at the end of the payload.
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

    def _decode_dual_response(self, content: str, thinking: str, allow_json_repair: bool) -> tuple[str, str]:
        payload_text = content or thinking
        if not payload_text:
            return "", ""

        try:
            parsed = self._parse_json_relaxed(payload_text, allow_repair=allow_json_repair)
        except Exception:
            cleaned = payload_text.strip()
            return cleaned, cleaned

        # Preferred schema: {"speech_text": "...", "data_text": ...}
        speech_text = str(parsed.get("speech_text", "") or "").strip()
        data_value = parsed.get("data_text", "")
        if isinstance(data_value, str):
            data_text = data_value.strip()
        else:
            data_text = json.dumps(data_value, ensure_ascii=False)

        # Backward-compatible schema: model returned only the structured result.
        has_structured_fields = any(key in parsed for key in ("task", "reason", "entities"))
        if not speech_text and not data_text and has_structured_fields:
            data_text = json.dumps(parsed, ensure_ascii=False)

        # If the model returned both speech_text and structured fields at the top level,
        # preserve the structured fields in data_text.
        if speech_text and not data_text and has_structured_fields:
            structured = {
                key: value
                for key, value in parsed.items()
                if key not in {"speech_text", "data_text"}
            }
            if structured:
                data_text = json.dumps(structured, ensure_ascii=False)

        # Another common failure mode: the model places the structured JSON string
        # inside speech_text and leaves data_text empty.
        if speech_text and not data_text:
            try:
                nested = self._parse_json_relaxed(speech_text, allow_repair=allow_json_repair)
            except Exception:
                nested = None
            if isinstance(nested, dict) and any(
                key in nested for key in ("task", "reason", "entities")
            ):
                data_text = json.dumps(nested, ensure_ascii=False)
                speech_text = ""

        return speech_text, data_text

    def _validate_dual_response(
        self,
        speech_text: str,
        data_text: str,
        allow_json_repair: bool,
        request_profile: str,
    ) -> str | None:
        if not speech_text and not data_text:
            return "VLM returned an empty response"
        if not data_text:
            return "data_text is empty"
        try:
            parsed = self._parse_json_relaxed(data_text, allow_repair=allow_json_repair)
        except Exception as exc:
            return f"data_text is not valid JSON: {exc}"
        if not isinstance(parsed, dict):
            return "data_text is not a JSON object"
        entities = parsed.get("entities")
        if entities is None or not isinstance(entities, dict):
            return "data_text.entities is missing or not an object"
        if "reason" not in parsed:
            return "data_text.reason is missing"
        if "task" not in parsed:
            return "data_text.task is missing"
        requires_complete = request_profile not in {"dialogue_text"}
        if requires_complete and "complete" not in parsed:
            return "data_text.complete is missing"
        return None

    def _resolve_query_policy(
        self,
        request_profile: str,
        reasoning_mode: str,
        use_vision: bool,
        max_retry_count: int,
        json_repair_mode: int,
        num_predict_override: int,
        timeout_sec_override: float,
    ) -> QueryPolicy:
        profile = str(request_profile).strip().lower() or "default"
        think = reasoning_mode == "thinking"

        if use_vision:
            base_timeout = self.vlm_timeout_sec
            base_num_predict = self.thinking_num_predict if think else self.fast_num_predict
            retry_timeout = max(self.vlm_timeout_sec, 120.0)
            retry_num_predict = max(self.thinking_num_predict * 2, 768) if think else max(self.fast_num_predict * 2, 192)
        else:
            base_timeout = self.text_timeout_sec
            base_num_predict = self.text_thinking_num_predict if think else self.text_fast_num_predict
            retry_timeout = self.text_retry_timeout_sec
            retry_num_predict = max(self.text_thinking_num_predict * 2, 192) if think else max(self.text_fast_num_predict * 2, 96)

        profile_defaults: dict[str, tuple[int, bool]] = {
            "default": (1, True),
            "fast_text": (0, True),
            "dialogue_text": (1, True),
            "strict_text": (1, True),
            "vision_strict": (1, True),
            "vision_gate": (0, True),
        }
        default_retry_count, default_allow_repair = profile_defaults.get(
            profile,
            profile_defaults["vision_strict" if use_vision else "default"],
        )

        # Gate-style vision checks already have outer workflow retries, so keep
        # each individual query lightweight.
        if profile == "vision_gate":
            base_timeout = min(base_timeout, 45.0)
            retry_timeout = max(base_timeout, 60.0)
            if think:
                base_num_predict = min(base_num_predict, 256)
                retry_num_predict = max(base_num_predict * 2, 512)
            else:
                base_num_predict = min(base_num_predict, 128)
                retry_num_predict = max(base_num_predict * 2, 192)
        elif profile == "fast_text":
            # Short extraction tasks for live dialogue should return quickly.
            if think:
                base_timeout = min(base_timeout, 18.0)
                base_num_predict = min(base_num_predict, 72)
                retry_num_predict = max(base_num_predict * 2, 144)
            else:
                base_timeout = min(base_timeout, 10.0)
                base_num_predict = min(base_num_predict, 48)
                retry_num_predict = max(base_num_predict * 2, 96)
            retry_timeout = max(base_timeout, min(self.text_retry_timeout_sec, 20.0))
        elif profile == "dialogue_text":
            # Conversational slot-filling needs slightly more room than a terse
            # extractor, while still staying responsive.
            if think:
                base_timeout = min(base_timeout, 24.0)
                base_num_predict = min(base_num_predict, 112)
                retry_num_predict = max(base_num_predict * 2, 192)
            else:
                base_timeout = min(base_timeout, 14.0)
                base_num_predict = min(base_num_predict, 72)
                retry_num_predict = max(base_num_predict * 2, 128)
            retry_timeout = max(base_timeout, min(self.text_retry_timeout_sec, 24.0))

        resolved_retry_count = default_retry_count if max_retry_count <= 0 else max_retry_count
        if json_repair_mode > 0:
            allow_json_repair = True
        elif json_repair_mode < 0:
            allow_json_repair = False
        else:
            allow_json_repair = default_allow_repair

        resolved_num_predict = base_num_predict if num_predict_override <= 0 else max(16, num_predict_override)
        resolved_timeout_sec = base_timeout if timeout_sec_override <= 0 else max(1.0, timeout_sec_override)

        policy = QueryPolicy(
            request_profile=profile,
            max_retry_count=max(0, resolved_retry_count),
            allow_json_repair=allow_json_repair,
            timeout_sec=resolved_timeout_sec,
            retry_timeout_sec=retry_timeout,
            num_predict=resolved_num_predict,
            retry_num_predict=max(resolved_num_predict, retry_num_predict),
        )
        self._trace(
            "query policy | "
            f"profile={policy.request_profile!r} "
            f"use_vision={use_vision} "
            f"max_retry_count={policy.max_retry_count} "
            f"allow_json_repair={policy.allow_json_repair} "
            f"timeout_sec={policy.timeout_sec:.1f} "
            f"retry_timeout_sec={policy.retry_timeout_sec:.1f} "
            f"num_predict={policy.num_predict} "
            f"retry_num_predict={policy.retry_num_predict}"
        )
        return policy

    def _set_robot_status(self, status: str) -> bool:
        if not self._robot_status_client.wait_for_service(timeout_sec=self.robot_status_timeout_sec):
            self.get_logger().warn(
                f"Robot status service '{self.robot_status_service}' is not available."
            )
            return False

        request = RobotStatus.Request()
        request.status = status
        future = self._robot_status_client.call_async(request)

        deadline = time.time() + self.robot_status_timeout_sec
        while time.time() < deadline:
            if future.done():
                break
            time.sleep(0.05)

        if not future.done():
            self.get_logger().warn(
                f"Timed out setting robot status to '{status}' via {self.robot_status_service}."
            )
            return False

        exception = future.exception()
        if exception is not None:
            self.get_logger().warn(
                f"Failed to set robot status to '{status}': {exception}"
            )
            return False

        result = future.result()
        if result is None:
            self.get_logger().warn(f"Robot status service returned no result for '{status}'.")
            return False
        if not result.success:
            self.get_logger().warn(
                f"Robot status update to '{status}' was rejected: {result.message}"
            )
            return False
        return True

    @staticmethod
    def _normalize_reasoning_mode(reasoning_mode: str) -> str:
        value = str(reasoning_mode).strip().lower()
        if value in {"thinking", "think", "reasoning", "slow", "deep"}:
            return "thinking"
        if value in {"fast", "quick", "low_latency", "low-latency"}:
            return "fast"
        if not value:
            return "fast"
        raise ValueError("reasoning_mode must be either 'fast' or 'thinking'")

    @staticmethod
    def _build_prompt(
        prompt: str,
        user_input: str,
        reasoning_mode: str,
        need_image: bool,
        camera_used: str,
    ) -> str:
        return (
            "Return JSON only with exactly these keys:\n"
            '{"speech_text": string, "data_text": object}\n'
            "speech_text is what the robot should say aloud.\n"
            "data_text is the structured or reasoning result for downstream handling.\n"
            "data_text must be a JSON object, not a string.\n\n"
            "Example response:\n"
            '{"speech_text":"Thank you John, I have noted your favourite drink as tea.","data_text":{"task":"Both","reason":"The guest provided both name and drink.","entities":{"name":"John","drink":"tea"}}}\n\n'
            f"Task prompt:\n{prompt.strip()}\n\n"
            f"User input or command:\n{user_input.strip()}\n\n"
            f"Reasoning mode requested: {reasoning_mode}\n"
            f"Image attached: {'yes' if need_image else 'no'}\n"
            f"Camera used: {camera_used or 'none'}\n\n"
            "Do not merge speech_text and data_text into one field."
        )

    @staticmethod
    def _encode_image(image_bgr: Any) -> str:
        ok, encoded = cv2.imencode(".jpg", image_bgr)
        if not ok:
            raise RuntimeError("Failed to encode image for VLM request")
        return base64.b64encode(encoded.tobytes()).decode("ascii")

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
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"Unable to start ollama serve: {exc}") from exc

        deadline = time.time() + self.ollama_start_timeout_sec
        while time.time() < deadline:
            if self._ollama_ready(timeout=1.0):
                return
            time.sleep(0.5)
        raise RuntimeError(f"Ollama is not reachable at {self.ollama_base_url}")


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = VlmQueryServiceNode()
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

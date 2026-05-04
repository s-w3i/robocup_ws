#!/usr/bin/env python3
"""Dedicated lost-and-found VLM service.

This node keeps the generic /vlm/query service unchanged. It accepts the
lost-and-found-specific request shape, can check a saved local image directly,
and falls back to the existing camera-based VlmQuery service when no image path
is supplied.
"""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import cv2
import requests
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node

from vlm_interfaces.srv import CaptureImage, LostFoundVlmCheck, VisualQuestion, VlmQuery


DUAL_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "speech_text": {"type": "string"},
        "data_text": {"type": "object"},
    },
    "required": ["speech_text", "data_text"],
}

DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_OPENAI_API_URL = "https://api.openai.com/v1/responses"
DEFAULT_OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
DEFAULT_OPENAI_MODEL = "gpt-5.5"

LOST_FOUND_VLM_PROMPT = """You are checking whether a specific object is visible in a robot camera image.
Return JSON only.

Set speech_text to a very short natural sentence that only says whether the object was found.

Set data_text to a JSON object with exactly these top-level keys:
- task
- complete
- reason
- entities

Set task to "lost_found_check".
Set complete to true when the image was checked.
Set reason to a very short explanation grounded only in visible image evidence.

Set entities to a JSON object with exactly these keys:
- object
- found
- confidence
- evidence
- bbox_2d

Use the requested object name.
Ignore room/location labels such as bedroom, kitchen, table, desk, etc.
Set found to true only if the requested object is visibly present anywhere in the image.
Set confidence from 0.0 to 1.0.
Set evidence to a short visible clue, or null if not found.
Set bbox_2d to [x1, y1, x2, y2] using normalized image coordinates from 0 to 1000 when the object is visible.
Use null for bbox_2d if the object is not visible or the box is uncertain.
Do not describe the whole scene.
If found, speech_text should be similar to "I found the bottle."
If not found, speech_text should be similar to "I did not find the bottle."
Do not guess beyond the image.
"""

VISUAL_QUESTION_PROMPT = """You answer a user's visual question using only the robot camera image.
Return JSON only.

Set speech_text to a very short natural answer for the user.

Set data_text to a JSON object with exactly these top-level keys:
- task
- complete
- reason
- entities

Set task to "visual_question".
Set complete to true when the image was checked.
Set reason to a very short explanation grounded only in visible image evidence.

Set entities to a JSON object with exactly these keys:
- object
- found
- confidence
- evidence
- bbox_2d

If a target object is provided, search for that object. If no target object is provided,
infer the target object from the user's question when possible.
Set found to true only if the target object is visibly present.
Set confidence from 0.0 to 1.0.
Set evidence to a short visible clue, or null if not found.
Set bbox_2d to [x1, y1, x2, y2] using normalized image coordinates from 0 to 1000 when the target is visible.
Use null for bbox_2d if the target is not visible or the box is uncertain.
Do not describe the whole scene.
If a target object is provided, answer mainly whether it is found or not.
Do not guess beyond the image.
"""


class LostFoundVlmCheckNode(Node):
    def __init__(self) -> None:
        super().__init__("lost_found_vlm_check_node")

        self.declare_parameter("service_name", "/lost_found/vlm_check")
        self.declare_parameter("visual_question_service_name", "/lost_found/visual_question")
        self.declare_parameter("vlm_query_service", "/vlm/query")
        self.declare_parameter("capture_image_service", "/camera/capture")
        self.declare_parameter("default_camera_name", "camera0")
        self.declare_parameter("capture_before_visual_question", True)
        self.declare_parameter("visual_question_save_dir", "/home/usern/robocup_ws/captures")
        self.declare_parameter("ollama_base_url", os.environ.get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL))
        self.declare_parameter("use_openai_vlm", False)
        self.declare_parameter("openai_api_url", os.environ.get("OPENAI_API_URL", DEFAULT_OPENAI_API_URL))
        self.declare_parameter("openai_api_key_env", os.environ.get("OPENAI_API_KEY_ENV", DEFAULT_OPENAI_API_KEY_ENV))
        self.declare_parameter("openai_model", os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL))
        self.declare_parameter("openai_image_detail", "auto")
        self.declare_parameter("vlm_model", os.environ.get("VISION_MODEL", "qwen3.5:9b"))
        self.declare_parameter("vlm_keep_alive", os.environ.get("OLLAMA_KEEP_ALIVE", "30m"))
        self.declare_parameter("system_prompt", "You are a robotics vision-language assistant.")
        self.declare_parameter("ollama_timeout_sec", 45.0)
        self.declare_parameter("ollama_num_ctx", 4096)
        self.declare_parameter("ollama_num_predict", 256)
        self.declare_parameter("auto_start_ollama", True)
        self.declare_parameter("ollama_start_timeout_sec", 20.0)
        self.declare_parameter("default_confidence_threshold", 0.65)
        self.declare_parameter("vlm_query_timeout_sec", 75.0)
        self.declare_parameter("service_wait_timeout_sec", 5.0)

        self.service_name = str(self.get_parameter("service_name").value).strip() or "/lost_found/vlm_check"
        self.visual_question_service_name = (
            str(self.get_parameter("visual_question_service_name").value).strip()
            or "/lost_found/visual_question"
        )
        self.vlm_query_service = str(self.get_parameter("vlm_query_service").value).strip() or "/vlm/query"
        self.capture_image_service = str(self.get_parameter("capture_image_service").value).strip() or "/camera/capture"
        self.default_camera_name = str(self.get_parameter("default_camera_name").value).strip() or "camera0"
        self.capture_before_visual_question = bool(self.get_parameter("capture_before_visual_question").value)
        self.visual_question_save_dir = (
            str(self.get_parameter("visual_question_save_dir").value).strip()
            or "/home/usern/robocup_ws/captures"
        )
        self.ollama_base_url = str(self.get_parameter("ollama_base_url").value).strip().rstrip("/")
        self.use_openai_vlm = bool(self.get_parameter("use_openai_vlm").value)
        self.openai_api_url = str(self.get_parameter("openai_api_url").value).strip() or DEFAULT_OPENAI_API_URL
        self.openai_api_key_env = (
            str(self.get_parameter("openai_api_key_env").value).strip() or DEFAULT_OPENAI_API_KEY_ENV
        )
        self.openai_model = str(self.get_parameter("openai_model").value).strip() or DEFAULT_OPENAI_MODEL
        self.openai_image_detail = str(self.get_parameter("openai_image_detail").value).strip() or "auto"
        self.vlm_model = str(self.get_parameter("vlm_model").value).strip()
        self.vlm_keep_alive = str(self.get_parameter("vlm_keep_alive").value).strip()
        self.system_prompt = str(self.get_parameter("system_prompt").value).strip()
        self.ollama_timeout_sec = max(1.0, float(self.get_parameter("ollama_timeout_sec").value))
        self.ollama_num_ctx = max(512, int(self.get_parameter("ollama_num_ctx").value))
        self.ollama_num_predict = max(32, int(self.get_parameter("ollama_num_predict").value))
        self.auto_start_ollama = bool(self.get_parameter("auto_start_ollama").value)
        self.ollama_start_timeout_sec = max(1.0, float(self.get_parameter("ollama_start_timeout_sec").value))
        self.default_confidence_threshold = max(
            0.0,
            min(1.0, float(self.get_parameter("default_confidence_threshold").value)),
        )
        self.vlm_query_timeout_sec = max(1.0, float(self.get_parameter("vlm_query_timeout_sec").value))
        self.service_wait_timeout_sec = max(0.1, float(self.get_parameter("service_wait_timeout_sec").value))

        self._callback_group = ReentrantCallbackGroup()
        self._vlm_query_client = self.create_client(
            VlmQuery,
            self.vlm_query_service,
            callback_group=self._callback_group,
        )
        self._capture_image_client = self.create_client(
            CaptureImage,
            self.capture_image_service,
            callback_group=self._callback_group,
        )
        self.create_service(
            LostFoundVlmCheck,
            self.service_name,
            self._handle_check,
            callback_group=self._callback_group,
        )
        self.create_service(
            VisualQuestion,
            self.visual_question_service_name,
            self._handle_visual_question,
            callback_group=self._callback_group,
        )

        if self.auto_start_ollama and not self.use_openai_vlm:
            self._ensure_ollama_running()

        self.get_logger().info(
            f"Lost-found VLM check service ready on {self.service_name} | "
            f"visual_question_service={self.visual_question_service_name} | "
            f"default_camera={self.default_camera_name} | "
            f"backend={'openai' if self.use_openai_vlm else 'ollama'} | "
            f"model={self.openai_model if self.use_openai_vlm else self.vlm_model}"
        )

    def _handle_check(
        self,
        request: LostFoundVlmCheck.Request,
        response: LostFoundVlmCheck.Response,
    ) -> LostFoundVlmCheck.Response:
        response.success = False
        response.message = ""
        response.found = False
        response.confidence = 0.0
        response.reason = ""
        response.image_path = str(request.image_path).strip()
        response.camera_used = str(request.camera_name).strip() or self.default_camera_name
        response.data_text = ""

        object_name = str(request.object_name).strip()
        location_name = str(request.location_name).strip()
        if not object_name or not location_name:
            response.message = "object_name and location_name are required."
            return response

        threshold = float(request.confidence_threshold)
        if threshold <= 0.0:
            threshold = self.default_confidence_threshold
        threshold = max(0.0, min(1.0, threshold))

        try:
            image_path = str(request.image_path).strip()
            if image_path:
                data_text = self._query_image_path(object_name, location_name, image_path)
            else:
                data_text = self._query_vlm_camera(object_name, location_name, response.camera_used)
            found, confidence, reason = self._parse_decision(data_text, threshold)

            response.success = True
            response.message = "ok"
            response.found = found
            response.confidence = float(confidence)
            response.reason = reason
            response.data_text = data_text
            return response
        except Exception as exc:
            response.message = str(exc)
            self.get_logger().error(f"Lost-found VLM check failed: {exc}")
            return response

    def _handle_visual_question(
        self,
        request: VisualQuestion.Request,
        response: VisualQuestion.Response,
    ) -> VisualQuestion.Response:
        response.success = False
        response.message = ""
        response.reply_text = ""
        response.found = False
        response.confidence = 0.0
        response.reason = ""
        response.image_path = str(request.image_path).strip()
        response.camera_used = str(request.camera_name).strip() or self.default_camera_name
        response.data_text = ""

        question = str(request.question).strip()
        object_name = str(request.object_name).strip()
        if not question:
            if object_name:
                question = f"Can you see {object_name}?"
            else:
                response.message = "question or object_name is required."
                return response

        threshold = float(request.confidence_threshold)
        if threshold <= 0.0:
            threshold = self.default_confidence_threshold
        threshold = max(0.0, min(1.0, threshold))

        try:
            image_path = str(request.image_path).strip()
            captured_for_reply = ""
            if not image_path and self.capture_before_visual_question:
                captured_for_reply = self._capture_image(response.camera_used, "visual_question")
                image_path = captured_for_reply
            if image_path:
                speech_text, data_text = self._query_visual_question_image_path(
                    question,
                    object_name,
                    image_path,
                )
            else:
                speech_text, data_text = self._query_vlm_visual_question_camera(
                    question,
                    object_name,
                    response.camera_used,
                )
            found, confidence, reason = self._parse_decision(data_text, threshold)

            response.success = True
            response.message = "ok"
            response.reply_text = speech_text or self._default_visual_reply(object_name, found, reason)
            response.found = found
            response.confidence = float(confidence)
            response.reason = reason
            response.image_path = image_path or captured_for_reply
            response.data_text = data_text
            return response
        except Exception as exc:
            response.message = str(exc)
            self.get_logger().error(f"Visual question failed: {exc}")
            return response

    def _query_image_path(self, object_name: str, location_name: str, image_path: str) -> str:
        try:
            if self.use_openai_vlm:
                return self._query_openai_image_path(object_name, location_name, image_path)
            return self._query_ollama_image_path(object_name, location_name, image_path)
        except Exception as exc:
            # If direct image-path VLM parsing fails, reuse the camera-service path
            # so one malformed response does not fail the whole location check.
            self.get_logger().warn(
                "Direct image-path query failed; falling back to /vlm/query camera path. "
                f"reason={exc}"
            )
            return self._query_vlm_camera(object_name, location_name, self.default_camera_name)

    def _query_visual_question_image_path(
        self,
        question: str,
        object_name: str,
        image_path: str,
    ) -> tuple[str, str]:
        if self.use_openai_vlm:
            return self._query_openai_visual_question_image_path(question, object_name, image_path)
        return self._query_ollama_visual_question_image_path(question, object_name, image_path)

    def _query_ollama_image_path(self, object_name: str, location_name: str, image_path: str) -> str:
        image_b64 = self._encode_image_path(image_path)
        prompt_text = self._build_prompt(object_name, location_name, image_attached=True)
        payload: dict[str, Any] = {
            "model": self.vlm_model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt_text, "images": [image_b64]},
            ],
            "stream": False,
            "keep_alive": self.vlm_keep_alive,
            "think": False,
            "format": DUAL_RESPONSE_SCHEMA,
            "options": {
                "temperature": 0.0,
                "num_ctx": self.ollama_num_ctx,
                "num_predict": self.ollama_num_predict,
            },
        }
        result = requests.post(
            f"{self.ollama_base_url}/api/chat",
            json=payload,
            timeout=self.ollama_timeout_sec,
        )
        result.raise_for_status()
        content = str(result.json().get("message", {}).get("content", "") or "").strip()
        return self._extract_data_text(content)

    def _query_openai_image_path(self, object_name: str, location_name: str, image_path: str) -> str:
        image_b64 = self._encode_image_path(image_path)
        prompt_text = self._build_prompt(object_name, location_name, image_attached=True)
        output_text = self._query_openai_content(prompt_text, image_b64)
        return self._extract_data_text(output_text)

    def _query_ollama_visual_question_image_path(
        self,
        question: str,
        object_name: str,
        image_path: str,
    ) -> tuple[str, str]:
        image_b64 = self._encode_image_path(image_path)
        prompt_text = self._build_visual_question_prompt(
            question,
            object_name,
            image_attached=True,
        )
        payload: dict[str, Any] = {
            "model": self.vlm_model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt_text, "images": [image_b64]},
            ],
            "stream": False,
            "keep_alive": self.vlm_keep_alive,
            "think": False,
            "format": DUAL_RESPONSE_SCHEMA,
            "options": {
                "temperature": 0.0,
                "num_ctx": self.ollama_num_ctx,
                "num_predict": self.ollama_num_predict,
            },
        }
        result = requests.post(
            f"{self.ollama_base_url}/api/chat",
            json=payload,
            timeout=self.ollama_timeout_sec,
        )
        result.raise_for_status()
        content = str(result.json().get("message", {}).get("content", "") or "").strip()
        return self._extract_dual_text(content)

    def _query_openai_visual_question_image_path(
        self,
        question: str,
        object_name: str,
        image_path: str,
    ) -> tuple[str, str]:
        image_b64 = self._encode_image_path(image_path)
        prompt_text = self._build_visual_question_prompt(
            question,
            object_name,
            image_attached=True,
        )
        output_text = self._query_openai_content(prompt_text, image_b64)
        return self._extract_dual_text(output_text)

    def _query_openai_content(self, prompt_text: str, image_b64: str) -> str:
        last_error: Exception | None = None
        attempt_specs = [
            (self.ollama_num_predict, self.ollama_timeout_sec),
            (max(self.ollama_num_predict * 2, 384), max(self.ollama_timeout_sec, 60.0)),
        ]
        for attempt_index, (max_output_tokens, timeout_sec) in enumerate(attempt_specs, start=1):
            payload: dict[str, Any] = {
                "model": self.openai_model,
                "instructions": "Return only JSON that matches the requested schema.",
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt_text},
                            {
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{image_b64}",
                                "detail": self.openai_image_detail,
                            },
                        ],
                    }
                ],
                "text": {"format": {"type": "json_object"}},
                "max_output_tokens": max_output_tokens,
                "store": False,
            }
            try:
                result = requests.post(
                    self.openai_api_url,
                    headers={
                        "Authorization": f"Bearer {self._openai_api_key()}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=timeout_sec,
                )
                if not result.ok:
                    raise RuntimeError(f"OpenAI request failed: HTTP {result.status_code} {result.text[:500]}")
                response_payload = result.json()
                output_text = self._extract_openai_output_text(response_payload)
                if output_text:
                    # Validate that we can parse the returned JSON shape before
                    # accepting it, so malformed/truncated text triggers retry.
                    _ = self._parse_json_relaxed(output_text)
                    return output_text
                payload_snippet = json.dumps(response_payload, ensure_ascii=False)[:700]
                raise RuntimeError(
                    "OpenAI response did not contain output text. "
                    f"payload_snippet={payload_snippet}"
                )
            except Exception as exc:
                last_error = exc
                if attempt_index < len(attempt_specs):
                    self.get_logger().warn(
                        "OpenAI image query returned empty or malformed content; retrying once with a larger output budget."
                    )
                    continue
                break

        raise RuntimeError(str(last_error) if last_error is not None else "OpenAI image query failed")

    def _query_vlm_camera(self, object_name: str, location_name: str, camera_name: str) -> str:
        if not self._vlm_query_client.wait_for_service(timeout_sec=self.service_wait_timeout_sec):
            raise RuntimeError(f"VLM query service is not available: {self.vlm_query_service}")

        request = VlmQuery.Request()
        request.need_image = True
        request.camera_name = camera_name or self.default_camera_name
        request.prompt = LOST_FOUND_VLM_PROMPT
        request.reasoning_mode = "fast"
        request.user_input = f"Check whether the user's {object_name} is visible in this image."
        request.request_profile = "vision_gate"
        request.max_retry_count = 0
        request.json_repair_mode = 1
        request.num_predict_override = self.ollama_num_predict
        request.timeout_sec_override = min(self.vlm_query_timeout_sec, 45.0)
        future = self._vlm_query_client.call_async(request)
        deadline = time.monotonic() + self.vlm_query_timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            if future.done():
                if future.exception() is not None:
                    raise future.exception()
                result = future.result()
                if result is None:
                    raise RuntimeError("VLM query service returned no response.")
                if not result.success:
                    raise RuntimeError(result.message or "VLM query service failed.")
                return str(result.data_text).strip()
            time.sleep(0.05)
        raise RuntimeError("Timed out waiting for VLM query response.")

    def _capture_image(self, camera_name: str, file_prefix: str) -> str:
        if not self._capture_image_client.wait_for_service(timeout_sec=self.service_wait_timeout_sec):
            self.get_logger().warn(
                f"Capture image service is not available: {self.capture_image_service}. "
                "Continuing without Telegram image attachment."
            )
            return ""

        request = CaptureImage.Request()
        request.camera_name = camera_name or self.default_camera_name
        request.save_dir = self.visual_question_save_dir
        request.file_prefix = file_prefix
        future = self._capture_image_client.call_async(request)
        deadline = time.monotonic() + min(self.vlm_query_timeout_sec, 10.0)
        while rclpy.ok() and time.monotonic() < deadline:
            if future.done():
                if future.exception() is not None:
                    raise future.exception()
                result = future.result()
                if result is None or not result.success:
                    message = "" if result is None else str(result.message)
                    self.get_logger().warn(f"Capture image failed: {message}")
                    return ""
                return str(result.saved_image_path).strip()
            time.sleep(0.05)
        self.get_logger().warn("Timed out waiting for capture image response.")
        return ""

    def _query_vlm_visual_question_camera(
        self,
        question: str,
        object_name: str,
        camera_name: str,
    ) -> tuple[str, str]:
        if not self._vlm_query_client.wait_for_service(timeout_sec=self.service_wait_timeout_sec):
            raise RuntimeError(f"VLM query service is not available: {self.vlm_query_service}")

        request = VlmQuery.Request()
        request.need_image = True
        request.camera_name = camera_name or self.default_camera_name
        request.prompt = self._build_visual_question_prompt(
            question,
            object_name,
            image_attached=True,
        )
        request.reasoning_mode = "fast"
        request.user_input = question
        request.request_profile = "vision_gate"
        request.max_retry_count = 0
        request.json_repair_mode = 1
        request.num_predict_override = self.ollama_num_predict
        request.timeout_sec_override = min(self.vlm_query_timeout_sec, 45.0)
        future = self._vlm_query_client.call_async(request)
        deadline = time.monotonic() + self.vlm_query_timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            if future.done():
                if future.exception() is not None:
                    raise future.exception()
                result = future.result()
                if result is None:
                    raise RuntimeError("VLM query service returned no response.")
                if not result.success:
                    raise RuntimeError(result.message or "VLM query service failed.")
                return str(result.speech_text).strip(), str(result.data_text).strip()
            time.sleep(0.05)
        raise RuntimeError("Timed out waiting for VLM query response.")

    def _build_prompt(self, object_name: str, location_name: str, image_attached: bool) -> str:
        return (
            "Return JSON only with exactly these keys:\n"
            '{"speech_text": string, "data_text": object}\n'
            "data_text must be a JSON object, not a string.\n\n"
            f"Task prompt:\n{LOST_FOUND_VLM_PROMPT}\n\n"
            f"Requested object: {object_name}\n"
            "Requested location: ignore this field for detection, use image evidence only.\n"
            f"Image attached: {'yes' if image_attached else 'no'}\n"
        )

    def _build_visual_question_prompt(self, question: str, object_name: str, image_attached: bool) -> str:
        return (
            "Return JSON only with exactly these keys:\n"
            '{"speech_text": string, "data_text": object}\n'
            "data_text must be a JSON object, not a string.\n\n"
            f"Task prompt:\n{VISUAL_QUESTION_PROMPT}\n\n"
            f"User question: {question}\n"
            f"Target object, if provided: {object_name or 'none'}\n"
            f"Image attached: {'yes' if image_attached else 'no'}\n"
        )

    @staticmethod
    def _extract_data_text(content: str) -> str:
        parsed = LostFoundVlmCheckNode._normalize_structured_result(
            LostFoundVlmCheckNode._parse_json_relaxed(content)
        )
        data_text = parsed.get("data_text", "")
        if isinstance(data_text, str):
            stripped = data_text.strip()
            try:
                nested = LostFoundVlmCheckNode._parse_json_relaxed(stripped)
            except Exception:
                return stripped
            return json.dumps(LostFoundVlmCheckNode._normalize_structured_result(nested), ensure_ascii=False)
        if isinstance(data_text, dict):
            return json.dumps(LostFoundVlmCheckNode._normalize_structured_result(data_text), ensure_ascii=False)
        if any(key in parsed for key in ("task", "reason", "entities")):
            return json.dumps(parsed, ensure_ascii=False)
        raise RuntimeError("VLM response did not contain data_text.")

    @staticmethod
    def _extract_dual_text(content: str) -> tuple[str, str]:
        parsed = LostFoundVlmCheckNode._normalize_structured_result(
            LostFoundVlmCheckNode._parse_json_relaxed(content)
        )
        speech_text = str(parsed.get("speech_text") or "").strip()
        data_text = parsed.get("data_text", "")
        if isinstance(data_text, str):
            stripped = data_text.strip()
            try:
                nested = LostFoundVlmCheckNode._parse_json_relaxed(stripped)
            except Exception:
                return speech_text, stripped
            return (
                speech_text,
                json.dumps(LostFoundVlmCheckNode._normalize_structured_result(nested), ensure_ascii=False),
            )
        if isinstance(data_text, dict):
            return (
                speech_text,
                json.dumps(LostFoundVlmCheckNode._normalize_structured_result(data_text), ensure_ascii=False),
            )
        if any(key in parsed for key in ("task", "reason", "entities")):
            return speech_text, json.dumps(parsed, ensure_ascii=False)
        raise RuntimeError("VLM response did not contain data_text.")

    @staticmethod
    def _default_visual_reply(object_name: str, found: bool, reason: str) -> str:
        if object_name:
            if found:
                return f"I can see {object_name}."
            return f"I cannot see {object_name}."
        return reason or "I checked the camera image."

    @staticmethod
    def _parse_decision(data_text: str, threshold: float) -> tuple[bool, float, str]:
        parsed = LostFoundVlmCheckNode._normalize_structured_result(
            LostFoundVlmCheckNode._parse_json_relaxed(data_text)
        )
        entities = parsed.get("entities")
        if not isinstance(entities, dict):
            raise RuntimeError("VLM data_text.entities is missing.")
        confidence_value = entities.get("confidence")
        confidence = max(0.0, min(1.0, LostFoundVlmCheckNode._float_from_any(confidence_value)))
        found_raw = LostFoundVlmCheckNode._bool_from_any(entities.get("found"))
        confidence_missing = confidence_value is None or str(confidence_value).strip() == ""
        found = found_raw and (confidence_missing or confidence >= threshold)
        if found_raw and confidence_missing:
            confidence = float(threshold)
        reason = str(parsed.get("reason") or entities.get("evidence") or "").strip()
        return found, confidence, reason

    @staticmethod
    def _parse_json_relaxed(text: str) -> dict[str, Any]:
        cleaned = str(text).strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            repaired = LostFoundVlmCheckNode._repair_json_tail(cleaned)
            if repaired != cleaned:
                try:
                    parsed = json.loads(repaired)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    pass
            match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            if match is None:
                raise
            candidate = match.group(0)
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                repaired = LostFoundVlmCheckNode._repair_json_tail(candidate)
                parsed = json.loads(repaired)
        if not isinstance(parsed, dict):
            raise RuntimeError("Parsed VLM JSON is not an object.")
        return parsed

    @staticmethod
    def _repair_json_tail(text: str) -> str:
        repaired = str(text).strip()
        if not repaired:
            return repaired

        literal_repairs = {
            r":\s*nul(?=(?:\s*[}\]])*\s*$)": ": null",
            r":\s*tru(?=(?:\s*[}\]])*\s*$)": ": true",
            r":\s*fals(?=(?:\s*[}\]])*\s*$)": ": false",
        }
        for pattern, replacement in literal_repairs.items():
            repaired = re.sub(pattern, replacement, repaired)

        if repaired.count('"') % 2 == 1:
            repaired += '"'

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
    def _bool_from_any(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        return str(value).strip().lower() in {"1", "true", "yes", "y", "found"}

    @staticmethod
    def _float_from_any(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _normalize_structured_result(parsed: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(parsed)
        entities = normalized.get("entities")
        if isinstance(entities, dict):
            return normalized

        reserved = {"task", "reason", "complete", "speech_text", "data_text"}
        candidate_entities: dict[str, Any] = {}
        for key, value in list(normalized.items()):
            if key in reserved:
                continue
            candidate_entities[key] = value
        if candidate_entities:
            normalized["entities"] = candidate_entities
        return normalized

    @staticmethod
    def _encode_image_path(image_path: str) -> str:
        path = Path(image_path).expanduser()
        if not path.is_file():
            raise RuntimeError(f"Image path does not exist: {path}")
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image from path: {path}")
        ok, encoded = cv2.imencode(".jpg", image)
        if not ok:
            raise RuntimeError("Failed to encode image for VLM request.")
        return base64.b64encode(encoded.tobytes()).decode("ascii")

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
                elif content.get("type") == "refusal":
                    text = str(content.get("refusal") or "").strip()
                    if text:
                        parts.append(text)
                elif "json" in content and isinstance(content.get("json"), (dict, list)):
                    parts.append(json.dumps(content["json"], ensure_ascii=False))
                elif "value" in content and isinstance(content.get("value"), (dict, list)):
                    parts.append(json.dumps(content["value"], ensure_ascii=False))
        if parts:
            return "\n".join(parts).strip()
        if isinstance(payload.get("output"), dict):
            return json.dumps(payload["output"], ensure_ascii=False)
        if isinstance(payload, dict):
            for key in ("response", "result", "message"):
                value = payload.get(key)
                if isinstance(value, dict):
                    return json.dumps(value, ensure_ascii=False)
        return "\n".join(parts).strip()

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
            raise RuntimeError(f"Unable to start ollama serve: {exc}") from exc

        deadline = time.monotonic() + self.ollama_start_timeout_sec
        while time.monotonic() < deadline:
            if self._ollama_ready(timeout=1.0):
                return
            time.sleep(0.5)
        raise RuntimeError(f"Ollama is not reachable at {self.ollama_base_url}")


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = LostFoundVlmCheckNode()
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

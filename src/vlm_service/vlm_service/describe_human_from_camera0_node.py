#!/usr/bin/env python3
"""ROS 2 client node that asks the VLM service to describe a person from camera0."""

from __future__ import annotations

import json
import os
from typing import Any

import rclpy
from rclpy.node import Node

from vlm_interfaces.srv import VlmQuery


VLM_QUERY_SERVICE = os.environ.get("VLM_QUERY_SERVICE", "/vlm/query")
DEFAULT_REASONING_MODE = os.environ.get("VLM_REASONING_MODE", "fast")

HUMAN_DESCRIPTION_PROMPT = """You are describing the main human visible in the image.
Return JSON only.

Set speech_text to a short one-sentence spoken summary.

Set data_text to a JSON object with exactly these top-level keys:
- task
- reason
- complete
- entities

Set task to "describe_human".
Set complete to true if a human is present, otherwise false.
Set reason to a short explanation.

Set entities to a JSON object with exactly these keys:
- gender
- cloth_color
- pant_color
- wearing_glasses
- hair_color

Use short values.
If a value is unclear or no human is present, use null.
For wearing_glasses use true, false, or null.
Describe only what is visible in the image and do not guess beyond the image.
"""


class DescribeHumanFromCamera0Node(Node):
    def __init__(self) -> None:
        super().__init__("describe_human_from_camera0_node")

        self.declare_parameter("vlm_query_service", VLM_QUERY_SERVICE)
        self.declare_parameter("camera_name", "camera0")
        self.declare_parameter("reasoning_mode", DEFAULT_REASONING_MODE)
        self.declare_parameter("service_wait_timeout_sec", 5.0)
        self.declare_parameter("request_timeout_sec", 30.0)

        self.vlm_query_service = str(self.get_parameter("vlm_query_service").value).strip() or VLM_QUERY_SERVICE
        self.camera_name = str(self.get_parameter("camera_name").value).strip() or "camera0"
        self.reasoning_mode = str(self.get_parameter("reasoning_mode").value).strip() or "fast"
        self.service_wait_timeout_sec = max(0.1, float(self.get_parameter("service_wait_timeout_sec").value))
        self.request_timeout_sec = max(1.0, float(self.get_parameter("request_timeout_sec").value))

        self._client = self.create_client(VlmQuery, self.vlm_query_service)

    def run_once(self) -> int:
        self.get_logger().info(
            f"Waiting for VLM query service '{self.vlm_query_service}' using camera '{self.camera_name}'"
        )
        if not self._client.wait_for_service(timeout_sec=self.service_wait_timeout_sec):
            self.get_logger().error(f"VLM query service '{self.vlm_query_service}' is not available.")
            return 1

        request = VlmQuery.Request()
        request.need_image = True
        request.camera_name = self.camera_name
        request.prompt = HUMAN_DESCRIPTION_PROMPT
        request.reasoning_mode = self.reasoning_mode
        request.user_input = "Describe the person in the current image."
        request.request_profile = "vision_strict"
        request.max_retry_count = 1
        request.json_repair_mode = 1
        request.num_predict_override = 0
        request.timeout_sec_override = 0.0

        self.get_logger().info("Sending image description request to VLM service.")
        future = self._client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=self.request_timeout_sec)

        if not future.done():
            self.get_logger().error(
                f"Timed out waiting for VLM response after {self.request_timeout_sec:.1f}s."
            )
            return 1

        exc = future.exception()
        if exc is not None:
            self.get_logger().error(f"VLM request failed: {exc}")
            return 1

        response = future.result()
        if response is None:
            self.get_logger().error("VLM service returned no response.")
            return 1
        if not response.success:
            self.get_logger().error(f"VLM service error: {response.message}")
            return 1

        self.get_logger().info(
            f"VLM response received | model={response.model_name} camera_used={response.camera_used}"
        )
        if response.speech_text:
            self.get_logger().info(f"speech_text: {response.speech_text}")
        if response.data_text:
            self.get_logger().info(f"data_text: {response.data_text}")
            parsed = self._parse_json(response.data_text)
            if parsed is not None:
                entities = parsed.get("entities", {})
                self.get_logger().info(
                    "person description | "
                    f"gender={entities.get('gender')} "
                    f"cloth_color={entities.get('cloth_color')} "
                    f"pant_color={entities.get('pant_color')} "
                    f"wearing_glasses={entities.get('wearing_glasses')} "
                    f"hair_color={entities.get('hair_color')}"
                )
        return 0

    def _parse_json(self, text: str) -> dict[str, Any] | None:
        try:
            parsed = json.loads(text)
        except Exception as exc:
            self.get_logger().warn(f"Failed to parse data_text as JSON: {exc}")
            return None
        if not isinstance(parsed, dict):
            self.get_logger().warn("Parsed data_text is not a JSON object.")
            return None
        return parsed


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = DescribeHumanFromCamera0Node()
    exit_code = 1
    try:
        exit_code = node.run_once()
    finally:
        node.destroy_node()
        rclpy.shutdown()
    raise SystemExit(exit_code)

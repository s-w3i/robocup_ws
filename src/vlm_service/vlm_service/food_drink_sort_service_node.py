#!/usr/bin/env python3
"""Service node that classifies visible items into food and drinks from camera input."""

from __future__ import annotations

import json
import re
import threading
from typing import Any

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node

from vlm_interfaces.srv import CaptureImage, FoodDrinkSort, VlmQuery


PROMPT = """You are a robot vision assistant.
Return JSON only with exactly:
{"speech_text": string, "data_text": object}

Task:
- Identify visible items that are FOOD and DRINKS in the current camera image.
- If uncertain, only include items with reasonable confidence.
- Do not invent unseen items.

Set data_text as an object with keys:
- task: "food_drink_sort"
- reason: short explanation from image
- complete: true if at least one relevant item category detected, else false
- entities: object with keys:
  - food_items: array of strings
  - drink_items: array of strings
  - confidence: number from 0.0 to 1.0 (or "low"/"medium"/"high")
"""


class FoodDrinkSortServiceNode(Node):
    def __init__(self) -> None:
        super().__init__("food_drink_sort_service_node")

        self.declare_parameter("service_name", "/food_drink/sort")
        self.declare_parameter("capture_image_service", "/camera/capture")
        self.declare_parameter("vlm_query_service", "/vlm/query")
        self.declare_parameter("default_camera_name", "camera0")
        self.declare_parameter("capture_save_dir", "/home/usern/robocup_ws/captures")
        self.declare_parameter("service_wait_timeout_sec", 5.0)
        self.declare_parameter("query_timeout_sec", 60.0)
        self.declare_parameter("default_confidence_threshold", 0.25)

        self.service_name = str(self.get_parameter("service_name").value).strip() or "/food_drink/sort"
        self.capture_image_service = (
            str(self.get_parameter("capture_image_service").value).strip() or "/camera/capture"
        )
        self.vlm_query_service = str(self.get_parameter("vlm_query_service").value).strip() or "/vlm/query"
        self.default_camera_name = str(self.get_parameter("default_camera_name").value).strip() or "camera0"
        self.capture_save_dir = str(self.get_parameter("capture_save_dir").value).strip() or "/home/usern/robocup_ws/captures"
        self.service_wait_timeout_sec = max(0.1, float(self.get_parameter("service_wait_timeout_sec").value))
        self.query_timeout_sec = max(1.0, float(self.get_parameter("query_timeout_sec").value))
        self.default_confidence_threshold = max(
            0.0,
            min(1.0, float(self.get_parameter("default_confidence_threshold").value)),
        )

        self._callback_group = ReentrantCallbackGroup()
        self._capture_client = self.create_client(
            CaptureImage, self.capture_image_service, callback_group=self._callback_group
        )
        self._vlm_query_client = self.create_client(
            VlmQuery, self.vlm_query_service, callback_group=self._callback_group
        )
        self.create_service(
            FoodDrinkSort,
            self.service_name,
            self._handle_food_drink_sort,
            callback_group=self._callback_group,
        )
        self.get_logger().info(
            "Food-drink sort service ready on %s | capture=%s | vlm=%s | default_camera=%s"
            % (
                self.service_name,
                self.capture_image_service,
                self.vlm_query_service,
                self.default_camera_name,
            )
        )

    def _handle_food_drink_sort(
        self,
        request: FoodDrinkSort.Request,
        response: FoodDrinkSort.Response,
    ) -> FoodDrinkSort.Response:
        response.success = False
        response.message = ""
        response.camera_used = ""
        response.image_path = ""
        response.announcement = ""
        response.food_items = []
        response.drink_items = []
        response.data_text = ""

        try:
            camera_name = str(request.camera_name).strip() or self.default_camera_name
            threshold = float(request.confidence_threshold)
            if threshold <= 0.0:
                threshold = self.default_confidence_threshold

            image_path = self._capture_image(camera_name)
            if not image_path:
                response.message = "Failed to capture image from camera."
                return response

            speech_text, data_text = self._query_vlm(camera_name)
            parsed = self._parse_data_text(data_text)
            entities = parsed.get("entities", {}) if isinstance(parsed, dict) else {}
            food_items = self._extract_items(entities.get("food_items", []))
            drink_items = self._extract_items(entities.get("drink_items", []))
            confidence = self._confidence_to_float(entities.get("confidence"))

            if confidence < threshold and not (food_items or drink_items):
                response.message = f"Low confidence result ({confidence:.2f})."
                response.camera_used = camera_name
                response.image_path = image_path
                response.data_text = data_text
                return response

            announcement = self._build_announcement(food_items, drink_items, speech_text)
            response.success = True
            response.message = "ok"
            response.camera_used = camera_name
            response.image_path = image_path
            response.food_items = food_items
            response.drink_items = drink_items
            response.announcement = announcement
            response.data_text = data_text
            return response
        except Exception as exc:
            response.message = str(exc)
            self.get_logger().error(f"Food-drink sort failed: {exc}")
            return response

    def _capture_image(self, camera_name: str) -> str:
        if not self._capture_client.wait_for_service(timeout_sec=self.service_wait_timeout_sec):
            raise RuntimeError(f"Capture image service not available: {self.capture_image_service}")
        request = CaptureImage.Request()
        request.camera_name = camera_name
        request.save_dir = self.capture_save_dir
        request.file_prefix = "food_drink_sort"
        result = self._wait_for_future(self._capture_client.call_async(request), self.query_timeout_sec)
        if result is None or not bool(result.success):
            return ""
        return str(result.saved_image_path).strip()

    def _query_vlm(self, camera_name: str) -> tuple[str, str]:
        if not self._vlm_query_client.wait_for_service(timeout_sec=self.service_wait_timeout_sec):
            raise RuntimeError(f"VLM query service not available: {self.vlm_query_service}")
        request = VlmQuery.Request()
        request.need_image = True
        request.camera_name = camera_name
        request.prompt = PROMPT
        request.reasoning_mode = "fast"
        request.user_input = "Separate visible food items and drink items."
        request.request_profile = "vision_gate"
        request.max_retry_count = -1
        request.json_repair_mode = 1
        request.num_predict_override = 192
        request.timeout_sec_override = self.query_timeout_sec
        result = self._wait_for_future(self._vlm_query_client.call_async(request), self.query_timeout_sec + 10.0)
        if result is None:
            raise RuntimeError("VLM query returned no response.")
        if not bool(result.success):
            raise RuntimeError(str(result.message).strip() or "VLM query failed")
        return str(result.speech_text).strip(), str(result.data_text).strip()

    def _wait_for_future(self, future, timeout_sec: float) -> Any | None:
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

    @staticmethod
    def _parse_json_relaxed(text: str) -> dict[str, Any]:
        cleaned = str(text).strip()
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

    def _parse_data_text(self, data_text: str) -> dict[str, Any]:
        if not data_text:
            return {}
        parsed = self._parse_json_relaxed(data_text)
        if isinstance(parsed, dict):
            return parsed
        return {}

    @staticmethod
    def _extract_items(value: Any) -> list[str]:
        items: list[str] = []
        if isinstance(value, list):
            raw = value
        elif isinstance(value, str):
            raw = re.split(r"\s*,\s*", value.strip())
        else:
            raw = []
        seen: set[str] = set()
        for item in raw:
            text = str(item).strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            items.append(text)
        return items

    @staticmethod
    def _confidence_to_float(value: Any) -> float:
        if isinstance(value, (int, float)):
            return max(0.0, min(1.0, float(value)))
        text = str(value or "").strip().lower()
        if text == "high":
            return 0.85
        if text == "medium":
            return 0.6
        if text == "low":
            return 0.2
        try:
            return max(0.0, min(1.0, float(text)))
        except ValueError:
            return 0.5

    @staticmethod
    def _build_announcement(food_items: list[str], drink_items: list[str], fallback: str) -> str:
        if not food_items and not drink_items and fallback:
            return f"{fallback.strip()} I will put food on left, drinks on right."

        all_items = [*food_items, *drink_items]
        found_text = FoodDrinkSortServiceNode._natural_join(all_items)
        food_text = FoodDrinkSortServiceNode._natural_join(food_items) if food_items else "food items"
        drink_text = FoodDrinkSortServiceNode._natural_join(drink_items) if drink_items else "drink items"
        return (
            f"I found {found_text}. "
            f"I will put {food_text} on left, {drink_text} on right."
        )

    @staticmethod
    def _natural_join(items: list[str]) -> str:
        cleaned = [str(item).strip() for item in items if str(item).strip()]
        if not cleaned:
            return ""
        if len(cleaned) == 1:
            return cleaned[0]
        if len(cleaned) == 2:
            return f"{cleaned[0]} and {cleaned[1]}"
        return f"{', '.join(cleaned[:-1])}, and {cleaned[-1]}"


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = FoodDrinkSortServiceNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

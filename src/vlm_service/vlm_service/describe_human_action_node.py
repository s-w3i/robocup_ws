#!/usr/bin/env python3
"""ROS 2 action server that describes a visible person using camera0 and OpenAI."""

from __future__ import annotations

import base64
import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import py_trees
import rclpy
import requests
from coqui_tts_interfaces.action import SpeakText
from cv_bridge import CvBridge, CvBridgeError
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from vlm_interfaces.action import DescribeHuman


VLM_QUERY_SERVICE = os.environ.get("VLM_QUERY_SERVICE", "/vlm/query")
ACTION_NAME = os.environ.get("DESCRIBE_HUMAN_ACTION_NAME", "/describe_human")
SPEAK_ACTION_NAME = os.environ.get("SPEAK_ACTION_NAME", "/coqui_tts/speak")
OPENAI_API_URL = os.environ.get("OPENAI_API_URL", "https://api.openai.com/v1/responses")
OPENAI_API_KEY_ENV = os.environ.get("OPENAI_API_KEY_ENV", "OPENAI_API_KEY")
# Change this line to switch the default OpenAI model for this node.
OPENAI_MODEL = "gpt-5.5"
# Used when force_thinking is true for the human-description vision request.
OPENAI_THINKING_EFFORT = "medium"
Status = py_trees.common.Status

HUMAN_DESCRIPTION_PROMPT = """You are describing the main human visible in the image.
Return JSON only.

Set speech_text to a short one-sentence spoken summary.

Set data_text to a JSON object with exactly these top-level keys:
- task
- reason
- complete
- entities

Set task to "describe_human".
Set complete to true only when a human is present and the face is visible enough to describe the requested attributes.
Set reason to a short explanation.

Set entities to a JSON object with exactly these keys:
- human_present
- face_visible
- gender
- cloth_color
- pant_color
- wearing_glasses
- hair_color

Use short values.
If a value is unclear or no human is present, use null.
For human_present and face_visible use true, false, or null.
For wearing_glasses use true, false, or null.
Describe only what is visible in the image and do not guess beyond the image.
If a human is present but the face is not clearly visible, set human_present=true and face_visible=false.
Decision policy:
- If no human is visible, set complete=false, human_present=false, face_visible=null.
- If a human is visible but the face is not clearly visible, set complete=false, human_present=true, face_visible=false.
- Only if a human is visible and the face is clearly visible, set complete=true, human_present=true, face_visible=true.
"""

OPENAI_RESPONSE_FORMAT = {
    "type": "json_schema",
    "name": "describe_human_result",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "speech_text": {"type": "string"},
            "data_text": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "task": {"type": "string", "enum": ["describe_human"]},
                    "reason": {"type": "string"},
                    "complete": {"type": "boolean"},
                    "entities": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "human_present": {"type": ["boolean", "null"]},
                            "face_visible": {"type": ["boolean", "null"]},
                            "gender": {"type": ["string", "null"]},
                            "cloth_color": {"type": ["string", "null"]},
                            "pant_color": {"type": ["string", "null"]},
                            "wearing_glasses": {"type": ["boolean", "null"]},
                            "hair_color": {"type": ["string", "null"]},
                        },
                        "required": [
                            "human_present",
                            "face_visible",
                            "gender",
                            "cloth_color",
                            "pant_color",
                            "wearing_glasses",
                            "hair_color",
                        ],
                    },
                },
                "required": ["task", "reason", "complete", "entities"],
            },
        },
        "required": ["speech_text", "data_text"],
    },
}


@dataclass
class CameraFrame:
    image_bgr: Any
    received_monotonic: float


@dataclass
class OpenAIQueryResponse:
    speech_text: str
    data_text: str
    camera_used: str
    model_name: str


@dataclass
class DescribeHumanBlackboard:
    request: dict[str, Any] | None = None
    attempt_count: int = 0
    max_retries: int = 2
    last_error: str = ""
    last_reason: str = ""
    last_status: str = ""
    robot_text: str = ""
    should_retry: bool = False
    spoke_response: bool = False
    response_speech_text: str = ""
    response_data_text: str = ""
    camera_used: str = ""
    model_name: str = ""
    human_present: bool = False
    human_present_known: bool = False
    face_visible: bool = False
    face_visible_known: bool = False
    gender: str = ""
    cloth_color: str = ""
    pant_color: str = ""
    hair_color: str = ""
    wearing_glasses: bool = False
    wearing_glasses_known: bool = False
    result_payload: dict[str, Any] = field(default_factory=dict)
    finished: bool = False

    def clear_turn(self) -> None:
        self.last_error = ""
        self.last_reason = ""
        self.last_status = ""
        self.robot_text = ""
        self.should_retry = False
        self.spoke_response = False
        self.response_speech_text = ""
        self.response_data_text = ""
        self.camera_used = ""
        self.model_name = ""
        self.human_present = False
        self.human_present_known = False
        self.face_visible = False
        self.face_visible_known = False
        self.gender = ""
        self.cloth_color = ""
        self.pant_color = ""
        self.hair_color = ""
        self.wearing_glasses = False
        self.wearing_glasses_known = False
        self.result_payload = {}
        self.finished = False


class BlackboardCondition(py_trees.behaviour.Behaviour):
    def __init__(self, name: str, blackboard: DescribeHumanBlackboard, fn) -> None:
        super().__init__(name=name)
        self._blackboard = blackboard
        self._fn = fn

    def update(self) -> Status:
        return Status.SUCCESS if self._fn(self._blackboard) else Status.FAILURE


class BlackboardAction(py_trees.behaviour.Behaviour):
    def __init__(self, name: str, blackboard: DescribeHumanBlackboard, node: "DescribeHumanActionNode", fn) -> None:
        super().__init__(name=name)
        self._blackboard = blackboard
        self._node = node
        self._fn = fn

    def update(self) -> Status:
        return self._fn(self._blackboard, self._node)


class DescribeHumanActionNode(Node):
    def __init__(self) -> None:
        super().__init__("describe_human_action_node")

        self.declare_parameter("action_name", ACTION_NAME)
        self.declare_parameter("vlm_query_service", VLM_QUERY_SERVICE)
        self.declare_parameter("openai_api_url", OPENAI_API_URL)
        self.declare_parameter("openai_api_key_env", OPENAI_API_KEY_ENV)
        self.declare_parameter("openai_model", OPENAI_MODEL)
        self.declare_parameter("openai_thinking_effort", OPENAI_THINKING_EFFORT)
        self.declare_parameter("openai_timeout_sec", 90.0)
        self.declare_parameter("openai_max_output_tokens", 512)
        self.declare_parameter("openai_image_detail", "auto")
        self.declare_parameter("speak_action_name", SPEAK_ACTION_NAME)
        self.declare_parameter("default_camera_name", "camera0")
        self.declare_parameter("default_camera_topic", "/camera0/color/image_raw")
        self.declare_parameter("image_wait_timeout_sec", 3.0)
        self.declare_parameter("service_wait_timeout_sec", 5.0)
        self.declare_parameter("request_timeout_sec", 90.0)
        self.declare_parameter("speak_timeout_sec", 30.0)
        self.declare_parameter("max_retries", 2)
        self.declare_parameter("retry_wait_sec", 0.5)
        self.declare_parameter("force_thinking", True)
        self.declare_parameter("enable_speaking", True)
        self.declare_parameter("capture_start_prompt", "Capturing your image to register you into the system.")
        self.declare_parameter("no_human_retry_prompt", "Failed to detect a human, retrying.")
        self.declare_parameter("step_back_prompt", "Please step back a bit so I can see your face clearly.")
        self.declare_parameter("register_success_prompt", "Successfully register you into the system.")

        self.action_name = str(self.get_parameter("action_name").value).strip() or ACTION_NAME
        self.vlm_query_service = str(self.get_parameter("vlm_query_service").value).strip() or VLM_QUERY_SERVICE
        self.openai_api_url = str(self.get_parameter("openai_api_url").value).strip() or OPENAI_API_URL
        self.openai_api_key_env = str(self.get_parameter("openai_api_key_env").value).strip() or OPENAI_API_KEY_ENV
        self.openai_model = str(self.get_parameter("openai_model").value).strip() or OPENAI_MODEL
        self.openai_thinking_effort = (
            str(self.get_parameter("openai_thinking_effort").value).strip()
            or OPENAI_THINKING_EFFORT
        )
        self.openai_timeout_sec = max(1.0, float(self.get_parameter("openai_timeout_sec").value))
        self.openai_max_output_tokens = max(128, int(self.get_parameter("openai_max_output_tokens").value))
        self.openai_image_detail = str(self.get_parameter("openai_image_detail").value).strip() or "auto"
        self.speak_action_name = str(self.get_parameter("speak_action_name").value).strip() or SPEAK_ACTION_NAME
        self.default_camera_name = (
            str(self.get_parameter("default_camera_name").value).strip() or "camera0"
        )
        self.default_camera_topic = (
            str(self.get_parameter("default_camera_topic").value).strip() or "/camera0/color/image_raw"
        )
        self.image_wait_timeout_sec = max(0.1, float(self.get_parameter("image_wait_timeout_sec").value))
        self.service_wait_timeout_sec = max(0.1, float(self.get_parameter("service_wait_timeout_sec").value))
        self.request_timeout_sec = max(1.0, float(self.get_parameter("request_timeout_sec").value))
        self.speak_timeout_sec = max(1.0, float(self.get_parameter("speak_timeout_sec").value))
        self.max_retries = max(0, int(self.get_parameter("max_retries").value))
        self.retry_wait_sec = max(0.0, float(self.get_parameter("retry_wait_sec").value))
        self.force_thinking = bool(self.get_parameter("force_thinking").value)
        self.enable_speaking = bool(self.get_parameter("enable_speaking").value)
        self.capture_start_prompt = (
            str(self.get_parameter("capture_start_prompt").value).strip()
            or "Capturing your image to register you into the system."
        )
        self.no_human_retry_prompt = (
            str(self.get_parameter("no_human_retry_prompt").value).strip()
            or "Failed to detect a human, retrying."
        )
        self.step_back_prompt = (
            str(self.get_parameter("step_back_prompt").value).strip()
            or "Please step back a bit so I can see your face clearly."
        )
        self.register_success_prompt = (
            str(self.get_parameter("register_success_prompt").value).strip()
            or "Successfully register you into the system."
        )

        self._callback_group = ReentrantCallbackGroup()
        self._bridge = CvBridge()
        self._frame_lock = threading.Lock()
        self._latest_frame: CameraFrame | None = None
        self._image_subscription = self.create_subscription(
            Image,
            self.default_camera_topic,
            self._on_image,
            qos_profile_sensor_data,
            callback_group=self._callback_group,
        )
        self._speak_action_client = ActionClient(
            self,
            SpeakText,
            self.speak_action_name,
            callback_group=self._callback_group,
        )
        self._action_server = ActionServer(
            self,
            DescribeHuman,
            self.action_name,
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self._callback_group,
        )
        self.blackboard = DescribeHumanBlackboard(max_retries=self.max_retries)
        self._tree = py_trees.trees.BehaviourTree(root=self._create_tree())

        self.get_logger().info(
            f"Describe-human action ready on {self.action_name} | "
            f"openai_model={self.openai_model} | speak_action={self.speak_action_name} | "
            f"default_camera={self.default_camera_name} topic={self.default_camera_topic} | "
            f"max_retries={self.max_retries} | "
            f"force_thinking={self.force_thinking}"
        )

    def _create_tree(self) -> py_trees.behaviour.Behaviour:
        return py_trees.composites.Sequence(
            name="DescribeHumanTurn",
            memory=False,
            children=[
                BlackboardAction("QueryOpenAI", self.blackboard, self, self._bt_query_vlm),
                py_trees.composites.Selector(
                    name="Decision",
                    memory=False,
                    children=[
                        py_trees.composites.Sequence(
                            name="SuccessIfFaceVisible",
                            memory=False,
                            children=[
                                BlackboardCondition(
                                    "HumanPresent",
                                    self.blackboard,
                                    lambda bb: bb.human_present,
                                ),
                                BlackboardCondition(
                                    "FaceVisible",
                                    self.blackboard,
                                    lambda bb: bb.face_visible,
                                ),
                                BlackboardAction("FinishSuccess", self.blackboard, self, self._bt_finish_success),
                            ],
                        ),
                        py_trees.composites.Sequence(
                            name="RetryIfHumanButNoFace",
                            memory=False,
                            children=[
                                BlackboardCondition(
                                    "HumanPresentButNoFace",
                                    self.blackboard,
                                    lambda bb: bb.human_present and not bb.face_visible,
                                ),
                                BlackboardAction("PrepareStepBack", self.blackboard, self, self._bt_prepare_step_back),
                                BlackboardAction("SpeakStepBack", self.blackboard, self, self._bt_speak_text),
                                BlackboardAction("RetryAfterStepBack", self.blackboard, self, self._bt_retry),
                            ],
                        ),
                        py_trees.composites.Sequence(
                            name="RetryIfNoHuman",
                            memory=False,
                            children=[
                                BlackboardCondition(
                                    "NoHumanDetected",
                                    self.blackboard,
                                    lambda bb: not bb.human_present,
                                ),
                                BlackboardAction("PrepareNoHumanRetry", self.blackboard, self, self._bt_prepare_no_human_retry),
                                BlackboardAction("SpeakNoHumanRetry", self.blackboard, self, self._bt_speak_text),
                                BlackboardAction("RetryNoHuman", self.blackboard, self, self._bt_retry),
                            ],
                        ),
                        BlackboardAction("FinishFailure", self.blackboard, self, self._bt_finish_failure),
                    ],
                ),
            ],
        )

    def goal_callback(self, goal_request: DescribeHuman.Goal) -> GoalResponse:
        _ = goal_request
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle) -> CancelResponse:
        _ = goal_handle
        return CancelResponse.ACCEPT

    def _publish_feedback(self, goal_handle, stage: str, status: str) -> None:
        feedback = DescribeHuman.Feedback()
        feedback.stage = stage
        feedback.status = status
        goal_handle.publish_feedback(feedback)

    def _wait_for_future(self, future, timeout_sec: float) -> tuple[bool, Any, str]:
        deadline = time.time() + max(0.1, timeout_sec)
        while time.time() < deadline:
            if future.done():
                exc = future.exception()
                if exc is not None:
                    return False, None, str(exc)
                return True, future.result(), ""
            time.sleep(0.05)
        return False, None, "timeout"

    def _on_image(self, msg: Image) -> None:
        try:
            image = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            self.get_logger().warn(f"Failed to convert image from '{self.default_camera_name}': {exc}")
            return

        with self._frame_lock:
            self._latest_frame = CameraFrame(image_bgr=image, received_monotonic=time.monotonic())

    def _wait_for_frame(self, min_received_monotonic: float) -> CameraFrame | None:
        deadline = time.time() + self.image_wait_timeout_sec
        while time.time() < deadline:
            with self._frame_lock:
                frame = self._latest_frame
            if frame is not None and frame.received_monotonic >= min_received_monotonic:
                return frame
            time.sleep(0.05)
        return None

    @staticmethod
    def _encode_image(image_bgr: Any) -> str:
        ok, encoded = cv2.imencode(".jpg", image_bgr)
        if not ok:
            raise RuntimeError("Failed to encode image for OpenAI request.")
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
            for content in item.get("content", []):
                if not isinstance(content, dict):
                    continue
                if content.get("type") in {"output_text", "text"}:
                    text = str(content.get("text") or "")
                    if text:
                        parts.append(text)
        return "\n".join(parts).strip()

    def _build_request(self) -> dict[str, Any]:
        return {
            "camera_name": self.default_camera_name,
            "prompt": HUMAN_DESCRIPTION_PROMPT,
            "reasoning_mode": "thinking" if self.force_thinking else "fast",
            "user_input": (
                "First determine whether the image satisfies these conditions: "
                "a human must be present and the human face must be clearly visible. "
                "If either condition is not satisfied, report that in the JSON."
            ),
        }

    def _query_vlm(self, request: dict[str, Any] | None) -> OpenAIQueryResponse:
        request = request or {}
        request_started_monotonic = time.monotonic()
        frame = self._wait_for_frame(min_received_monotonic=request_started_monotonic)
        if frame is None:
            raise RuntimeError(
                f"No fresh image available from camera '{self.default_camera_name}'. "
                f"Check {self.default_camera_topic} or increase image_wait_timeout_sec."
            )

        image_b64 = self._encode_image(frame.image_bgr)
        prompt_text = str(request.get("prompt") or HUMAN_DESCRIPTION_PROMPT).strip()
        user_input = str(request.get("user_input") or "").strip()
        payload = {
            "model": self.openai_model,
            "instructions": "Return only JSON that matches the requested schema.",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": f"{prompt_text}\n\n{user_input}"},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{image_b64}",
                            "detail": self.openai_image_detail,
                        },
                    ],
                }
            ],
            "text": {"format": OPENAI_RESPONSE_FORMAT},
            "max_output_tokens": self.openai_max_output_tokens,
            "store": False,
        }
        if str(request.get("reasoning_mode") or "").strip() == "thinking":
            payload["reasoning"] = {"effort": self.openai_thinking_effort}
        response = requests.post(
            self.openai_api_url,
            headers={
                "Authorization": f"Bearer {self._openai_api_key()}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=min(self.openai_timeout_sec, self.request_timeout_sec),
        )
        if not response.ok:
            raise RuntimeError(f"OpenAI request failed: HTTP {response.status_code} {response.text[:500]}")

        response_payload = response.json()
        output_text = self._extract_openai_output_text(response_payload)
        if not output_text:
            raise RuntimeError("OpenAI response did not contain output text.")
        parsed = json.loads(output_text)
        if not isinstance(parsed, dict):
            raise RuntimeError("OpenAI response JSON is not an object.")

        data_text = parsed.get("data_text")
        if not isinstance(data_text, dict):
            raise RuntimeError("OpenAI response data_text is not a JSON object.")

        return OpenAIQueryResponse(
            speech_text=str(parsed.get("speech_text") or "").strip(),
            data_text=json.dumps(data_text, ensure_ascii=False),
            camera_used=self.default_camera_name,
            model_name=self.openai_model,
        )

    def _speak_text(self, text: str) -> tuple[bool, str]:
        cleaned = str(text).strip()
        if not cleaned:
            return True, "Nothing to speak."
        if not self.enable_speaking:
            return True, "Speaking disabled."
        if not self._speak_action_client.wait_for_server(timeout_sec=0.8):
            return False, f"Speak action server '{self.speak_action_name}' not ready."

        goal = SpeakText.Goal()
        goal.text = cleaned
        send_goal_future = self._speak_action_client.send_goal_async(goal)
        ok, goal_handle, error_text = self._wait_for_future(send_goal_future, self.speak_timeout_sec)
        if not ok:
            return False, f"Failed to send SpeakText goal: {error_text}"
        if goal_handle is None or not goal_handle.accepted:
            return False, "SpeakText goal rejected."

        result_future = goal_handle.get_result_async()
        ok, result_wrap, error_text = self._wait_for_future(result_future, self.speak_timeout_sec)
        if not ok:
            return False, f"SpeakText result wait failed: {error_text}"
        if result_wrap is None:
            return False, "SpeakText returned no result."
        result = result_wrap.result
        if result.success:
            return True, result.message
        return False, result.message

    @staticmethod
    def _normalize_str(value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    @staticmethod
    def _normalize_optional_bool(value: Any) -> tuple[bool, bool]:
        if isinstance(value, bool):
            return value, True
        if value is None:
            return False, False
        text = str(value).strip().lower()
        if text in {"true", "yes", "1"}:
            return True, True
        if text in {"false", "no", "0"}:
            return False, True
        return False, False

    def _parse_data_text(self, data_text: str) -> tuple[dict[str, Any], dict[str, Any]]:
        parsed = json.loads(data_text)
        if not isinstance(parsed, dict):
            raise RuntimeError("OpenAI data_text is not a JSON object.")
        entities = parsed.get("entities", {})
        if not isinstance(entities, dict):
            raise RuntimeError("OpenAI data_text.entities is not a JSON object.")
        return parsed, entities

    def _bt_query_vlm(self, blackboard: DescribeHumanBlackboard, node: "DescribeHumanActionNode") -> Status:
        blackboard.clear_turn()
        blackboard.attempt_count += 1
        reasoning_mode = "unknown"
        if blackboard.request is not None:
            reasoning_mode = str(blackboard.request.get("reasoning_mode") or "unknown")
        blackboard.last_status = (
            f"Attempt {blackboard.attempt_count}: querying OpenAI with reasoning_mode="
            f"{reasoning_mode}"
        )
        try:
            response = node._query_vlm(blackboard.request)
            blackboard.response_speech_text = str(response.speech_text)
            blackboard.response_data_text = str(response.data_text)
            blackboard.camera_used = str(response.camera_used)
            blackboard.model_name = str(response.model_name)
            parsed, entities = node._parse_data_text(blackboard.response_data_text)
            blackboard.result_payload = parsed
            blackboard.last_reason = node._normalize_str(parsed.get("reason"))
            blackboard.human_present, blackboard.human_present_known = node._normalize_optional_bool(
                entities.get("human_present")
            )
            blackboard.face_visible, blackboard.face_visible_known = node._normalize_optional_bool(
                entities.get("face_visible")
            )
            if not blackboard.human_present_known:
                blackboard.human_present = bool(parsed.get("complete"))
            if blackboard.human_present and not blackboard.face_visible_known:
                blackboard.face_visible = bool(parsed.get("complete"))
            if not blackboard.human_present:
                blackboard.face_visible = False
            blackboard.gender = node._normalize_str(entities.get("gender"))
            blackboard.cloth_color = node._normalize_str(entities.get("cloth_color"))
            blackboard.pant_color = node._normalize_str(entities.get("pant_color"))
            blackboard.hair_color = node._normalize_str(entities.get("hair_color"))
            blackboard.wearing_glasses, blackboard.wearing_glasses_known = node._normalize_optional_bool(
                entities.get("wearing_glasses")
            )
            if blackboard.human_present and blackboard.face_visible:
                blackboard.last_status = "Human and face visible. Finishing successfully."
            elif blackboard.human_present:
                blackboard.last_status = "Human present but face not visible enough."
            else:
                blackboard.last_status = "No human detected."
            return Status.SUCCESS
        except Exception as exc:
            blackboard.last_error = str(exc)
            blackboard.last_status = f"Attempt {blackboard.attempt_count}: query failed"
            return Status.FAILURE

    def _bt_prepare_step_back(self, blackboard: DescribeHumanBlackboard, node: "DescribeHumanActionNode") -> Status:
        _ = node
        blackboard.robot_text = self.step_back_prompt
        blackboard.last_status = "Face not visible, asking person to step backward."
        return Status.SUCCESS

    def _bt_prepare_no_human_retry(self, blackboard: DescribeHumanBlackboard, node: "DescribeHumanActionNode") -> Status:
        _ = node
        blackboard.robot_text = self.no_human_retry_prompt
        blackboard.last_status = "No human detected, announcing retry."
        return Status.SUCCESS

    def _bt_speak_text(self, blackboard: DescribeHumanBlackboard, node: "DescribeHumanActionNode") -> Status:
        ok, message = node._speak_text(blackboard.robot_text)
        blackboard.spoke_response = ok
        if not ok:
            blackboard.last_error = message
            return Status.FAILURE
        return Status.SUCCESS

    def _bt_retry(self, blackboard: DescribeHumanBlackboard, node: "DescribeHumanActionNode") -> Status:
        if blackboard.attempt_count > blackboard.max_retries:
            blackboard.should_retry = False
            if blackboard.human_present and not blackboard.face_visible:
                blackboard.last_error = blackboard.last_reason or "Human detected but face is not visible."
            else:
                blackboard.last_error = blackboard.last_reason or "No human detected."
            return Status.FAILURE
        blackboard.should_retry = True
        blackboard.last_status = f"Retrying after attempt {blackboard.attempt_count}"
        if node.retry_wait_sec > 0.0:
            time.sleep(node.retry_wait_sec)
        return Status.SUCCESS

    def _bt_finish_success(self, blackboard: DescribeHumanBlackboard, node: "DescribeHumanActionNode") -> Status:
        _ = node
        blackboard.finished = True
        blackboard.should_retry = False
        blackboard.last_status = "Human description completed."
        return Status.SUCCESS

    def _bt_finish_failure(self, blackboard: DescribeHumanBlackboard, node: "DescribeHumanActionNode") -> Status:
        _ = node
        if not blackboard.last_error:
            blackboard.last_error = blackboard.last_reason or "Failed to describe a human."
        blackboard.finished = True
        blackboard.should_retry = False
        blackboard.last_status = "Human description failed."
        return Status.FAILURE

    def _build_result(self, blackboard: DescribeHumanBlackboard, success: bool, elapsed_s: float) -> DescribeHuman.Result:
        result = DescribeHuman.Result()
        result.success = success
        result.human_present = blackboard.human_present
        result.reason = blackboard.last_reason
        result.gender = blackboard.gender
        result.cloth_color = blackboard.cloth_color
        result.pant_color = blackboard.pant_color
        result.wearing_glasses = blackboard.wearing_glasses
        result.wearing_glasses_known = blackboard.wearing_glasses_known
        result.hair_color = blackboard.hair_color
        result.speech_text = blackboard.response_speech_text
        result.data_text = blackboard.response_data_text
        result.message = "ok" if success else blackboard.last_error
        result.camera_used = blackboard.camera_used
        result.model_name = blackboard.model_name
        result.elapsed_seconds = float(elapsed_s)
        return result

    def execute_callback(self, goal_handle) -> DescribeHuman.Result:
        started = time.time()
        self.blackboard = DescribeHumanBlackboard(max_retries=self.max_retries)
        self.blackboard.request = self._build_request()
        self._tree = py_trees.trees.BehaviourTree(root=self._create_tree())

        self._publish_feedback(goal_handle, "start", "Announcing image capture.")
        start_speak_ok, start_speak_message = self._speak_text(self.capture_start_prompt)
        if not start_speak_ok:
            result = self._build_result(self.blackboard, False, time.time() - started)
            result.message = start_speak_message or "Failed to announce capture start."
            goal_handle.abort()
            return result

        while rclpy.ok():
            if goal_handle.is_cancel_requested:
                result = self._build_result(self.blackboard, False, time.time() - started)
                result.message = "Goal canceled."
                goal_handle.canceled()
                return result

            self._publish_feedback(goal_handle, "tick", self.blackboard.last_status or "Running describe-human BT.")
            self._tree.tick()
            status = self._tree.root.status

            if self.blackboard.finished and status == Status.SUCCESS:
                self._publish_feedback(goal_handle, "success_speaking", "Announcing successful registration.")
                success_speak_ok, success_speak_message = self._speak_text(self.register_success_prompt)
                if not success_speak_ok:
                    result = self._build_result(self.blackboard, False, time.time() - started)
                    result.message = success_speak_message or "Failed to announce successful registration."
                    goal_handle.abort()
                    return result
                result = self._build_result(self.blackboard, True, time.time() - started)
                goal_handle.succeed()
                return result

            if self.blackboard.should_retry:
                self.blackboard.should_retry = False
                continue

            if self.blackboard.finished or status == Status.FAILURE:
                result = self._build_result(self.blackboard, False, time.time() - started)
                goal_handle.abort()
                return result

        result = self._build_result(self.blackboard, False, time.time() - started)
        result.message = "ROS shutdown before action completed."
        goal_handle.abort()
        return result


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = DescribeHumanActionNode()
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

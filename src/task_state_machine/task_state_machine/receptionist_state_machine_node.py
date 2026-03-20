#!/usr/bin/env python3
"""YASMIN ROS 2 receptionist state machine."""

from __future__ import annotations

import json
import math
import time
from typing import Any

import rclpy
from coqui_tts_interfaces.action import SpeakText
from rclpy.action import ActionClient
from rclpy.task import Future

import yasmin
from yasmin import Blackboard, State, StateMachine
from yasmin_ros import set_ros_loggers
from yasmin_ros.action_state import ActionState
from yasmin_ros.basic_outcomes import ABORT, SUCCEED, TIMEOUT
from yasmin_ros.service_state import ServiceState
from yasmin_ros.yasmin_node import YasminNode
from yasmin_viewer import YasminViewerPub

from vlm_interfaces.action import AskNameAndDrink, DescribeHuman
from vlm_interfaces.srv import VlmQuery
from yoloe_detection_interfaces.srv import DetectObjectPrompt


HOST = "David"

FINAL_OUTCOME = "task_finished"
RETRY = "retry"
DETECT_GUEST_OUTCOME = "guest_detected"
NAME_DRINK_CAPTURED = "name_drink_captured"
CHARACTERISTICS_CAPTURED = "characteristics_captured"
DELAY_DONE = "delay_done"
INTRO_READY = "intro_ready"
GUEST_SLOT_PREPARED = "guest_slot_prepared"
NEXT_GUEST_READY = "next_guest_ready"
DUPLICATE_GUEST_NAME = "duplicate_guest_name"


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


def wait_for_future(_node, future: Future, timeout_sec: float) -> Any:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if future.done():
            return future.result()
        time.sleep(0.05)
    raise TimeoutError("Timed out waiting for ROS response.")


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
    def __init__(self, delay_key: str, outcome_on_success: str = DELAY_DONE) -> None:
        super().__init__({outcome_on_success})
        self._delay_key = delay_key
        self._outcome_on_success = outcome_on_success

    def execute(self, blackboard: Blackboard) -> str:
        time.sleep(float(blackboard[self._delay_key]))
        return self._outcome_on_success


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
        self._node = YasminNode.get_instance()
        self._service_name = "/vlm/query"
        self._client = self._node.create_client(VlmQuery, self._service_name)

    def execute(self, blackboard: Blackboard) -> str:
        if not self._client.wait_for_service(timeout_sec=5.0):
            blackboard["last_error"] = f"Service '{self._service_name}' is not available."
            return TIMEOUT

        request = VlmQuery.Request()
        request.need_image = False
        request.camera_name = ""
        request.reasoning_mode = "fast"
        request.request_profile = "default"
        request.max_retry_count = 1
        request.json_repair_mode = 0
        request.num_predict_override = 120
        request.timeout_sec_override = 30.0
        request.user_input = ""
        request.prompt = self._prompt_factory(blackboard)

        future = self._client.call_async(request)

        if self._delay_key is not None:
            time.sleep(float(blackboard[self._delay_key]))

        query_wait_deadline = time.monotonic() + float(request.timeout_sec_override)
        while time.monotonic() < query_wait_deadline:
            if future.done():
                response = future.result()
                if response is None:
                    blackboard["last_error"] = "VLM query returned no response."
                    return ABORT
                if not response.success:
                    blackboard["last_error"] = response.message or "VLM query failed."
                    return ABORT
                speech_text = (response.speech_text or "").strip()
                if not speech_text:
                    blackboard["last_error"] = "VLM query returned empty speech_text."
                    return ABORT
                blackboard[self._result_field] = speech_text
                return INTRO_READY
            time.sleep(0.05)

        blackboard["last_error"] = "Timed out waiting for VLM speech generation."
        return TIMEOUT


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
        goal.think = True
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
        return request

    def response_handler(self, blackboard: Blackboard, response: DetectObjectPrompt.Response) -> str:
        max_distance_m = float(blackboard["detect_guest_max_distance_m"])
        blackboard["detect_guest_attempt_count"] = int(blackboard["detect_guest_attempt_count"]) + 1

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


def host_intro_prompt(blackboard: Blackboard) -> str:
    guest = get_current_guest_memory(blackboard)
    return (
        "Generate exactly one short spoken sentence to introduce a guest to the host. "
        "Return JSON through the normal VLM response format. "
        "Set speech_text to the sentence the robot should say aloud. "
        "Set data_text to a JSON object with keys task, reason, complete, and entities. "
        "Set task to 'introduce_guest_to_host'. "
        "Set reason to a short explanation of what was included. "
        "Set complete to true. "
        "Set entities to an object with keys guest_name and favourite_drink. "
        "Only include the guest name and favourite drink in the spoken sentence. "
        f"Host name: {HOST}. "
        f"Guest name: {guest['name']}. "
        f"Favourite drink: {guest['drink']}. "
        "Do not mention appearance or any extra facts."
    )


def guest1_to_guest2_prompt(blackboard: Blackboard) -> str:
    guest1 = ensure_guest_memory(blackboard, "guest1")
    guest2 = ensure_guest_memory(blackboard, "guest2")
    return (
        "Generate exactly one short spoken sentence to introduce guest1 to guest2. "
        "Return JSON through the normal VLM response format. "
        "Set speech_text to the sentence the robot should say aloud. "
        "Set data_text to a JSON object with keys task, reason, complete, and entities. "
        "Set task to 'introduce_guest1_to_guest2'. "
        "Set reason to a short explanation of which guest1 details were used. "
        "Set complete to true. "
        "Set entities to an object with keys guest1_name, guest2_name, favourite_drink, and characteristics_used. "
        "Speak directly to guest2. "
        "Introduce only guest1 to guest2. "
        "You must include guest1 name, favourite drink, and the supplied appearance characteristics. "
        "The spoken sentence must mention the favourite drink and all supplied characteristics when they are short enough to fit naturally in one sentence. "
        "Do not omit gender, shirt color, pant color, hair color, or glasses if they are present in the supplied characteristics list. "
        "Use the supplied characteristics list as the authoritative source for what must be mentioned. "
        f"Guest2 name: {guest2['name']}. "
        f"Guest1 name: {guest1['name']}. "
        f"Guest1 favourite drink: {guest1['drink']}. "
        f"Guest1 characteristics list: {guest1['characteristics_list_text']}. "
        f"Guest1 characteristics summary: {guest1['characteristics_summary']}. "
        f"Guest1 characteristics raw data: {guest1['characteristics']}. "
        "Do not mention the host in this sentence."
    )


def build_state_machine() -> StateMachine:
    sm = StateMachine(outcomes=[FINAL_OUTCOME, ABORT])

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
            DETECT_GUEST_OUTCOME: "GREET_GUEST1",
            RETRY: "DETECT_GUEST1",
            SUCCEED: "GREET_GUEST1",
            ABORT: ABORT,
            TIMEOUT: ABORT,
        },
    )
    sm.add_state(
        "GREET_GUEST1",
        SpeakState(lambda _: f"Hello, Welcome to {HOST} house", "spoken"),
        transitions={"spoken": "ASK_NAME_AND_DRINK_GUEST1", ABORT: ABORT},
    )
    sm.add_state(
        "ASK_NAME_AND_DRINK_GUEST1",
        StoreAskNameAndDrinkState(),
        transitions={
            NAME_DRINK_CAPTURED: "DESCRIBE_GUEST1",
            SUCCEED: "DESCRIBE_GUEST1",
            ABORT: ABORT,
            TIMEOUT: ABORT,
        },
    )
    sm.add_state(
        "DESCRIBE_GUEST1",
        StoreDescribeHumanState(),
        transitions={
            CHARACTERISTICS_CAPTURED: "ASK_FOLLOW_GUEST1",
            SUCCEED: "ASK_FOLLOW_GUEST1",
            ABORT: ABORT,
            TIMEOUT: ABORT,
        },
    )
    sm.add_state(
        "ASK_FOLLOW_GUEST1",
        SpeakState(lambda bb: f"Hi {get_current_guest_memory(bb)['name']}, please follow me", "spoken"),
        transitions={"spoken": "NAVIGATE_AND_PREPARE_INTRO_GUEST1", ABORT: ABORT},
    )
    sm.add_state(
        "NAVIGATE_AND_PREPARE_INTRO_GUEST1",
        VlmSpeechState(host_intro_prompt, "current_host_intro_text", delay_key="navigation_delay_sec"),
        transitions={INTRO_READY: "INTRODUCE_GUEST1_TO_HOST", ABORT: ABORT, TIMEOUT: ABORT},
    )
    sm.add_state(
        "INTRODUCE_GUEST1_TO_HOST",
        SpeakState(lambda bb: str(bb["current_host_intro_text"]), "spoken"),
        transitions={"spoken": "NAVIGATE_BACK_TO_START", ABORT: ABORT},
    )
    sm.add_state(
        "NAVIGATE_BACK_TO_START",
        MockDelayState("return_navigation_delay_sec"),
        transitions={DELAY_DONE: "PREPARE_GUEST2"},
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
            DETECT_GUEST_OUTCOME: "GREET_GUEST2",
            RETRY: "DETECT_GUEST2",
            SUCCEED: "GREET_GUEST2",
            ABORT: ABORT,
            TIMEOUT: ABORT,
        },
    )
    sm.add_state(
        "GREET_GUEST2",
        SpeakState(lambda _: f"Hello, Welcome to {HOST} house", "spoken"),
        transitions={"spoken": "ASK_NAME_AND_DRINK_GUEST2", ABORT: ABORT},
    )
    sm.add_state(
        "ASK_NAME_AND_DRINK_GUEST2",
        StoreAskNameAndDrinkState(),
        transitions={
            NAME_DRINK_CAPTURED: "DESCRIBE_GUEST2",
            DUPLICATE_GUEST_NAME: "DUPLICATE_GUEST2_WARNING",
            SUCCEED: "DESCRIBE_GUEST2",
            ABORT: ABORT,
            TIMEOUT: ABORT,
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
        transitions={"spoken": "DETECT_GUEST2", ABORT: ABORT},
    )
    sm.add_state(
        "DESCRIBE_GUEST2",
        StoreDescribeHumanState(),
        transitions={
            CHARACTERISTICS_CAPTURED: "ASK_FOLLOW_GUEST2",
            SUCCEED: "ASK_FOLLOW_GUEST2",
            ABORT: ABORT,
            TIMEOUT: ABORT,
        },
    )
    sm.add_state(
        "ASK_FOLLOW_GUEST2",
        SpeakState(lambda bb: f"Hi {get_current_guest_memory(bb)['name']}, please follow me", "spoken"),
        transitions={"spoken": "NAVIGATE_AND_PREPARE_INTRO_GUEST2", ABORT: ABORT},
    )
    sm.add_state(
        "NAVIGATE_AND_PREPARE_INTRO_GUEST2",
        VlmSpeechState(host_intro_prompt, "current_host_intro_text", delay_key="navigation_delay_sec"),
        transitions={INTRO_READY: "INTRODUCE_GUEST2_TO_HOST", ABORT: ABORT, TIMEOUT: ABORT},
    )
    sm.add_state(
        "INTRODUCE_GUEST2_TO_HOST",
        SpeakState(lambda bb: str(bb["current_host_intro_text"]), "spoken"),
        transitions={"spoken": "PREPARE_GUEST1_TO_GUEST2_INTRO", ABORT: ABORT},
    )
    sm.add_state(
        "PREPARE_GUEST1_TO_GUEST2_INTRO",
        VlmSpeechState(guest1_to_guest2_prompt, "guest1_to_guest2_intro_text"),
        transitions={INTRO_READY: "INTRODUCE_GUEST1_TO_GUEST2", ABORT: ABORT, TIMEOUT: ABORT},
    )
    sm.add_state(
        "INTRODUCE_GUEST1_TO_GUEST2",
        SpeakState(lambda bb: str(bb["guest1_to_guest2_intro_text"]), "spoken"),
        transitions={"spoken": FINAL_OUTCOME, ABORT: ABORT},
    )

    return sm


def create_blackboard() -> Blackboard:
    blackboard = Blackboard()
    blackboard["guest1"] = create_guest_memory()
    blackboard["guest2"] = create_guest_memory()
    blackboard["current_guest_key"] = "guest1"
    blackboard["current_guest_number"] = 1
    blackboard["detect_guest_attempt_count"] = 0
    blackboard["detect_guest_max_attempts"] = 0
    blackboard["detect_guest_max_distance_m"] = 2.0
    blackboard["startup_delay_sec"] = 3.0
    blackboard["navigation_delay_sec"] = 8.0
    blackboard["return_navigation_delay_sec"] = 5.0
    blackboard["last_detect_message"] = ""
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

    try:
        outcome = state_machine(blackboard)
        yasmin.YASMIN_LOG_INFO(
            "Receptionist state machine finished with outcome=%s guest1=%s guest2=%s"
            % (
                outcome,
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

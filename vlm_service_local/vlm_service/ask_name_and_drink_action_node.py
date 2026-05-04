#!/usr/bin/env python3
"""ROS 2 action server for ask-name-and-drink driven fully inside a behaviour tree."""

from __future__ import annotations

import json
import os
import re
import select
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import pydot
import py_trees
import rclpy
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from std_srvs.srv import Trigger
from std_msgs.msg import String

from coqui_tts_interfaces.action import SpeakText
from vlm_interfaces.action import AskNameAndDrink
from vlm_interfaces.srv import VlmQuery


VLM_QUERY_SERVICE = os.environ.get("VLM_QUERY_SERVICE", "/vlm/query")
GET_COMMAND_SERVICE = os.environ.get("GET_COMMAND_SERVICE", "/get_command")
SPEAK_ACTION_NAME = os.environ.get("SPEAK_ACTION_NAME", "/coqui_tts/speak")
DEFAULT_TEXT_THINK = os.environ.get("DEFAULT_TEXT_THINK", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
SERVICE_TIMEOUT_SEC = float(os.environ.get("VLM_QUERY_TIMEOUT_SEC", "180.0"))

SYSTEM_PROMPT = (
    "You are a home service robot assistant.\n"
    "Use the known state provided in the prompt as conversation memory across turns.\n"
    "A current_collection_round value will be provided in the prompt.\n"
    "Analyze only the current guest utterance, update the known state mentally, and decide what the robot should say next.\n"
    "If current_collection_round is name_and_drink, extract only the guest's name and favourite drink from the current utterance.\n"
    "If current_collection_round is food_preferences_and_allergy, extract only the guest's food flavour preference and food allergy from the current utterance.\n"
    "When current_collection_round is name_and_drink, set food_flavour_preference and food_allergy to null.\n"
    "When current_collection_round is food_preferences_and_allergy, set name and drink to null.\n"
    "If the guest explicitly says they have no food allergy, set food_allergy to 'no food allergy' instead of null.\n"
    "If the guest explicitly says they have no particular food flavour preference, set food_flavour_preference to 'no particular preference' instead of null.\n"
    "If the mentioned item is food/object or not a beverage, set drink to null.\n"
    "Return JSON data_text with keys: complete, continue_conversation, task, reason, entities{name,drink,food_flavour_preference,food_allergy}.\n"
    "Use null for missing values in data_text.\n"
    "complete is true only when the final name, drink, food_flavour_preference, and food_allergy are all known after considering the known state and current utterance.\n"
    "continue_conversation is true only when one or more required fields are still missing.\n"
    "task should briefly describe which fields were extracted from the current utterance. Use All when all four are supplied, otherwise join field labels with '+', for example Name+Drink or FoodFlavourPreference+FoodAllergy. Use unknown if nothing useful was extracted.\n"
    "speech_text must be exactly what the robot should say next.\n"
    "Do not ask any open-ended follow-up such as asking whether the guest wants to add anything else.\n"
    "Do not extend the conversation beyond the required fields for the active round.\n"
    "The conversation must happen in two separate rounds.\n"
    "In round 1, ask only for the guest's name and favourite drink.\n"
    "Only after both the name and favourite drink are known, start round 2 and ask for the food flavour preference and any food allergy.\n"
    "If some fields are already known within the active round, ask only for the missing fields from that round.\n"
    "If all four fields are known after this turn, confirm all four naturally and stop.\n"
    "Examples:\n"
    "- if current_collection_round='name_and_drink', known_name=null, known_drink=null, known_food_flavour_preference=null, known_food_allergy=null, and input is 'my name is Jason and I like tea', data_text should contain complete=false, continue_conversation=true, name='Jason', drink='tea', food_flavour_preference=null, food_allergy=null, task='Name+Drink', and speech_text should begin round 2 by asking for food flavour preference and food allergy\n"
    "- if current_collection_round='name_and_drink', known_name='Jason', known_drink=null, known_food_flavour_preference=null, known_food_allergy=null, and input is 'I like spicy food and tea', data_text should contain drink='tea' and must set food_flavour_preference=null because round 1 only extracts name and drink\n"
    "- if current_collection_round='food_preferences_and_allergy', known_name='Jason', known_drink='tea', known_food_flavour_preference=null, known_food_allergy=null, and input is 'I like spicy food and I have no food allergy', data_text should contain complete=true, continue_conversation=false, name=null, drink=null, food_flavour_preference='spicy food', food_allergy='no food allergy', task='FoodFlavourPreference+FoodAllergy'"
)
GENERIC_NAME_VALUES = {"guest", "user", "person", "someone", "visitor"}
NO_FOOD_ALLERGY_TEXT = "no food allergy"
NO_FOOD_PREFERENCE_TEXT = "no particular preference"
Status = py_trees.common.Status


class TreeCondition(py_trees.behaviour.Behaviour):
    def __init__(
        self,
        name: str,
        blackboard: "AskNameAndDrinkBlackboard",
        fn: Callable[["AskNameAndDrinkBlackboard"], bool],
    ) -> None:
        super().__init__(name=name)
        self._blackboard = blackboard
        self._fn = fn

    def update(self) -> Status:
        return Status.SUCCESS if self._fn(self._blackboard) else Status.FAILURE


class TreeAction(py_trees.behaviour.Behaviour):
    def __init__(
        self,
        name: str,
        blackboard: "AskNameAndDrinkBlackboard",
        node: "AskNameAndDrinkActionNode",
        fn: Callable[["AskNameAndDrinkBlackboard", "AskNameAndDrinkActionNode"], Status],
    ) -> None:
        super().__init__(name=name)
        self._blackboard = blackboard
        self._node = node
        self._fn = fn

    def update(self) -> Status:
        return self._fn(self._blackboard, self._node)


@dataclass
class AskNameAndDrinkBlackboard:
    guest_name: str | None = None
    guest_drink: str | None = None
    guest_food_flavour_preference: str | None = None
    guest_food_allergy: str | None = None
    last_result: dict[str, Any] = field(default_factory=dict)
    last_speech_text: str = ""
    last_error: str = ""
    robot_text: str = ""
    last_user_input: str = ""
    completed: bool = False
    elapsed_s: float = 0.0
    attempt_count: int = 0
    task: str = "unknown"
    reason: str = ""
    think: bool = DEFAULT_TEXT_THINK
    max_attempts: int = 3
    spoke_response: bool = False
    prompt_spoken_for_session: bool = False
    llm_complete: bool = False
    llm_continue_conversation: bool = True
    stop_after_turn: bool = False
    error_stage: str = ""

    def clear_turn(self) -> None:
        self.last_result = {}
        self.last_speech_text = ""
        self.last_error = ""
        self.robot_text = ""
        self.last_user_input = ""
        self.completed = False
        self.elapsed_s = 0.0
        self.task = "unknown"
        self.reason = ""
        self.spoke_response = False
        self.llm_complete = False
        self.llm_continue_conversation = True
        self.stop_after_turn = False
        self.error_stage = ""

    def reset_session(self) -> None:
        self.guest_name = None
        self.guest_drink = None
        self.guest_food_flavour_preference = None
        self.guest_food_allergy = None
        self.attempt_count = 0
        self.prompt_spoken_for_session = False
        self.clear_turn()

    def reset_context(self) -> None:
        self.guest_name = None
        self.guest_drink = None
        self.guest_food_flavour_preference = None
        self.guest_food_allergy = None
        self.prompt_spoken_for_session = False
        self.last_result = {}
        self.last_speech_text = ""
        self.task = "unknown"
        self.reason = ""
        self.llm_complete = False
        self.llm_continue_conversation = True


def parse_json_relaxed(text: str) -> dict[str, Any]:
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


def normalize(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"null", "none", "unknown"}:
        return None
    return text


def is_generic_name(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in GENERIC_NAME_VALUES


def normalize_food_flavour_preference(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None

    lowered = text.lower()
    if lowered in {"null", "unknown"}:
        return None
    if lowered in {
        "none",
        "no preference",
        "no particular preference",
        "anything",
        "any",
        "no food preference",
        "no flavour preference",
        "no flavor preference",
    }:
        return NO_FOOD_PREFERENCE_TEXT
    return text


def normalize_food_allergy(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None

    lowered = text.lower()
    if lowered in {"null", "unknown"}:
        return None
    if lowered in {
        "none",
        "no",
        "nil",
        "n/a",
        "no allergy",
        "no allergies",
        "no food allergy",
        "no food allergies",
        "nothing",
    }:
        return NO_FOOD_ALLERGY_TEXT
    return text


def has_no_food_allergy(value: str | None) -> bool:
    return bool(value) and value.strip().lower() == NO_FOOD_ALLERGY_TEXT


def has_no_food_preference(value: str | None) -> bool:
    return bool(value) and value.strip().lower() == NO_FOOD_PREFERENCE_TEXT


def has_name_and_drink(name: str | None, drink: str | None) -> bool:
    return bool(name and drink)


def has_food_details(food_flavour_preference: str | None, food_allergy: str | None) -> bool:
    return bool(food_flavour_preference and food_allergy)


def current_collection_round(name: str | None, drink: str | None) -> str:
    if has_name_and_drink(name, drink):
        return "food_preferences_and_allergy"
    return "name_and_drink"


def infer_task(
    name: str | None,
    drink: str | None,
    food_flavour_preference: str | None,
    food_allergy: str | None,
) -> str:
    present: list[str] = []
    if name:
        present.append("Name")
    if drink:
        present.append("Drink")
    if food_flavour_preference:
        present.append("FoodFlavourPreference")
    if food_allergy:
        present.append("FoodAllergy")
    if len(present) == 4:
        return "All"
    return "+".join(present) if present else "unknown"


def build_name_and_drink_prompt(name: str | None, drink: str | None) -> str:
    if not name and not drink:
        return "Please tell me your name and your favourite drink."
    if not name:
        return "Please tell me your name."
    if not drink:
        return f"Hello {name}, please tell me your favourite drink."
    return ""


def build_food_details_prompt(
    name: str | None,
    food_flavour_preference: str | None,
    food_allergy: str | None,
) -> str:
    guest_prefix = f"Thank you, {name}. " if name else ""
    if not food_flavour_preference and not food_allergy:
        return guest_prefix + "Now please tell me your food flavour preference and any food allergy."
    if not food_flavour_preference:
        return guest_prefix + "Please tell me your food flavour preference."
    if not food_allergy:
        return guest_prefix + "Please tell me whether you have any food allergy."
    return ""


def initial_prompt() -> str:
    return build_name_and_drink_prompt(None, None)


def confirmation_sentence(
    name: str | None,
    drink: str | None,
    food_flavour_preference: str | None,
    food_allergy: str | None,
) -> str:
    if not has_name_and_drink(name, drink):
        return build_name_and_drink_prompt(name, drink)
    if not has_food_details(food_flavour_preference, food_allergy):
        return build_food_details_prompt(name, food_flavour_preference, food_allergy)

    preference_text = (
        "that you have no particular food flavour preference"
        if has_no_food_preference(food_flavour_preference)
        else f"your food flavour preference as {food_flavour_preference}"
    )
    allergy_text = (
        "that you have no food allergy"
        if has_no_food_allergy(food_allergy)
        else f"your food allergy as {food_allergy}"
    )
    return (
        f"Thank you, {name}! I have noted your favourite drink as {drink}, "
        f"{preference_text}, and {allergy_text}."
    )


def fallback_followup(
    name: str | None,
    drink: str | None,
    food_flavour_preference: str | None,
    food_allergy: str | None,
) -> str:
    if not has_name_and_drink(name, drink):
        return build_name_and_drink_prompt(name, drink)
    return build_food_details_prompt(name, food_flavour_preference, food_allergy)


class AskNameAndDrinkActionNode(Node):
    def __init__(self) -> None:
        super().__init__("ask_name_and_drink_action_node")

        self.declare_parameter("action_name", "/ask_name_and_drink")
        self.declare_parameter("vlm_query_service", VLM_QUERY_SERVICE)
        self.declare_parameter("get_command_service", GET_COMMAND_SERVICE)
        self.declare_parameter("speak_action_name", SPEAK_ACTION_NAME)
        self.declare_parameter("service_timeout_sec", SERVICE_TIMEOUT_SEC)
        self.declare_parameter("default_max_attempts", 3)
        self.declare_parameter("enable_speaking", True)
        self.declare_parameter("debug_text_input_mode", False)
        self.declare_parameter("debug_text_input_prompt", "guest")
        self.declare_parameter("input_retry_count", 2)
        self.declare_parameter("input_retry_prompt", "I could not hear you clearly. Please say it again.")
        self.declare_parameter(
            "restart_prompt",
            (
                "I could not reliably extract your guest details. Let us start again. "
                "Please tell me your name and your favourite drink."
            ),
        )
        self.declare_parameter("bt_monitor_topic", "/bt_tree/snapshots")
        self.declare_parameter("bt_monitor_heartbeat_hz", 1.0)
        self.declare_parameter("trace_logging", True)
        self.declare_parameter("trace_log_prompt", True)

        self.action_name = str(self.get_parameter("action_name").value).strip() or "/ask_name_and_drink"
        self.vlm_query_service = str(self.get_parameter("vlm_query_service").value).strip() or VLM_QUERY_SERVICE
        self.get_command_service = str(self.get_parameter("get_command_service").value).strip() or GET_COMMAND_SERVICE
        self.speak_action_name = str(self.get_parameter("speak_action_name").value).strip() or SPEAK_ACTION_NAME
        self.service_timeout_sec = max(1.0, float(self.get_parameter("service_timeout_sec").value))
        self.default_max_attempts = max(1, int(self.get_parameter("default_max_attempts").value))
        self.enable_speaking = bool(self.get_parameter("enable_speaking").value)
        self.debug_text_input_mode = bool(self.get_parameter("debug_text_input_mode").value)
        self.debug_text_input_prompt = (
            str(self.get_parameter("debug_text_input_prompt").value).strip() or "guest"
        )
        self.input_retry_count = max(1, int(self.get_parameter("input_retry_count").value))
        self.input_retry_prompt = (
            str(self.get_parameter("input_retry_prompt").value).strip()
            or "I could not hear you clearly. Please say it again."
        )
        self.restart_prompt = (
            str(self.get_parameter("restart_prompt").value).strip()
            or (
                "I could not reliably extract your guest details. Let us start again. "
                "Please tell me your name and your favourite drink."
            )
        )
        self.bt_monitor_topic = (
            str(self.get_parameter("bt_monitor_topic").value).strip() or "/bt_tree/snapshots"
        )
        self.bt_monitor_heartbeat_hz = max(0.2, float(self.get_parameter("bt_monitor_heartbeat_hz").value))
        self.trace_logging = bool(self.get_parameter("trace_logging").value)
        self.trace_log_prompt = bool(self.get_parameter("trace_log_prompt").value)
        self._debug_tty_path = "/dev/tty"
        self._bt_tick_count = 0
        self._active_goal = False
        self._visual_phase = "startup"
        self._visual_note = ""

        self._callback_group = ReentrantCallbackGroup()
        self._vlm_client = self.create_client(
            VlmQuery,
            self.vlm_query_service,
            callback_group=self._callback_group,
        )
        self._get_command_client = self.create_client(
            Trigger,
            self.get_command_service,
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
            AskNameAndDrink,
            self.action_name,
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self._callback_group,
        )
        self._bt_monitor_publisher = self.create_publisher(String, self.bt_monitor_topic, 10)
        self.create_timer(1.0 / self.bt_monitor_heartbeat_hz, self._bt_monitor_heartbeat)
        self.blackboard = AskNameAndDrinkBlackboard()
        self._turn_tree = self._create_turn_tree()
        self._tree = py_trees.trees.BehaviourTree(root=self._turn_tree)

        self.get_logger().info(
            f"Ask-name-and-drink action ready on {self.action_name} | "
            f"vlm_service={self.vlm_query_service} | get_command={self.get_command_service} "
            f"speak_action={self.speak_action_name} | debug_text_input_mode={self.debug_text_input_mode}"
        )
        self._trace(
            "py_trees structure\n"
            f"{py_trees.display.unicode_tree(self._turn_tree, show_status=False)}"
        )
        self._publish_bt_snapshot(event="startup")

    def _create_turn_tree(self) -> py_trees.behaviour.Behaviour:
        return py_trees.composites.Sequence(
            name="AskNameAndDrinkTurn",
            memory=False,
            children=[
                py_trees.composites.Selector(
                    name="InitialPromptGate",
                    memory=False,
                    children=[
                        py_trees.composites.Sequence(
                            name="SpeakInitialPrompt",
                            memory=False,
                            children=[
                                TreeCondition(
                                    "PromptNotSpoken",
                                    self.blackboard,
                                    lambda bb: not bb.prompt_spoken_for_session,
                                ),
                                TreeAction(
                                    "PrepareInitialPrompt",
                                    self.blackboard,
                                    self,
                                    self._bt_prepare_initial_prompt,
                                ),
                                TreeAction(
                                    "SpeakInitialPrompt",
                                    self.blackboard,
                                    self,
                                    self._bt_speak_robot_text,
                                ),
                                TreeAction(
                                    "MarkPromptSpoken",
                                    self.blackboard,
                                    self,
                                    self._bt_mark_prompt_spoken,
                                ),
                            ],
                        ),
                        TreeAction("NoopSuccess", self.blackboard, self, self._bt_noop_success),
                    ],
                ),
                TreeAction("GetInput", self.blackboard, self, self._bt_get_input),
                TreeAction("ClassifyInput", self.blackboard, self, self._bt_classify_input),
                TreeAction("MergeEntities", self.blackboard, self, self._bt_merge_entities),
                py_trees.composites.Selector(
                    name="CompletionDecision",
                    memory=False,
                    children=[
                        py_trees.composites.Sequence(
                            name="HaveBothEntities",
                            memory=False,
                            children=[
                                TreeCondition(
                                    "AllGuestDetailsKnown",
                                    self.blackboard,
                                    lambda bb: bool(
                                        bb.guest_name
                                        and bb.guest_drink
                                        and bb.guest_food_flavour_preference
                                        and bb.guest_food_allergy
                                    ),
                                ),
                                TreeAction(
                                    "HandleComplete",
                                    self.blackboard,
                                    self,
                                    self._bt_handle_complete,
                                ),
                            ],
                        ),
                        TreeAction(
                            "HandleIncomplete",
                            self.blackboard,
                            self,
                            self._bt_handle_incomplete,
                        ),
                    ],
                ),
                TreeAction("SpeakRobotText", self.blackboard, self, self._bt_speak_robot_text),
            ],
        )

    def _trace(self, message: str) -> None:
        if self.trace_logging:
            self.get_logger().info(f"[trace] {message}")

    def _bt_monitor_heartbeat(self) -> None:
        if self._active_goal:
            self._publish_bt_snapshot(event="heartbeat")

    def _publish_bt_snapshot(self, event: str) -> None:
        tree_text = py_trees.display.unicode_tree(self._turn_tree, show_status=True)
        dot_graph = py_trees.display.dot_tree(self._turn_tree)
        status_styles = {
            py_trees.common.Status.SUCCESS: {"fillcolor": "#c8f7c5", "color": "#2e7d32", "penwidth": "2"},
            py_trees.common.Status.FAILURE: {"fillcolor": "#ffcdd2", "color": "#c62828", "penwidth": "2"},
            py_trees.common.Status.RUNNING: {"fillcolor": "#bbdefb", "color": "#1565c0", "penwidth": "3"},
            py_trees.common.Status.INVALID: {"fillcolor": "#eceff1", "color": "#607d8b", "penwidth": "1"},
        }
        for behaviour in self._turn_tree.iterate():
            style = status_styles.get(behaviour.status, status_styles[py_trees.common.Status.INVALID])
            node_name = behaviour.name
            matches = dot_graph.get_node(node_name)
            if not matches:
                continue
            for dot_node in matches:
                dot_node.set("style", "filled")
                dot_node.set("fillcolor", style["fillcolor"])
                dot_node.set("color", style["color"])
                dot_node.set("penwidth", style["penwidth"])
                dot_node.set("label", f"{node_name}\\n[{behaviour.status.name}]")
        flow_cluster = pydot.Cluster(
            graph_name="cluster_action_flow",
            label="Action Flow",
            color="#9aa4b2",
            style="rounded",
        )
        flow_nodes = {
            "goal_start": ("Goal Start", "#e8f0fe"),
            "turn_start": (f"Turn Start\\n{self.blackboard.attempt_count}/{self.blackboard.max_attempts}", "#e8f0fe"),
            "get_input": ("Get Input", "#eef3f8"),
            "input_retry": ("Retry Input", "#fff4e5"),
            "classify": ("Classify Input", "#eef3f8"),
            "classify_recover": ("Reset And Restart", "#fff4e5"),
            "decision": ("LLM Decision", "#eef3f8"),
            "speak": ("Speak Response", "#eef3f8"),
            "retry_turn": ("Retry Turn", "#fff4e5"),
            "goal_end": ("Goal End", "#e8f5e9"),
        }
        active_flow_style = {"fillcolor": "#90caf9", "color": "#1565c0", "penwidth": "3"}
        for node_id, (label, fillcolor) in flow_nodes.items():
            flow_node = pydot.Node(
                f"flow_{node_id}",
                label=label,
                shape="box",
                style='"rounded,filled"',
                fillcolor=fillcolor,
                color="#607d8b",
                penwidth="1.5",
                fontsize="10",
            )
            if self._visual_phase == node_id:
                flow_node.set("fillcolor", active_flow_style["fillcolor"])
                flow_node.set("color", active_flow_style["color"])
                flow_node.set("penwidth", active_flow_style["penwidth"])
            flow_cluster.add_node(flow_node)
        flow_edges = [
            ("goal_start", "turn_start", ""),
            ("turn_start", "get_input", ""),
            ("get_input", "classify", "ok"),
            ("get_input", "input_retry", "fail"),
            ("input_retry", "retry_turn", "retry"),
            ("classify", "decision", "ok"),
            ("classify", "classify_recover", "fail"),
            ("classify_recover", "retry_turn", "reset"),
            ("decision", "speak", "continue/complete"),
            ("speak", "retry_turn", "if incomplete"),
            ("speak", "goal_end", "if done"),
            ("retry_turn", "turn_start", "next attempt"),
        ]
        for src, dst, label in flow_edges:
            flow_cluster.add_edge(
                pydot.Edge(
                    f"flow_{src}",
                    f"flow_{dst}",
                    label=label,
                    fontsize="9",
                    color="#78909c",
                )
            )
        dot_graph.add_subgraph(flow_cluster)
        dot_graph.add_edge(pydot.Edge("AskNameAndDrinkTurn", "flow_goal_start", style="invis"))
        dot_text = dot_graph.to_string()
        payload = {
            "tree_id": self.action_name,
            "node_name": self.get_name(),
            "event": event,
            "tick_count": self._bt_tick_count,
            "timestamp_sec": time.time(),
            "tree_text": tree_text,
            "dot_text": dot_text,
            "summary": {
                "completed": self.blackboard.completed,
                "attempt_count": int(self.blackboard.attempt_count),
                "task": self.blackboard.task,
                "reason": self.blackboard.reason,
                "name": self.blackboard.guest_name or "",
                "drink": self.blackboard.guest_drink or "",
                "food_flavour_preference": self.blackboard.guest_food_flavour_preference or "",
                "food_allergy": self.blackboard.guest_food_allergy or "",
                "robot_text": self.blackboard.robot_text,
                "last_user_input": self.blackboard.last_user_input,
                "last_error": self.blackboard.last_error,
                "llm_complete": self.blackboard.llm_complete,
                "llm_continue_conversation": self.blackboard.llm_continue_conversation,
                "debug_text_input_mode": self.debug_text_input_mode,
                "visual_phase": self._visual_phase,
                "visual_note": self._visual_note,
            },
        }
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self._bt_monitor_publisher.publish(msg)

    def goal_callback(self, goal_request: AskNameAndDrink.Goal) -> GoalResponse:
        if int(goal_request.max_attempts) < 0:
            self.get_logger().warn("Rejecting goal with negative max_attempts.")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle) -> CancelResponse:
        return CancelResponse.ACCEPT

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

    def _build_contextual_prompt(self, blackboard: AskNameAndDrinkBlackboard) -> str:
        known_name = blackboard.guest_name or "null"
        known_drink = blackboard.guest_drink or "null"
        known_food_flavour_preference = blackboard.guest_food_flavour_preference or "null"
        known_food_allergy = blackboard.guest_food_allergy or "null"
        collection_round = current_collection_round(blackboard.guest_name, blackboard.guest_drink)
        return (
            f"{SYSTEM_PROMPT}\n"
            "Known state before the current utterance:\n"
            f"- current_collection_round: {collection_round}\n"
            f"- known_name: {known_name}\n"
            f"- known_drink: {known_drink}\n"
            f"- known_food_flavour_preference: {known_food_flavour_preference}\n"
            f"- known_food_allergy: {known_food_allergy}\n"
        )

    def _debug_prompt_summary(self, request: VlmQuery.Request, blackboard: AskNameAndDrinkBlackboard) -> str:
        return (
            "vlm classify request | "
            f"user_input={str(request.user_input)[:160]!r} "
            f"reasoning_mode={request.reasoning_mode} "
            f"collection_round={current_collection_round(blackboard.guest_name, blackboard.guest_drink)!r} "
            f"known_name={blackboard.guest_name!r} "
            f"known_drink={blackboard.guest_drink!r} "
            f"known_food_flavour_preference={blackboard.guest_food_flavour_preference!r} "
            f"known_food_allergy={blackboard.guest_food_allergy!r}"
        )

    def _classify_input(
        self,
        user_text: str,
        think: bool,
        blackboard: AskNameAndDrinkBlackboard,
    ) -> tuple[dict[str, Any], str, float]:
        if not self._vlm_client.wait_for_service(timeout_sec=1.0):
            raise RuntimeError(f"VLM query service '{self.vlm_query_service}' not ready.")

        request = VlmQuery.Request()
        request.need_image = False
        request.camera_name = ""
        request.prompt = self._build_contextual_prompt(blackboard)
        request.reasoning_mode = "thinking" if think else "fast"
        request.user_input = user_text
        request.request_profile = "dialogue_text"
        request.max_retry_count = 0
        request.json_repair_mode = 1
        request.num_predict_override = 0
        request.timeout_sec_override = 0.0
        if self.trace_log_prompt and self.debug_text_input_mode:
            self._trace(self._debug_prompt_summary(request, blackboard))

        started = time.time()
        future = self._vlm_client.call_async(request)
        ok, response, error_text = self._wait_for_future(future, self.service_timeout_sec)
        elapsed_s = time.time() - started

        if not ok:
            raise RuntimeError(f"VLM request failed: {error_text}")
        if response is None:
            raise RuntimeError("VLM service returned no response")
        if not response.success:
            raise RuntimeError(response.message or "VLM service request failed")

        speech_text = str(response.speech_text).strip()
        data_text = str(response.data_text).strip()
        self._trace(
            "vlm classify response | "
            f"reasoning_mode={request.reasoning_mode} "
            f"elapsed_s={elapsed_s:.3f} "
            f"speech_text={speech_text[:200]!r} "
            f"data_text={data_text[:400]!r}"
        )

        if data_text:
            parsed = parse_json_relaxed(data_text)
            self._trace(
                "vlm parsed data_text | "
                f"parsed={json.dumps(parsed, ensure_ascii=False)[:500]}"
            )
            return parsed, speech_text, elapsed_s
        if speech_text:
            parsed = parse_json_relaxed(speech_text)
            self._trace(
                "vlm parsed speech_text fallback | "
                f"parsed={json.dumps(parsed, ensure_ascii=False)[:500]}"
            )
            return parsed, "", elapsed_s
        raise RuntimeError(
            "Service returned no usable JSON payload. "
            f"speech_text={response.speech_text!r} data_text={response.data_text!r}"
        )

    def _get_command(self) -> str:
        if self.debug_text_input_mode:
            return self._get_debug_text_input()

        if not self._get_command_client.wait_for_service(timeout_sec=1.0):
            raise RuntimeError(f"Get-command service '{self.get_command_service}' not ready.")

        request = Trigger.Request()
        future = self._get_command_client.call_async(request)
        ok, response, error_text = self._wait_for_future(future, self.service_timeout_sec)
        if not ok:
            raise RuntimeError(f"Get-command request failed: {error_text}")
        if response is None:
            raise RuntimeError("Get-command service returned no response")
        if not response.success:
            raise RuntimeError(response.message or "Get-command failed.")

        text = str(response.message).strip()
        if not text:
            raise RuntimeError("Empty speech transcription.")
        return text

    def _get_command_with_retry(self) -> str:
        attempts = self.input_retry_count if not self.debug_text_input_mode else 1
        last_error = "unknown input failure"
        for attempt_index in range(1, attempts + 1):
            try:
                return self._get_command()
            except Exception as exc:
                last_error = str(exc)
                self._trace(
                    "input failure | "
                    f"attempt={attempt_index}/{attempts} "
                    f"error={last_error[:240]!r}"
                )
                if attempt_index >= attempts:
                    break
                if not self.debug_text_input_mode:
                    speak_ok, speak_message = self._speak_text(self.input_retry_prompt)
                    self._trace(
                        "input retry prompt | "
                        f"success={speak_ok} "
                        f"message={speak_message[:240]!r}"
                    )
        raise RuntimeError(last_error)

    def _get_debug_text_input(self) -> str:
        prompt = f"{self.debug_text_input_prompt}> "
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
                raise RuntimeError("No interactive terminal available for debug text input.")

            prompt_written = False
            deadline = time.time() + self.service_timeout_sec
            while time.time() < deadline:
                if write_stream is not None and not prompt_written:
                    try:
                        write_stream.write(prompt)
                        write_stream.flush()
                    except Exception:
                        pass
                    prompt_written = True

                ready, _, _ = select.select([read_stream], [], [], 0.2)
                if not ready:
                    continue

                line = read_stream.readline()
                if line == "":
                    raise RuntimeError("Debug text input closed.")
                text = line.strip()
                if not text:
                    prompt_written = False
                    continue
                return text

            raise RuntimeError("Timed out waiting for debug text input.")
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
        if not self.enable_speaking:
            return True, "Speaking disabled."

        if not self._speak_action_client.wait_for_server(timeout_sec=0.8):
            return False, f"Speak action server '{self.speak_action_name}' not ready."

        goal = SpeakText.Goal()
        goal.text = cleaned
        speak_started = time.time()
        self._trace(f"speak start | text={cleaned[:240]!r}")
        send_goal_future = self._speak_action_client.send_goal_async(goal)
        ok, goal_handle, error_text = self._wait_for_future(send_goal_future, self.service_timeout_sec)
        if not ok:
            return False, f"Failed to send SpeakText goal: {error_text}"
        if goal_handle is None or not goal_handle.accepted:
            return False, "SpeakText goal rejected."

        result_future = goal_handle.get_result_async()
        ok, result_wrap, error_text = self._wait_for_future(result_future, self.service_timeout_sec)
        if not ok:
            return False, f"SpeakText result wait failed: {error_text}"
        if result_wrap is None:
            return False, "SpeakText returned no result."

        result = result_wrap.result
        self._trace(
            "speak complete | "
            f"success={result.success} "
            f"elapsed_s={time.time() - speak_started:.3f} "
            f"message={result.message[:240]!r}"
        )
        if result.success:
            return True, result.message
        return False, result.message

    def _publish_feedback(self, goal_handle, stage: str) -> None:
        feedback = AskNameAndDrink.Feedback()
        feedback.stage = stage
        feedback.robot_text = self.blackboard.robot_text
        feedback.name = self.blackboard.guest_name or ""
        feedback.drink = self.blackboard.guest_drink or ""
        feedback.food_flavour_preference = self.blackboard.guest_food_flavour_preference or ""
        feedback.food_allergy = self.blackboard.guest_food_allergy or ""
        feedback.user_input = self.blackboard.last_user_input
        feedback.attempt_count = int(self.blackboard.attempt_count)
        goal_handle.publish_feedback(feedback)

    @staticmethod
    def _bt_noop_success(blackboard: AskNameAndDrinkBlackboard, node: "AskNameAndDrinkActionNode") -> Status:
        return Status.SUCCESS

    @staticmethod
    def _bt_prepare_initial_prompt(blackboard: AskNameAndDrinkBlackboard, node: "AskNameAndDrinkActionNode") -> Status:
        node._visual_phase = "speak"
        node._visual_note = "Preparing initial prompt"
        blackboard.robot_text = initial_prompt()
        return Status.SUCCESS

    @staticmethod
    def _bt_mark_prompt_spoken(blackboard: AskNameAndDrinkBlackboard, node: "AskNameAndDrinkActionNode") -> Status:
        blackboard.prompt_spoken_for_session = True
        return Status.SUCCESS

    def _bt_speak_robot_text(self, blackboard: AskNameAndDrinkBlackboard, node: "AskNameAndDrinkActionNode") -> Status:
        node._visual_phase = "speak"
        node._visual_note = "Speaking robot response"
        if not blackboard.robot_text:
            return Status.SUCCESS
        if self.debug_text_input_mode:
            self.get_logger().info(f"Robot debug reply: {blackboard.robot_text}")
        speak_ok, speak_message = self._speak_text(blackboard.robot_text)
        blackboard.spoke_response = speak_ok
        if not speak_ok:
            blackboard.last_error = speak_message
            return Status.FAILURE
        return Status.SUCCESS

    def _bt_get_input(self, blackboard: AskNameAndDrinkBlackboard, node: "AskNameAndDrinkActionNode") -> Status:
        node._visual_phase = "get_input"
        node._visual_note = "Waiting for user input"
        try:
            blackboard.last_user_input = self._get_command_with_retry()
            self._trace(f"input received | user_input={blackboard.last_user_input[:300]!r}")
            return Status.SUCCESS
        except Exception as exc:
            blackboard.last_error = str(exc)
            blackboard.error_stage = "input"
            node._visual_phase = "input_retry"
            node._visual_note = str(exc)
            return Status.FAILURE

    def _bt_classify_input(self, blackboard: AskNameAndDrinkBlackboard, node: "AskNameAndDrinkActionNode") -> Status:
        node._visual_phase = "classify"
        node._visual_note = "Sending user input to VLM"
        try:
            result, speech_text, elapsed_s = self._classify_input(
                blackboard.last_user_input,
                blackboard.think,
                blackboard,
            )
        except Exception as exc:
            blackboard.last_error = str(exc)
            blackboard.error_stage = "classify"
            node._visual_phase = "classify_recover"
            node._visual_note = str(exc)
            node._trace(f"bt classify failed | error={str(exc)[:400]!r}")
            return Status.FAILURE

        blackboard.last_result = result
        blackboard.last_speech_text = speech_text
        blackboard.elapsed_s = elapsed_s
        self._trace(
            "bt classify success | "
            f"elapsed_s={elapsed_s:.3f} "
            f"speech_text={speech_text[:200]!r}"
        )
        return Status.SUCCESS

    @staticmethod
    def _bt_merge_entities(blackboard: AskNameAndDrinkBlackboard, node: "AskNameAndDrinkActionNode") -> Status:
        node._visual_phase = "decision"
        node._visual_note = "Merging extracted entities"
        entities = blackboard.last_result.get("entities", {})
        active_round = current_collection_round(blackboard.guest_name, blackboard.guest_drink)
        parsed_name = normalize(entities.get("name"))
        parsed_drink = normalize(entities.get("drink"))
        parsed_food_flavour_preference = normalize_food_flavour_preference(
            entities.get("food_flavour_preference")
            or entities.get("food_flavor_preference")
            or entities.get("flavour_preference")
            or entities.get("flavor_preference")
        )
        parsed_food_allergy = normalize_food_allergy(
            entities.get("food_allergy")
            or entities.get("food_allergies")
            or entities.get("allergy")
            or entities.get("allergies")
        )
        blackboard.reason = normalize(blackboard.last_result.get("reason")) or ""

        if active_round == "name_and_drink":
            if parsed_name and not (blackboard.guest_name and is_generic_name(parsed_name)):
                blackboard.guest_name = parsed_name
            if parsed_drink:
                blackboard.guest_drink = parsed_drink
        else:
            if parsed_food_flavour_preference:
                blackboard.guest_food_flavour_preference = parsed_food_flavour_preference
            if parsed_food_allergy:
                blackboard.guest_food_allergy = parsed_food_allergy

        blackboard.task = infer_task(
            blackboard.guest_name,
            blackboard.guest_drink,
            blackboard.guest_food_flavour_preference,
            blackboard.guest_food_allergy,
        )
        blackboard.llm_complete = bool(
            blackboard.guest_name
            and blackboard.guest_drink
            and blackboard.guest_food_flavour_preference
            and blackboard.guest_food_allergy
        )
        blackboard.llm_continue_conversation = not blackboard.llm_complete
        node._trace(
            "bt merge entities | "
            f"active_round={active_round!r} "
            f"task={blackboard.task!r} "
            f"complete={blackboard.llm_complete} "
            f"continue_conversation={blackboard.llm_continue_conversation} "
            f"name={blackboard.guest_name!r} "
            f"drink={blackboard.guest_drink!r} "
            f"food_flavour_preference={blackboard.guest_food_flavour_preference!r} "
            f"food_allergy={blackboard.guest_food_allergy!r} "
            f"reason={blackboard.reason[:300]!r}"
        )
        return Status.SUCCESS

    @staticmethod
    def _bt_handle_complete(blackboard: AskNameAndDrinkBlackboard, node: "AskNameAndDrinkActionNode") -> Status:
        node._visual_phase = "goal_end"
        node._visual_note = "Task completed"
        blackboard.robot_text = confirmation_sentence(
            blackboard.guest_name,
            blackboard.guest_drink,
            blackboard.guest_food_flavour_preference,
            blackboard.guest_food_allergy,
        )
        blackboard.completed = True
        node._trace(
            "bt handle complete | "
            f"robot_text={blackboard.robot_text[:240]!r}"
        )
        return Status.SUCCESS

    @staticmethod
    def _bt_handle_incomplete(blackboard: AskNameAndDrinkBlackboard, node: "AskNameAndDrinkActionNode") -> Status:
        node._visual_phase = "decision"
        node._visual_note = "Missing one entity"
        blackboard.robot_text = confirmation_sentence(
            blackboard.guest_name,
            blackboard.guest_drink,
            blackboard.guest_food_flavour_preference,
            blackboard.guest_food_allergy,
        )
        node._trace(
            "bt handle incomplete | "
            f"robot_text={blackboard.robot_text[:240]!r}"
        )
        return Status.SUCCESS

    @staticmethod
    def _bt_handle_error(blackboard: AskNameAndDrinkBlackboard, node: "AskNameAndDrinkActionNode") -> Status:
        blackboard.robot_text = (
            "I could not understand that properly. "
            "Please tell me your name and favourite drink again."
        )
        return Status.SUCCESS

    def execute_callback(self, goal_handle) -> AskNameAndDrink.Result:
        goal = goal_handle.request
        if bool(goal.reset_session):
            self.blackboard.reset_session()

        self._active_goal = True
        self._visual_phase = "goal_start"
        self._visual_note = "Action goal accepted"
        self.blackboard.think = bool(goal.think)
        self.blackboard.max_attempts = max(
            1,
            int(goal.max_attempts) if int(goal.max_attempts) > 0 else self.default_max_attempts,
        )
        self.blackboard.clear_turn()
        action_started = time.time()
        self._trace(
            "goal start | "
            f"think={self.blackboard.think} "
            f"max_attempts={self.blackboard.max_attempts} "
            f"reset_session={bool(goal.reset_session)}"
        )
        self._publish_bt_snapshot(event="goal_start")

        result = AskNameAndDrink.Result()

        while self.blackboard.attempt_count < self.blackboard.max_attempts and not self.blackboard.completed:
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result.success = False
                result.completed = self.blackboard.completed
                result.task = self.blackboard.task
                result.reason = self.blackboard.reason
                result.name = self.blackboard.guest_name or ""
                result.drink = self.blackboard.guest_drink or ""
                result.food_flavour_preference = self.blackboard.guest_food_flavour_preference or ""
                result.food_allergy = self.blackboard.guest_food_allergy or ""
                result.robot_text = self.blackboard.robot_text
                result.last_user_input = self.blackboard.last_user_input
                result.spoke_response = self.blackboard.spoke_response
                result.message = "Goal canceled."
                result.elapsed_seconds = float(self.blackboard.elapsed_s)
                result.attempt_count = int(self.blackboard.attempt_count)
                self._active_goal = False
                return result

            self.blackboard.clear_turn()
            self.blackboard.attempt_count += 1
            self._visual_phase = "turn_start"
            self._visual_note = f"Starting turn {self.blackboard.attempt_count}"
            self._trace(
                "turn start | "
                f"attempt={self.blackboard.attempt_count}/{self.blackboard.max_attempts} "
                f"known_name={self.blackboard.guest_name!r} "
                f"known_drink={self.blackboard.guest_drink!r} "
                f"known_food_flavour_preference={self.blackboard.guest_food_flavour_preference!r} "
                f"known_food_allergy={self.blackboard.guest_food_allergy!r}"
            )
            self._publish_bt_snapshot(event="turn_start")

            self._publish_feedback(goal_handle, "listening")
            self._tree.tick()
            self._bt_tick_count += 1
            turn_status = self._tree.root.status
            self._publish_bt_snapshot(event="tick")
            if turn_status != Status.SUCCESS and not self.blackboard.last_error:
                self.blackboard.last_error = "Behaviour tree turn failed."

            stage = "completed" if self.blackboard.completed else "followup"
            if self.blackboard.last_error:
                stage = "error"
            self._publish_feedback(goal_handle, stage)

            if self.blackboard.last_error and not self.blackboard.robot_text:
                self.blackboard.robot_text = (
                    "I could not understand that properly. "
                    "Please tell me your name and favourite drink again."
                )
            if self.blackboard.last_error and self.blackboard.error_stage == "classify":
                self._trace("classify failure recovery | resetting conversation context")
                self._visual_phase = "classify_recover"
                self._visual_note = "Classification failed, resetting context"
                self.blackboard.reset_context()
                self.blackboard.robot_text = self.restart_prompt
                self.blackboard.last_error = ""
                self.blackboard.error_stage = ""
                speak_ok, speak_message = self._speak_text(self.blackboard.robot_text)
                self.blackboard.spoke_response = speak_ok
                if not speak_ok:
                    self.blackboard.last_error = speak_message
            elif self.blackboard.last_error and self.blackboard.error_stage == "input":
                self._trace("input failure recovery | keeping current context for next turn")
                self._visual_phase = "input_retry"
                self._visual_note = "Input failed, retrying next turn"
                self.blackboard.error_stage = ""
            if self.blackboard.stop_after_turn:
                self._trace("llm requested stop | ending action after current turn")
                self._visual_phase = "goal_end"
                self._visual_note = "LLM requested stop"
                self._publish_bt_snapshot(event="llm_stop")
                break
            if not self.blackboard.completed:
                self._visual_phase = "retry_turn"
                self._visual_note = "Preparing next turn"

            self._trace(
                "turn end | "
                f"attempt={self.blackboard.attempt_count} "
                f"completed={self.blackboard.completed} "
                f"last_error={self.blackboard.last_error[:240]!r} "
                f"robot_text={self.blackboard.robot_text[:240]!r}"
            )

        goal_handle.succeed()
        result.success = not bool(self.blackboard.last_error) and self.blackboard.completed
        result.completed = self.blackboard.completed
        result.task = self.blackboard.task
        result.reason = self.blackboard.reason
        result.name = self.blackboard.guest_name or ""
        result.drink = self.blackboard.guest_drink or ""
        result.food_flavour_preference = self.blackboard.guest_food_flavour_preference or ""
        result.food_allergy = self.blackboard.guest_food_allergy or ""
        result.robot_text = self.blackboard.robot_text
        result.last_user_input = self.blackboard.last_user_input
        result.spoke_response = self.blackboard.spoke_response
        if self.blackboard.last_error:
            result.message = self.blackboard.last_error
        elif self.blackboard.completed:
            result.message = "ok"
        else:
            result.message = (
                f"Incomplete after {self.blackboard.attempt_count} attempts. "
                f"name={self.blackboard.guest_name or 'missing'}, "
                f"drink={self.blackboard.guest_drink or 'missing'}, "
                f"food_flavour_preference={self.blackboard.guest_food_flavour_preference or 'missing'}, "
                f"food_allergy={self.blackboard.guest_food_allergy or 'missing'}."
            )
        result.elapsed_seconds = float(self.blackboard.elapsed_s)
        result.attempt_count = int(self.blackboard.attempt_count)
        self._trace(
            "goal end | "
            f"success={result.success} "
            f"completed={result.completed} "
            f"name={result.name!r} "
            f"drink={result.drink!r} "
            f"food_flavour_preference={result.food_flavour_preference!r} "
            f"food_allergy={result.food_allergy!r} "
            f"attempt_count={result.attempt_count} "
            f"last_turn_elapsed_s={result.elapsed_seconds:.3f} "
            f"total_action_elapsed_s={time.time() - action_started:.3f}"
        )
        self._visual_phase = "goal_end"
        self._visual_note = "Action finished"
        self._publish_bt_snapshot(event="goal_end")
        self._active_goal = False
        return result


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = AskNameAndDrinkActionNode()
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

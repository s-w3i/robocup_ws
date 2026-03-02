#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

DEFAULT_COQUI_SITE_PACKAGES = "/home/usern/coqui-venv/lib/python3.10/site-packages"
SYSTEM_SITE_PATH_PREFIXES = (
    "/usr/lib/python3/dist-packages",
    "/usr/local/lib/python3.10/dist-packages",
)
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
VALID_STATUSES = ("sleep", "listening", "idle", "operating")

CHAT_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "assistant_reply": {"type": "string"},
        "end_session": {"type": "boolean"},
        "end_reason": {"type": ["string", "null"]},
    },
    "required": ["assistant_reply", "end_session", "end_reason"],
}


def prepend_site_packages(path: str) -> bool:
    p = Path(path).expanduser().resolve()
    if not p.is_dir():
        return False
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)
    return True


def activate_coqui_site_packages(path: str, isolate: bool) -> tuple[bool, int]:
    added = prepend_site_packages(path)
    removed = 0
    if isolate:
        filtered = []
        for entry in sys.path:
            if entry.startswith(SYSTEM_SITE_PATH_PREFIXES):
                removed += 1
                continue
            filtered.append(entry)
        sys.path[:] = filtered
    return added, removed


prepend_site_packages(os.environ.get("COQUI_VENV_SITE_PACKAGES", DEFAULT_COQUI_SITE_PACKAGES))

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


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class OllamaChatbotNode(Node):
    def __init__(self) -> None:
        super().__init__("ollama_chatbot_node")

        self.declare_parameter("awake_topic", "/awake")
        self.declare_parameter("awake_greeting_done_topic", "/awake_greeting_done")
        self.declare_parameter("get_command_service", "/get_command")
        self.declare_parameter("robot_status_service", "/robot_status")
        self.declare_parameter("speak_action_name", "/coqui_tts/speak")
        self.declare_parameter("get_command_fail_window_sec", 10.0)
        self.declare_parameter("get_command_retry_delay_sec", 0.2)
        self.declare_parameter("service_response_timeout_sec", 30.0)
        self.declare_parameter("ollama_base_url", os.environ.get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL))
        self.declare_parameter("ollama_model", os.environ.get("TEXT_MODEL", "qwen3:14b"))
        self.declare_parameter("ollama_keep_alive", os.environ.get("OLLAMA_KEEP_ALIVE", "30m"))
        self.declare_parameter("ollama_timeout_sec", 180.0)
        self.declare_parameter("ollama_temperature", 0.2)
        self.declare_parameter("ollama_num_ctx", 2048)
        self.declare_parameter("ollama_num_batch", 128)
        self.declare_parameter("ollama_think", False)
        self.declare_parameter("auto_start_ollama", True)
        self.declare_parameter("ollama_start_timeout_sec", 20.0)
        self.declare_parameter("max_history_messages", 12)
        self.declare_parameter(
            "chat_system_prompt",
            (
                "You are EVA, a helpful home service robot. Keep answers short and clear for speech output. "
                "Always respond in English. Set end_session=true when the user clearly wants to end the conversation "
                "(for example: bye, goodbye, that's all, no more, stop chatting, go to sleep). "
                "When end_session=true, provide a short polite closing sentence in assistant_reply."
            ),
        )
        self.declare_parameter("fallback_error_reply", "Sorry, I am having trouble right now.")
        self.declare_parameter("extra_site_packages", DEFAULT_COQUI_SITE_PACKAGES)
        self.declare_parameter("isolate_site_packages", True)

        self.awake_topic = str(self.get_parameter("awake_topic").value)
        self.awake_greeting_done_topic = str(self.get_parameter("awake_greeting_done_topic").value)
        self.get_command_service = str(self.get_parameter("get_command_service").value)
        self.robot_status_service = str(self.get_parameter("robot_status_service").value)
        self.speak_action_name = str(self.get_parameter("speak_action_name").value)
        self.get_command_fail_window_sec = float(self.get_parameter("get_command_fail_window_sec").value)
        self.get_command_retry_delay_sec = float(self.get_parameter("get_command_retry_delay_sec").value)
        self.service_response_timeout_sec = float(self.get_parameter("service_response_timeout_sec").value)
        self.ollama_base_url = str(self.get_parameter("ollama_base_url").value).rstrip("/")
        self.ollama_model = str(self.get_parameter("ollama_model").value)
        self.ollama_keep_alive = str(self.get_parameter("ollama_keep_alive").value)
        self.ollama_timeout_sec = float(self.get_parameter("ollama_timeout_sec").value)
        self.ollama_temperature = float(self.get_parameter("ollama_temperature").value)
        self.ollama_num_ctx = int(self.get_parameter("ollama_num_ctx").value)
        self.ollama_num_batch = int(self.get_parameter("ollama_num_batch").value)
        self.ollama_think = bool(self.get_parameter("ollama_think").value)
        self.auto_start_ollama = bool(self.get_parameter("auto_start_ollama").value)
        self.ollama_start_timeout_sec = float(self.get_parameter("ollama_start_timeout_sec").value)
        self.max_history_messages = max(2, int(self.get_parameter("max_history_messages").value))
        self.chat_system_prompt = str(self.get_parameter("chat_system_prompt").value).strip()
        self.fallback_error_reply = str(self.get_parameter("fallback_error_reply").value).strip()
        self.extra_site_packages = str(self.get_parameter("extra_site_packages").value)
        self.isolate_site_packages = bool(self.get_parameter("isolate_site_packages").value)

        added, removed = activate_coqui_site_packages(
            self.extra_site_packages, self.isolate_site_packages
        )

        self._shutdown_event = threading.Event()
        self._session_cancel_event = threading.Event()
        self._state_lock = threading.Lock()
        self._session_active = False
        self._pending_awake = False
        self._session_thread: threading.Thread | None = None

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

        if self.auto_start_ollama:
            try:
                self._ensure_ollama_running()
            except Exception as exc:
                self.get_logger().error(f"Failed to start/reach Ollama: {exc}")
        elif not self._ollama_ready(timeout=1.0):
            self.get_logger().warn(
                f"Ollama is not reachable at {self.ollama_base_url}. "
                "Chat sessions will fail until it is available."
            )

        self.get_logger().info(
            "Ollama chatbot node ready. "
            f"model='{self.ollama_model}' get_command={self.get_command_service} "
            f"speak_action={self.speak_action_name} status_service={self.robot_status_service}"
        )
        self.get_logger().info(
            f"Session start trigger: {self.awake_topic}=true + {self.awake_greeting_done_topic}=true"
        )
        self.get_logger().info(
            f"extra_site_packages={self.extra_site_packages} isolate_site_packages={self.isolate_site_packages} "
            f"(added={added}, removed={removed})"
        )

    def _awake_callback(self, msg: Bool) -> None:
        value = bool(msg.data)
        with self._state_lock:
            if not value:
                self._pending_awake = False
                if self._session_active:
                    self._session_cancel_event.set()
                return
            self._pending_awake = True
        self.get_logger().info(
            "Received /awake=true. Waiting for awake greeting completion to start chat session."
        )

    def _awake_greeting_done_callback(self, msg: Bool) -> None:
        if not bool(msg.data):
            return

        with self._state_lock:
            if not self._pending_awake:
                return
            if self._session_active:
                return
            self._pending_awake = False
            self._session_active = True
            self._session_cancel_event.clear()

            self._session_thread = threading.Thread(
                target=self._chat_session_loop,
                daemon=True,
            )
            self._session_thread.start()

    def _chat_session_loop(self) -> None:
        end_reason = "session complete"
        self.get_logger().info("Chat session started.")

        history: list[dict[str, str]] = []
        if self.chat_system_prompt:
            history.append({"role": "system", "content": self.chat_system_prompt})

        fail_window_start: float | None = None
        try:
            while not self._shutdown_event.is_set():
                if self._session_cancel_event.is_set():
                    end_reason = "Canceled by /awake=false."
                    break

                call_start_mono = time.monotonic()
                ok, user_text, fail_message = self._call_get_command()
                now_mono = time.monotonic()
                if not ok:
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

                try:
                    assistant_reply, should_end, llm_reason = self._chat_with_ollama(history)
                except Exception as exc:
                    self.get_logger().error(f"Ollama chat failed: {exc}")
                    assistant_reply = self.fallback_error_reply or "Sorry, I am having trouble right now."
                    should_end = True
                    llm_reason = f"Ollama error: {exc}"

                if assistant_reply:
                    history.append({"role": "assistant", "content": assistant_reply})
                    self._trim_history(history)
                    speak_ok, speak_message = self._speak_text(assistant_reply)
                    if not speak_ok:
                        end_reason = f"SpeakText action failed: {speak_message}"
                        break

                if should_end:
                    end_reason = llm_reason or "LLM set end_session=true."
                    break
        finally:
            self.get_logger().info(f"Chat session ending. reason={end_reason}")
            self._set_robot_status("sleep")
            with self._state_lock:
                self._session_active = False
                self._pending_awake = False
                self._session_cancel_event.clear()
            self.get_logger().info("Chat session ended.")

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

    def _ollama_ready(self, timeout: float = 1.0) -> bool:
        try:
            r = requests.get(f"{self.ollama_base_url}/api/tags", timeout=timeout)
            return r.ok
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

    def _chat_with_ollama(
        self, history: list[dict[str, str]]
    ) -> tuple[str, bool, str]:
        payload: dict[str, Any] = {
            "model": self.ollama_model,
            "messages": history,
            "stream": False,
            "keep_alive": self.ollama_keep_alive,
            "format": CHAT_RESPONSE_SCHEMA,
            "options": {
                "temperature": self.ollama_temperature,
                "num_ctx": self.ollama_num_ctx,
                "num_batch": self.ollama_num_batch,
            },
            "think": bool(self.ollama_think),
        }

        r = requests.post(
            f"{self.ollama_base_url}/api/chat",
            json=payload,
            timeout=max(1.0, self.ollama_timeout_sec),
        )
        r.raise_for_status()
        content = (r.json().get("message", {}).get("content") or "").strip()
        if not content:
            return "Could you repeat that, please?", False, ""

        try:
            parsed = self._parse_json_relaxed(content)
            reply = str(parsed.get("assistant_reply", "")).strip()
            should_end = _to_bool(parsed.get("end_session"))
            end_reason_raw = parsed.get("end_reason")
            end_reason = "" if end_reason_raw is None else str(end_reason_raw).strip()
            if not reply and not should_end:
                reply = "Could you repeat that, please?"
            return reply, should_end, end_reason
        except Exception:
            # Fall back to raw text if model ignores schema.
            return content, False, ""

    def _call_get_command(self) -> tuple[bool, str, str]:
        if not self._get_command_client.wait_for_service(timeout_sec=0.5):
            return False, "", f"Service '{self.get_command_service}' not ready."

        req = Trigger.Request()
        future = self._get_command_client.call_async(req)
        ok, response, error_text = self._wait_for_future(
            future,
            self.service_response_timeout_sec,
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

        req = RobotStatus.Request()
        req.status = normalized
        future = self._robot_status_client.call_async(req)
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
        self.get_logger().info(
            f"Robot status set to '{normalized}' (reported='{response.status}')."
        )
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
                return False, None, f"Timed out after {timeout_sec:.1f}s."

        done_future = holder.get("future")
        if done_future is None:
            return False, None, "Future finished without result."
        try:
            return True, done_future.result(), ""
        except Exception as exc:
            return False, None, str(exc)

    def destroy_node(self) -> bool:
        self._shutdown_event.set()
        self._session_cancel_event.set()
        if self._session_thread is not None and self._session_thread.is_alive():
            self._session_thread.join(timeout=2.0)
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = OllamaChatbotNode()
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

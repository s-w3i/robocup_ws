from __future__ import annotations

import os
import re
import shlex
import subprocess
import threading
import time
import uuid
from typing import Any

import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import Bool
from std_srvs.srv import Trigger

from coqui_tts_interfaces.action import SpeakText
from coqui_tts_interfaces.srv import RobotStatus


VALID_STATUSES = {"sleep", "listening", "idle", "thinking", "operating"}


def _to_bool_env(name: str, default: bool) -> bool:
    raw = (os.environ.get(name, "1" if default else "0") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


class NanobotVoiceAgentNode(Node):
    def __init__(self) -> None:
        super().__init__("nanobot_voice_agent_node")

        self.declare_parameter("awake_topic", "/awake")
        self.declare_parameter("awake_greeting_done_topic", "/awake_greeting_done")
        self.declare_parameter("get_command_service", "/get_command")
        self.declare_parameter("robot_status_service", "/robot_status")
        self.declare_parameter("speak_action_name", "/coqui_tts/speak")
        self.declare_parameter("service_response_timeout_sec", 30.0)
        self.declare_parameter("nanobot_executable", "/home/usern/nanobot/.venv/bin/nanobot")
        self.declare_parameter("nanobot_workdir", "/home/usern/nanobot")
        self.declare_parameter("nanobot_config_path", "")
        self.declare_parameter("nanobot_workspace", "")
        self.declare_parameter("nanobot_session_prefix", "voice")
        self.declare_parameter("nanobot_timeout_sec", 120.0)
        self.declare_parameter("nanobot_logs", False)
        self.declare_parameter("fallback_error_reply", "Sorry, I had trouble answering that.")

        self.awake_topic = str(self.get_parameter("awake_topic").value)
        self.awake_greeting_done_topic = str(self.get_parameter("awake_greeting_done_topic").value)
        self.get_command_service = str(self.get_parameter("get_command_service").value)
        self.robot_status_service = str(self.get_parameter("robot_status_service").value)
        self.speak_action_name = str(self.get_parameter("speak_action_name").value)
        self.service_response_timeout_sec = max(
            0.5, float(self.get_parameter("service_response_timeout_sec").value)
        )
        self.nanobot_executable = str(self.get_parameter("nanobot_executable").value).strip()
        self.nanobot_workdir = str(self.get_parameter("nanobot_workdir").value).strip() or "/home/usern/nanobot"
        self.nanobot_config_path = str(self.get_parameter("nanobot_config_path").value).strip()
        self.nanobot_workspace = str(self.get_parameter("nanobot_workspace").value).strip()
        self.nanobot_session_prefix = (
            str(self.get_parameter("nanobot_session_prefix").value).strip() or "voice"
        )
        self.nanobot_timeout_sec = max(5.0, float(self.get_parameter("nanobot_timeout_sec").value))
        self.nanobot_logs = bool(self.get_parameter("nanobot_logs").value)
        self.fallback_error_reply = (
            str(self.get_parameter("fallback_error_reply").value).strip()
            or "Sorry, I had trouble answering that."
        )

        self._shutdown_event = threading.Event()
        self._session_cancel_event = threading.Event()
        self._state_lock = threading.Lock()
        self._awake_initialized = False
        self._awake_is_true = False
        self._pending_awake = False
        self._session_active = False
        self._session_thread: threading.Thread | None = None
        self._current_session_id = ""

        status_qos = QoSProfile(depth=1)
        status_qos.reliability = QoSReliabilityPolicy.RELIABLE
        status_qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        self._awake_sub = self.create_subscription(
            Bool, self.awake_topic, self._awake_callback, status_qos
        )
        self._awake_greeting_done_sub = self.create_subscription(
            Bool, self.awake_greeting_done_topic, self._awake_greeting_done_callback, status_qos
        )

        self._callback_group = ReentrantCallbackGroup()
        self._get_command_client = self.create_client(
            Trigger, self.get_command_service, callback_group=self._callback_group
        )
        self._robot_status_client = self.create_client(
            RobotStatus, self.robot_status_service, callback_group=self._callback_group
        )
        self._speak_action_client = ActionClient(
            self, SpeakText, self.speak_action_name, callback_group=self._callback_group
        )

        self.get_logger().info(
            "Nanobot voice agent ready. "
            f"awake={self.awake_topic} awake_done={self.awake_greeting_done_topic} "
            f"get_command={self.get_command_service} speak_action={self.speak_action_name}"
        )

    def destroy_node(self) -> bool:
        self._shutdown_event.set()
        self._session_cancel_event.set()
        return super().destroy_node()

    def _awake_callback(self, msg: Bool) -> None:
        value = bool(msg.data)
        startup_state_observed = False
        ignored_startup_true = False
        with self._state_lock:
            if not self._awake_initialized:
                self._awake_initialized = True
                self._awake_is_true = value
                startup_state_observed = True
                ignored_startup_true = value
                # Always ignore the first latched /awake state at startup.
                # A session should only begin after a fresh wake event that
                # happens after the node is already running.
                self._pending_awake = False
            else:
                if not value:
                    self._awake_is_true = False
                    self._pending_awake = False
                    if self._session_active:
                        self._session_cancel_event.set()
                    else:
                        self._set_robot_status("sleep")
                    return
                if self._awake_is_true:
                    return
                self._awake_is_true = True
                self._pending_awake = True
        if startup_state_observed:
            if ignored_startup_true:
                self.get_logger().info(
                    "Ignoring latched /awake=true at startup. Waiting for a fresh wake cycle."
                )
                return
            self.get_logger().info("Startup /awake state is false. Waiting for a fresh /awake=true.")
            return
        self.get_logger().info(
            "Received /awake=true. Waiting for awake greeting completion to start nanobot session."
        )

    def _awake_greeting_done_callback(self, msg: Bool) -> None:
        if not bool(msg.data):
            return

        with self._state_lock:
            if not self._pending_awake or self._session_active:
                return
            self._pending_awake = False
            self._session_active = True
            self._session_cancel_event.clear()
            self._current_session_id = f"{self.nanobot_session_prefix}:{uuid.uuid4().hex[:12]}"
            self._session_thread = threading.Thread(
                target=self._voice_session_loop,
                name="nanobot-voice-session",
                daemon=True,
            )
            self._session_thread.start()

    def _voice_session_loop(self) -> None:
        end_reason = "session complete"
        self.get_logger().info(f"Nanobot voice session started: {self._current_session_id}")

        try:
            while not self._shutdown_event.is_set():
                if self._session_cancel_event.is_set():
                    end_reason = "Canceled by /awake=false."
                    break

                self._set_robot_status("listening")
                ok, user_text, fail_message = self._call_get_command()
                if not ok:
                    end_reason = fail_message or "No follow-up command received."
                    break

                cleaned_user = user_text.strip()
                if not cleaned_user:
                    end_reason = "Empty follow-up command."
                    break

                self.get_logger().info(f"User: {cleaned_user}")
                self._set_robot_status("thinking")
                ok, reply_text, fail_message = self._run_nanobot_turn(
                    cleaned_user,
                    self._current_session_id,
                )
                if not ok:
                    reply_text = self.fallback_error_reply
                    self.get_logger().warn(f"Nanobot turn failed: {fail_message}")

                explicit_say = self._looks_like_explicit_say(cleaned_user)
                if not explicit_say and reply_text.strip():
                    speak_ok, speak_message = self._speak_text(reply_text)
                    if not speak_ok:
                        end_reason = f"SpeakText action failed: {speak_message}"
                        break
        finally:
            self.get_logger().info(f"Nanobot voice session ending. reason={end_reason}")
            self._set_robot_status("sleep")
            with self._state_lock:
                self._pending_awake = False
                self._session_active = False
                self._session_cancel_event.clear()
                self._current_session_id = ""
            self.get_logger().info("Nanobot voice session ended.")

    def _call_get_command(self) -> tuple[bool, str, str]:
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

    def _run_nanobot_turn(self, user_text: str, session_id: str) -> tuple[bool, str, str]:
        env = os.environ.copy()
        env["NANOBOT_DISABLE_AUTO_TTS"] = "1"
        env["NANOBOT_SPEAK_TEXT_WAIT_UNTIL_DONE"] = "1"
        command = [
            self.nanobot_executable,
            "agent",
            "--session",
            session_id,
            "--message",
            user_text,
            "--raw-output",
            "--no-markdown",
        ]
        if self.nanobot_config_path:
            command.extend(["--config", self.nanobot_config_path])
        if self.nanobot_workspace:
            command.extend(["--workspace", self.nanobot_workspace])
        if self.nanobot_logs:
            command.append("--logs")
        self.get_logger().info(
            f"Running nanobot turn: {' '.join(shlex.quote(part) for part in command)}"
        )
        process = subprocess.Popen(  # pylint: disable=consider-using-with
            command,
            cwd=self.nanobot_workdir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        deadline = time.time() + self.nanobot_timeout_sec
        while time.time() < deadline:
            if self._shutdown_event.is_set() or self._session_cancel_event.is_set():
                process.terminate()
                try:
                    process.wait(timeout=2.0)
                except Exception:
                    process.kill()
                return False, "", "Canceled."
            if process.poll() is not None:
                stdout = process.stdout.read() if process.stdout is not None else ""
                stderr = process.stderr.read() if process.stderr is not None else ""
                reply = self._extract_nanobot_reply(stdout)
                if process.returncode != 0:
                    return False, reply, stderr.strip() or f"nanobot exited with {process.returncode}"
                if not reply:
                    return False, "", stderr.strip() or "Nanobot returned no reply."
                return True, reply, ""
            time.sleep(0.1)

        process.terminate()
        try:
            process.wait(timeout=2.0)
        except Exception:
            process.kill()
        return False, "", f"Nanobot timed out after {self.nanobot_timeout_sec:.1f}s."

    @staticmethod
    def _extract_nanobot_reply(stdout: str) -> str:
        text = str(stdout or "").strip()
        if not text:
            return ""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return ""
        return " ".join(lines)

    @staticmethod
    def _looks_like_explicit_say(user_text: str) -> bool:
        lowered = str(user_text or "").strip().lower()
        return bool(re.match(r"^(say|speak)\s+", lowered) or lowered.startswith("read aloud "))

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
        if not ok or response is None or not response.success:
            self.get_logger().warn(
                f"Failed to set robot status '{normalized}': {error_text or getattr(response, 'message', '')}"
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
                return False, None, "Timed out."

        fut = holder.get("future", future)
        exc = fut.exception()
        if exc is not None:
            return False, None, str(exc)
        return True, fut.result(), ""


def main() -> None:
    rclpy.init()
    node = NanobotVoiceAgentNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.remove_node(node)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

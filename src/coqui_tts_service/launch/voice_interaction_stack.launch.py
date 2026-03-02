#!/usr/bin/env python3

import os
from pathlib import Path

from ament_index_python.packages import get_package_prefix
from launch import LaunchDescription
from launch.actions import RegisterEventHandler, SetEnvironmentVariable
from launch.event_handlers import OnProcessIO
from launch_ros.actions import Node


def _resolved_pythonpath() -> str:
    current = os.environ.get("PYTHONPATH", "")
    entries = [p for p in current.split(":") if p]

    try:
        pkg_prefix = Path(get_package_prefix("coqui_tts_service")).resolve()
        ws_root = pkg_prefix.parents[1]
        dev_path = str((ws_root / "build" / "coqui_tts_service").resolve())
    except Exception:
        dev_path = ""

    merged = []
    if dev_path:
        merged.append(dev_path)
    for path in entries:
        if path not in merged:
            merged.append(path)
    return ":".join(merged)


def generate_launch_description() -> LaunchDescription:
    set_pythonpath = SetEnvironmentVariable(
        name="PYTHONPATH",
        value=_resolved_pythonpath(),
    )

    robot_status_node = Node(
        package="coqui_tts_service",
        executable="robot_status_node",
        name="robot_status_node",
        output="screen",
    )

    whisper_command_node = Node(
        package="coqui_tts_service",
        executable="whisper_command_node",
        name="whisper_command_node",
        output="screen",
    )

    coqui_talking_face_action_node = Node(
        package="coqui_tts_service",
        executable="coqui_talking_face_action_node",
        name="coqui_talking_face_action_node",
        output="screen",
    )

    whisper_started = {"value": False}

    def _start_whisper_after_ready(event, *_args, **_kwargs):
        text = event.text.decode(errors="ignore") if isinstance(event.text, bytes) else str(event.text)
        if "VOICE_STACK_READY" not in text:
            return []
        if whisper_started["value"]:
            return []
        whisper_started["value"] = True
        return [whisper_command_node]

    start_whisper_on_ready = RegisterEventHandler(
        OnProcessIO(
            target_action=coqui_talking_face_action_node,
            on_stdout=_start_whisper_after_ready,
            on_stderr=_start_whisper_after_ready,
        )
    )

    return LaunchDescription(
        [
            set_pythonpath,
            robot_status_node,
            coqui_talking_face_action_node,
            start_whisper_on_ready,
        ]
    )

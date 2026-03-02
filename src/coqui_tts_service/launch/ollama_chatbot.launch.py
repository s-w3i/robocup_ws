#!/usr/bin/env python3

import os
from pathlib import Path

from ament_index_python.packages import get_package_prefix
from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable
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

    ollama_chatbot_node = Node(
        package="coqui_tts_service",
        executable="ollama_chatbot_node",
        name="ollama_chatbot_node",
        output="screen",
    )

    return LaunchDescription(
        [
            set_pythonpath,
            ollama_chatbot_node,
        ]
    )

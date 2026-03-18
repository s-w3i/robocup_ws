#!/usr/bin/env python3

import os
from pathlib import Path

from ament_index_python.packages import get_package_prefix
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
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
    debug_text_input_mode = LaunchConfiguration("debug_text_input_mode")
    debug_text_input_prompt = LaunchConfiguration("debug_text_input_prompt")

    set_pythonpath = SetEnvironmentVariable(
        name="PYTHONPATH",
        value=_resolved_pythonpath(),
    )

    declare_debug_text_input_mode = DeclareLaunchArgument(
        "debug_text_input_mode",
        default_value="false",
        description="Use terminal text input instead of the /get_command voice service.",
    )
    declare_debug_text_input_prompt = DeclareLaunchArgument(
        "debug_text_input_prompt",
        default_value="You",
        description="Prompt label shown in chatbot debug text input mode.",
    )

    ollama_chatbot_node = Node(
        package="coqui_tts_service",
        executable="ollama_chatbot_node",
        name="ollama_chatbot_node",
        output="screen",
        emulate_tty=True,
        parameters=[
            {
                "debug_text_input_mode": debug_text_input_mode,
                "debug_text_input_prompt": debug_text_input_prompt,
            }
        ],
    )

    return LaunchDescription(
        [
            declare_debug_text_input_mode,
            declare_debug_text_input_prompt,
            set_pythonpath,
            ollama_chatbot_node,
        ]
    )

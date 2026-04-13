#!/usr/bin/env python3

import os
from pathlib import Path

from ament_index_python.packages import get_package_prefix
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


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
    launch_whisper = LaunchConfiguration("launch_whisper")
    default_camera_name = LaunchConfiguration("default_camera_name")
    color_topic = LaunchConfiguration("color_topic")
    depth_topic = LaunchConfiguration("depth_topic")
    camera_info_topic = LaunchConfiguration("camera_info_topic")
    camera_link_frame = LaunchConfiguration("camera_link_frame")
    ollama_base_url = LaunchConfiguration("ollama_base_url")
    chat_model = LaunchConfiguration("chat_model")
    vlm_model = LaunchConfiguration("vlm_model")
    yolo_model_path = LaunchConfiguration("yolo_model_path")
    yolo_python_site_packages = LaunchConfiguration("yolo_python_site_packages")
    face_fullscreen = LaunchConfiguration("face_fullscreen")

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

    coqui_talking_face_action_node = Node(
        package="coqui_tts_service",
        executable="coqui_talking_face_action_node",
        name="coqui_talking_face_action_node",
        output="screen",
        parameters=[
            {
                "face_fullscreen": face_fullscreen,
            }
        ],
    )

    whisper_command_node = Node(
        package="coqui_tts_service",
        executable="whisper_command_node",
        name="whisper_command_node",
        output="screen",
        condition=IfCondition(
            PythonExpression(
                [
                    "'",
                    launch_whisper,
                    "' == 'true' and '",
                    debug_text_input_mode,
                    "' != 'true'",
                ]
            )
        ),
    )

    vlm_query_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("vlm_service"), "launch", "vlm_query_service.launch.py"]
            )
        ),
        launch_arguments={
            "service_name": "/vlm/query",
            "ollama_base_url": ollama_base_url,
            "vlm_model": vlm_model,
            "default_camera_name": default_camera_name,
            "default_camera_topic": color_topic,
            "camera_names_csv": default_camera_name,
            "camera_topics_csv": color_topic,
            "manage_robot_status": "true",
            "robot_status_service": "/robot_status",
        }.items(),
    )

    yolo_detection_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("yoloe_detection_service"), "launch", "yoloe_detection_service.launch.py"]
            )
        ),
        launch_arguments={
            "service_name": "/yoloe/detect_prompt",
            "model_path": yolo_model_path,
            "default_camera_name": default_camera_name,
            "color_topic": color_topic,
            "depth_topic": depth_topic,
            "camera_info_topic": camera_info_topic,
            "camera_link_frame": camera_link_frame,
            "python_site_packages": yolo_python_site_packages,
        }.items(),
    )

    camera_snapshot_service_node = Node(
        package="vlm_service",
        executable="camera_snapshot_service_node",
        name="camera_snapshot_service_node",
        output="screen",
        parameters=[
            {
                "service_name": "/camera/capture",
                "default_camera_name": default_camera_name,
                "default_camera_topic": color_topic,
                "camera_names_csv": default_camera_name,
                "camera_topics_csv": color_topic,
                "default_save_dir": "/home/usern/robocup_ws/captures",
            }
        ],
    )

    home_assistant_orchestrator_node = Node(
        package="coqui_tts_service",
        executable="home_assistant_orchestrator_node",
        name="home_assistant_orchestrator_node",
        output="screen",
        emulate_tty=True,
        parameters=[
            {
                "debug_text_input_mode": debug_text_input_mode,
                "debug_text_input_prompt": debug_text_input_prompt,
                "ollama_base_url": ollama_base_url,
                "chat_model": chat_model,
                "default_camera_name": default_camera_name,
                "camera_names_csv": default_camera_name,
                "vlm_query_service": "/vlm/query",
                "detect_object_service": "/yoloe/detect_prompt",
                "capture_image_service": "/camera/capture",
                "robot_status_service": "/robot_status",
                "get_command_service": "/get_command",
                "speak_action_name": "/coqui_tts/speak",
            }
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("debug_text_input_mode", default_value="false"),
            DeclareLaunchArgument("debug_text_input_prompt", default_value="You"),
            DeclareLaunchArgument("launch_whisper", default_value="true"),
            DeclareLaunchArgument("default_camera_name", default_value="camera0"),
            DeclareLaunchArgument("color_topic", default_value="/camera0/color/image_raw"),
            DeclareLaunchArgument(
                "depth_topic", default_value="/camera0/realsense_splitter_node/output/depth"
            ),
            DeclareLaunchArgument("camera_info_topic", default_value="/camera0/color/camera_info"),
            DeclareLaunchArgument("camera_link_frame", default_value="camera0_link"),
            DeclareLaunchArgument("ollama_base_url", default_value="http://127.0.0.1:11434"),
            DeclareLaunchArgument("chat_model", default_value="qwen3.5:9b"),
            DeclareLaunchArgument("vlm_model", default_value="qwen3.5:9b"),
            DeclareLaunchArgument("yolo_model_path", default_value="/home/usern/yoloe-26l-seg.pt"),
            DeclareLaunchArgument(
                "yolo_python_site_packages",
                default_value="/home/usern/coqui-venv/lib/python3.10/site-packages",
            ),
            DeclareLaunchArgument("face_fullscreen", default_value="false"),
            set_pythonpath,
            robot_status_node,
            coqui_talking_face_action_node,
            whisper_command_node,
            vlm_query_launch,
            yolo_detection_launch,
            camera_snapshot_service_node,
            home_assistant_orchestrator_node,
        ]
    )

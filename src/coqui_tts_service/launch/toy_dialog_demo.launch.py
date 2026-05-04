#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    enabled = LaunchConfiguration("enabled")
    sim = LaunchConfiguration("sim")
    awake_topic = LaunchConfiguration("awake_topic")
    awake_greeting_done_topic = LaunchConfiguration("awake_greeting_done_topic")
    get_command_service = LaunchConfiguration("get_command_service")
    openai_api_url = LaunchConfiguration("openai_api_url")
    openai_api_key_env = LaunchConfiguration("openai_api_key_env")
    openai_model = LaunchConfiguration("openai_model")
    openai_timeout_sec = LaunchConfiguration("openai_timeout_sec")
    openai_retry_timeout_sec = LaunchConfiguration("openai_retry_timeout_sec")
    openai_max_retry_count = LaunchConfiguration("openai_max_retry_count")
    default_camera_name = LaunchConfiguration("default_camera_name")
    camera_names_csv = LaunchConfiguration("camera_names_csv")
    camera_topics_csv = LaunchConfiguration("camera_topics_csv")
    debug_text_input_mode = LaunchConfiguration("debug_text_input_mode")

    toy_dialog_demo_node = Node(
        package="coqui_tts_service",
        executable="toy_dialog_demo_node",
        name="toy_dialog_demo_node",
        output="screen",
        emulate_tty=True,
        parameters=[
            {
                "enabled": enabled,
                "sim": sim,
                "awake_topic": awake_topic,
                "awake_greeting_done_topic": awake_greeting_done_topic,
                "get_command_service": get_command_service,
                "robot_status_service": "/robot_status",
                "speak_action_name": "/coqui_tts/speak",
                "openai_api_url": openai_api_url,
                "openai_api_key_env": openai_api_key_env,
                "openai_model": openai_model,
                "openai_timeout_sec": openai_timeout_sec,
                "openai_retry_timeout_sec": openai_retry_timeout_sec,
                "openai_max_retry_count": openai_max_retry_count,
                "default_camera_name": default_camera_name,
                "camera_names_csv": camera_names_csv,
                "camera_topics_csv": camera_topics_csv,
                "image_wait_timeout_sec": 1.5,
                "debug_text_input_mode": debug_text_input_mode,
                "service_response_timeout_sec": 35.0,
                "get_command_fail_window_sec": 10.0,
            }
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("enabled", default_value="true"),
            DeclareLaunchArgument("sim", default_value="false"),
            DeclareLaunchArgument("awake_topic", default_value="/awake"),
            DeclareLaunchArgument("awake_greeting_done_topic", default_value="/awake_greeting_done"),
            DeclareLaunchArgument("get_command_service", default_value="/get_command"),
            DeclareLaunchArgument("openai_api_url", default_value="https://api.openai.com/v1/responses"),
            DeclareLaunchArgument("openai_api_key_env", default_value="OPENAI_API_KEY"),
            DeclareLaunchArgument("openai_model", default_value="gpt-5.5"),
            DeclareLaunchArgument("openai_timeout_sec", default_value="45.0"),
            DeclareLaunchArgument("openai_retry_timeout_sec", default_value="60.0"),
            DeclareLaunchArgument("openai_max_retry_count", default_value="1"),
            DeclareLaunchArgument("default_camera_name", default_value="camera0"),
            DeclareLaunchArgument("camera_names_csv", default_value="camera0,gripper_camera"),
            DeclareLaunchArgument(
                "camera_topics_csv",
                default_value="/camera0/color/image_raw,/gripper_camera/color/image_raw",
            ),
            DeclareLaunchArgument("debug_text_input_mode", default_value="false"),
            toy_dialog_demo_node,
        ]
    )

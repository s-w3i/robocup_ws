from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("awake_topic", default_value="/awake"),
            DeclareLaunchArgument("awake_greeting_done_topic", default_value="/awake_greeting_done"),
            DeclareLaunchArgument("get_command_service", default_value="/get_command"),
            DeclareLaunchArgument("robot_status_service", default_value="/robot_status"),
            DeclareLaunchArgument("speak_action_name", default_value="/coqui_tts/speak"),
            DeclareLaunchArgument(
                "nanobot_executable",
                default_value="/home/usern/nanobot/.venv/bin/nanobot",
            ),
            DeclareLaunchArgument(
                "nanobot_workdir",
                default_value="/home/usern/nanobot",
            ),
            DeclareLaunchArgument(
                "nanobot_config_path",
                default_value="",
            ),
            DeclareLaunchArgument(
                "nanobot_workspace",
                default_value="",
            ),
            DeclareLaunchArgument(
                "nanobot_logs",
                default_value="false",
            ),
            Node(
                package="coqui_tts_service",
                executable="nanobot_voice_agent_node",
                name="nanobot_voice_agent_node",
                output="screen",
                parameters=[
                    {
                        "awake_topic": LaunchConfiguration("awake_topic"),
                        "awake_greeting_done_topic": LaunchConfiguration("awake_greeting_done_topic"),
                        "get_command_service": LaunchConfiguration("get_command_service"),
                        "robot_status_service": LaunchConfiguration("robot_status_service"),
                        "speak_action_name": LaunchConfiguration("speak_action_name"),
                        "nanobot_executable": LaunchConfiguration("nanobot_executable"),
                        "nanobot_workdir": LaunchConfiguration("nanobot_workdir"),
                        "nanobot_config_path": LaunchConfiguration("nanobot_config_path"),
                        "nanobot_workspace": LaunchConfiguration("nanobot_workspace"),
                        "nanobot_logs": LaunchConfiguration("nanobot_logs"),
                    }
                ],
            ),
        ]
    )

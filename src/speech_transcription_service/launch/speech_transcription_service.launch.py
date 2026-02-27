from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            Node(
                package="speech_transcription_service",
                executable="speech_transcription_node",
                name="speech_transcription_node",
                output="screen",
            )
        ]
    )

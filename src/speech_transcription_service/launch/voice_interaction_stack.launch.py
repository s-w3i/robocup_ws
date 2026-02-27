from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    extra_site_packages_arg = DeclareLaunchArgument(
        "extra_site_packages",
        default_value="/home/usern/coqui-venv/lib/python3.10/site-packages",
        description="Python site-packages path for Coqui/Whisper runtime modules.",
    )
    isolate_site_packages_arg = DeclareLaunchArgument(
        "isolate_site_packages",
        default_value="true",
        description="Whether Coqui nodes isolate system site-packages.",
    )
    log_level_arg = DeclareLaunchArgument(
        "log_level",
        default_value="info",
        description="ROS log level for all nodes.",
    )
    wake_phrase_arg = DeclareLaunchArgument(
        "wake_phrase",
        default_value="hi eva",
        description="Wake phrase used by speech recognition node.",
    )
    warmup_enabled_arg = DeclareLaunchArgument(
        "warmup_enabled",
        default_value="true",
        description="Whether Coqui TTS model warmup runs at startup.",
    )
    warmup_text_arg = DeclareLaunchArgument(
        "warmup_text",
        default_value="service warmup",
        description="Warmup text for Coqui TTS startup pass.",
    )

    extra_site_packages = LaunchConfiguration("extra_site_packages")
    isolate_site_packages = LaunchConfiguration("isolate_site_packages")
    log_level = LaunchConfiguration("log_level")
    wake_phrase = LaunchConfiguration("wake_phrase")
    warmup_enabled = LaunchConfiguration("warmup_enabled")
    warmup_text = LaunchConfiguration("warmup_text")

    talking_face_node = Node(
        package="coqui_tts_service",
        executable="coqui_talking_face_action_node",
        name="coqui_talking_face_action_node",
        output="screen",
        parameters=[
            {
                "extra_site_packages": extra_site_packages,
                "isolate_site_packages": isolate_site_packages,
                "warmup_enabled": warmup_enabled,
                "warmup_text": warmup_text,
            }
        ],
        arguments=["--ros-args", "--log-level", log_level],
    )

    speech_recognition_node = Node(
        package="speech_transcription_service",
        executable="speech_transcription_node",
        name="speech_transcription_node",
        output="screen",
        parameters=[
            {
                "extra_site_packages": extra_site_packages,
                "wake_phrase": wake_phrase,
            }
        ],
        arguments=["--ros-args", "--log-level", log_level],
    )

    return LaunchDescription(
        [
            extra_site_packages_arg,
            isolate_site_packages_arg,
            log_level_arg,
            wake_phrase_arg,
            warmup_enabled_arg,
            warmup_text_arg,
            talking_face_node,
            speech_recognition_node,
        ]
    )

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    args = [
        DeclareLaunchArgument("service_name", default_value="/lost_found/vlm_check"),
        DeclareLaunchArgument("visual_question_service_name", default_value="/lost_found/visual_question"),
        DeclareLaunchArgument("vlm_query_service", default_value="/vlm/query"),
        DeclareLaunchArgument("capture_image_service", default_value="/camera/capture"),
        DeclareLaunchArgument("default_camera_name", default_value="camera0"),
        DeclareLaunchArgument("capture_before_visual_question", default_value="true"),
        DeclareLaunchArgument("visual_question_save_dir", default_value="/home/usern/robocup_ws/captures"),
        DeclareLaunchArgument("ollama_base_url", default_value="http://127.0.0.1:11434"),
        DeclareLaunchArgument("use_openai_vlm", default_value="false"),
        DeclareLaunchArgument("openai_api_url", default_value="https://api.openai.com/v1/responses"),
        DeclareLaunchArgument("openai_api_key_env", default_value="OPENAI_API_KEY"),
        DeclareLaunchArgument("openai_model", default_value="gpt-5.5"),
        DeclareLaunchArgument("vlm_model", default_value="qwen3.5:9b"),
        DeclareLaunchArgument("ollama_timeout_sec", default_value="45.0"),
        DeclareLaunchArgument("default_confidence_threshold", default_value="0.65"),
        DeclareLaunchArgument("vlm_query_timeout_sec", default_value="75.0"),
    ]

    node = Node(
        package="vlm_service",
        executable="lost_found_vlm_check_node",
        name="lost_found_vlm_check_node",
        output="screen",
        emulate_tty=True,
        parameters=[
            {
                "service_name": LaunchConfiguration("service_name"),
                "visual_question_service_name": LaunchConfiguration("visual_question_service_name"),
                "vlm_query_service": LaunchConfiguration("vlm_query_service"),
                "capture_image_service": LaunchConfiguration("capture_image_service"),
                "default_camera_name": LaunchConfiguration("default_camera_name"),
                "capture_before_visual_question": LaunchConfiguration("capture_before_visual_question"),
                "visual_question_save_dir": LaunchConfiguration("visual_question_save_dir"),
                "ollama_base_url": LaunchConfiguration("ollama_base_url"),
                "use_openai_vlm": LaunchConfiguration("use_openai_vlm"),
                "openai_api_url": LaunchConfiguration("openai_api_url"),
                "openai_api_key_env": LaunchConfiguration("openai_api_key_env"),
                "openai_model": LaunchConfiguration("openai_model"),
                "vlm_model": LaunchConfiguration("vlm_model"),
                "ollama_timeout_sec": LaunchConfiguration("ollama_timeout_sec"),
                "default_confidence_threshold": LaunchConfiguration("default_confidence_threshold"),
                "vlm_query_timeout_sec": LaunchConfiguration("vlm_query_timeout_sec"),
            }
        ],
    )

    return LaunchDescription(args + [node])

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    args = [
        DeclareLaunchArgument("service_name", default_value="/vlm/query"),
        DeclareLaunchArgument("ollama_base_url", default_value="http://127.0.0.1:11434"),
        DeclareLaunchArgument("use_openai_vlm", default_value="false"),
        DeclareLaunchArgument("openai_api_url", default_value="https://api.openai.com/v1/responses"),
        DeclareLaunchArgument("openai_api_key_env", default_value="OPENAI_API_KEY"),
        DeclareLaunchArgument("openai_model", default_value="gpt-5.5"),
        DeclareLaunchArgument("vlm_model", default_value="qwen3.5:9b"),
        DeclareLaunchArgument("default_camera_name", default_value="camera0"),
        DeclareLaunchArgument("default_camera_topic", default_value="/camera0/color/image_raw"),
        DeclareLaunchArgument("camera_names_csv", default_value=""),
        DeclareLaunchArgument("camera_topics_csv", default_value=""),
        DeclareLaunchArgument("image_wait_timeout_sec", default_value="3.0"),
        DeclareLaunchArgument("text_timeout_sec", default_value="60.0"),
        DeclareLaunchArgument("text_retry_timeout_sec", default_value="90.0"),
        DeclareLaunchArgument("text_fast_num_predict", default_value="24"),
        DeclareLaunchArgument("text_thinking_num_predict", default_value="96"),
        DeclareLaunchArgument("manage_robot_status", default_value="true"),
        DeclareLaunchArgument("robot_status_service", default_value="/robot_status"),
        DeclareLaunchArgument("robot_status_timeout_sec", default_value="2.0"),
    ]

    node = Node(
        package="vlm_service",
        executable="vlm_query_service_node",
        name="vlm_query_service_node",
        output="screen",
        emulate_tty=True,
        parameters=[
            {
                "service_name": LaunchConfiguration("service_name"),
                "ollama_base_url": LaunchConfiguration("ollama_base_url"),
                "use_openai_vlm": LaunchConfiguration("use_openai_vlm"),
                "openai_api_url": LaunchConfiguration("openai_api_url"),
                "openai_api_key_env": LaunchConfiguration("openai_api_key_env"),
                "openai_model": LaunchConfiguration("openai_model"),
                "vlm_model": LaunchConfiguration("vlm_model"),
                "default_camera_name": LaunchConfiguration("default_camera_name"),
                "default_camera_topic": LaunchConfiguration("default_camera_topic"),
                "camera_names_csv": LaunchConfiguration("camera_names_csv"),
                "camera_topics_csv": LaunchConfiguration("camera_topics_csv"),
                "image_wait_timeout_sec": LaunchConfiguration("image_wait_timeout_sec"),
                "text_timeout_sec": LaunchConfiguration("text_timeout_sec"),
                "text_retry_timeout_sec": LaunchConfiguration("text_retry_timeout_sec"),
                "text_fast_num_predict": LaunchConfiguration("text_fast_num_predict"),
                "text_thinking_num_predict": LaunchConfiguration("text_thinking_num_predict"),
                "manage_robot_status": LaunchConfiguration("manage_robot_status"),
                "robot_status_service": LaunchConfiguration("robot_status_service"),
                "robot_status_timeout_sec": LaunchConfiguration("robot_status_timeout_sec"),
            }
        ],
    )

    return LaunchDescription(args + [node])

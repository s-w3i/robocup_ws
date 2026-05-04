from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    args = [
        DeclareLaunchArgument("bot_token", default_value=""),
        DeclareLaunchArgument("bot_token_file", default_value="/home/usern/.config/telegram_lost_found/bot_token.txt"),
        DeclareLaunchArgument("allowed_user_ids", default_value=""),
        DeclareLaunchArgument("camera_name", default_value="gripper_camera"),
        DeclareLaunchArgument("camera_topic", default_value="/gripper_camera/color/image_raw"),
        DeclareLaunchArgument("location_image_service", default_value="/lost_found/location_image"),
        DeclareLaunchArgument("navigation_action_name", default_value="/navigate_to_pose"),
        DeclareLaunchArgument("arm_pose_service", default_value="/arm_pose"),
        DeclareLaunchArgument("arm_detect_pose_name", default_value="detect"),
        DeclareLaunchArgument("arm_zero_pose_name", default_value="zero"),
        DeclareLaunchArgument("demo_mode", default_value="false"),
        DeclareLaunchArgument("sim_mode", default_value="true"),
        DeclareLaunchArgument("demo_navigation_delay_sec", default_value="5.0"),
        DeclareLaunchArgument("location_pose_map_json", default_value=""),
        DeclareLaunchArgument("lost_found_vlm_service", default_value="/lost_found/vlm_check"),
        DeclareLaunchArgument("visual_question_service", default_value="/lost_found/visual_question"),
        DeclareLaunchArgument("telegram_reply_service", default_value="/telegram/reply"),
        DeclareLaunchArgument("telegram_visual_search_service", default_value="/telegram/visual_search"),
        DeclareLaunchArgument("capture_image_service", default_value="/camera/capture"),
        DeclareLaunchArgument("vlm_query_service", default_value="/vlm/query"),
        DeclareLaunchArgument("default_chat_id", default_value="8725865338"),
        DeclareLaunchArgument("ollama_base_url", default_value="http://127.0.0.1:11434"),
        DeclareLaunchArgument("use_openai_vlm", default_value="true"),
        DeclareLaunchArgument("vlm_model", default_value="qwen3.5:9b"),
        DeclareLaunchArgument("use_openai_chat", default_value="true"),
        DeclareLaunchArgument("openai_api_url", default_value="https://api.openai.com/v1/responses"),
        DeclareLaunchArgument("openai_api_key_env", default_value="OPENAI_API_KEY"),
        DeclareLaunchArgument("chat_model", default_value="gpt-5.5"),
        DeclareLaunchArgument("found_confidence_threshold", default_value="0.65"),
        DeclareLaunchArgument("use_mock_location_image", default_value="true"),
        DeclareLaunchArgument("mock_image_path", default_value=""),
        DeclareLaunchArgument("start_vlm_query_service", default_value="true"),
        DeclareLaunchArgument("start_camera_capture_service", default_value="true"),
    ]

    vlm_query_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("vlm_service"),
                "launch",
                "vlm_query_service.launch.py",
            ])
        ),
        condition=IfCondition(LaunchConfiguration("start_vlm_query_service")),
        launch_arguments={
            "service_name": LaunchConfiguration("vlm_query_service"),
            "ollama_base_url": LaunchConfiguration("ollama_base_url"),
            "use_openai_vlm": LaunchConfiguration("use_openai_vlm"),
            "openai_api_url": LaunchConfiguration("openai_api_url"),
            "openai_api_key_env": LaunchConfiguration("openai_api_key_env"),
            "openai_model": LaunchConfiguration("chat_model"),
            "vlm_model": LaunchConfiguration("vlm_model"),
            "default_camera_name": LaunchConfiguration("camera_name"),
            "default_camera_topic": LaunchConfiguration("camera_topic"),
        }.items(),
    )

    lost_found_vlm_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("vlm_service"),
                "launch",
                "lost_found_vlm_check.launch.py",
            ])
        ),
        launch_arguments={
            "service_name": LaunchConfiguration("lost_found_vlm_service"),
            "visual_question_service_name": LaunchConfiguration("visual_question_service"),
            "vlm_query_service": LaunchConfiguration("vlm_query_service"),
            "capture_image_service": LaunchConfiguration("capture_image_service"),
            "default_camera_name": LaunchConfiguration("camera_name"),
            "ollama_base_url": LaunchConfiguration("ollama_base_url"),
            "use_openai_vlm": LaunchConfiguration("use_openai_vlm"),
            "openai_api_url": LaunchConfiguration("openai_api_url"),
            "openai_api_key_env": LaunchConfiguration("openai_api_key_env"),
            "openai_model": LaunchConfiguration("chat_model"),
            "vlm_model": LaunchConfiguration("vlm_model"),
            "default_confidence_threshold": LaunchConfiguration("found_confidence_threshold"),
        }.items(),
    )

    camera_capture_node = Node(
        package="vlm_service",
        executable="camera_snapshot_service_node",
        name="camera_snapshot_service_node",
        output="screen",
        emulate_tty=True,
        condition=IfCondition(LaunchConfiguration("start_camera_capture_service")),
        parameters=[
            {
                "service_name": LaunchConfiguration("capture_image_service"),
                "default_camera_name": LaunchConfiguration("camera_name"),
                "default_camera_topic": LaunchConfiguration("camera_topic"),
            }
        ],
    )

    mock_location_image_node = Node(
        package="task_state_machine",
        executable="mock_location_image_service_node",
        name="mock_location_image_service_node",
        output="screen",
        emulate_tty=True,
        condition=IfCondition(LaunchConfiguration("use_mock_location_image")),
        parameters=[
            {
                "service_name": LaunchConfiguration("location_image_service"),
                "image_path": LaunchConfiguration("mock_image_path"),
                "camera_name": LaunchConfiguration("camera_name"),
                "success": True,
            }
        ],
    )

    telegram_node = Node(
        package="task_state_machine",
        executable="telegram_lost_found_node",
        name="telegram_lost_found_node",
        output="screen",
        emulate_tty=True,
        parameters=[
            {
                "bot_token": LaunchConfiguration("bot_token"),
                "bot_token_file": LaunchConfiguration("bot_token_file"),
                "allowed_user_ids_csv": LaunchConfiguration("allowed_user_ids"),
                "lost_found_vlm_service": LaunchConfiguration("lost_found_vlm_service"),
                "visual_question_service": LaunchConfiguration("visual_question_service"),
                "telegram_reply_service": LaunchConfiguration("telegram_reply_service"),
                "telegram_visual_search_service": LaunchConfiguration("telegram_visual_search_service"),
                "location_image_service": LaunchConfiguration("location_image_service"),
                "capture_image_service": LaunchConfiguration("capture_image_service"),
                "default_camera_name": LaunchConfiguration("camera_name"),
                "navigation_action_name": LaunchConfiguration("navigation_action_name"),
                "arm_pose_service": LaunchConfiguration("arm_pose_service"),
                "arm_detect_pose_name": LaunchConfiguration("arm_detect_pose_name"),
                "arm_zero_pose_name": LaunchConfiguration("arm_zero_pose_name"),
                "demo_mode": LaunchConfiguration("demo_mode"),
                "sim_mode": LaunchConfiguration("sim_mode"),
                "demo_navigation_delay_sec": LaunchConfiguration("demo_navigation_delay_sec"),
                "location_pose_map_json": LaunchConfiguration("location_pose_map_json"),
                "default_chat_id": LaunchConfiguration("default_chat_id"),
                "found_confidence_threshold": LaunchConfiguration("found_confidence_threshold"),
                "ollama_base_url": LaunchConfiguration("ollama_base_url"),
                "openai_api_url": LaunchConfiguration("openai_api_url"),
                "openai_api_key_env": LaunchConfiguration("openai_api_key_env"),
                "use_openai_chat": LaunchConfiguration("use_openai_chat"),
                "chat_model": LaunchConfiguration("chat_model"),
            }
        ],
    )

    return LaunchDescription(
        args
        + [
            vlm_query_launch,
            camera_capture_node,
            lost_found_vlm_launch,
            mock_location_image_node,
            telegram_node,
        ]
    )

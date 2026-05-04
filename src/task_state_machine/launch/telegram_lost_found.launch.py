from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    args = [
        DeclareLaunchArgument("bot_token", default_value=""),
        DeclareLaunchArgument("bot_token_file", default_value=""),
        DeclareLaunchArgument("allowed_user_ids", default_value=""),
        DeclareLaunchArgument("lost_found_vlm_service", default_value="/lost_found/vlm_check"),
        DeclareLaunchArgument("visual_question_service", default_value="/lost_found/visual_question"),
        DeclareLaunchArgument("telegram_reply_service", default_value="/telegram/reply"),
        DeclareLaunchArgument("telegram_visual_search_service", default_value="/telegram/visual_search"),
        DeclareLaunchArgument("location_image_service", default_value="/lost_found/location_image"),
        DeclareLaunchArgument("capture_image_service", default_value="/camera/capture"),
        DeclareLaunchArgument("default_camera_name", default_value="gripper_camera"),
        DeclareLaunchArgument("navigation_action_name", default_value="/navigate_to_pose"),
        DeclareLaunchArgument("arm_pose_service", default_value="/arm_pose"),
        DeclareLaunchArgument("arm_detect_pose_name", default_value="detect"),
        DeclareLaunchArgument("arm_zero_pose_name", default_value="zero"),
        DeclareLaunchArgument("demo_mode", default_value="true"),
        DeclareLaunchArgument("sim_mode", default_value="false"),
        DeclareLaunchArgument("demo_navigation_delay_sec", default_value="5.0"),
        DeclareLaunchArgument("location_pose_map_json", default_value=""),
        DeclareLaunchArgument("default_chat_id", default_value=""),
        DeclareLaunchArgument("service_wait_timeout_sec", default_value="5.0"),
        DeclareLaunchArgument("location_timeout_sec", default_value="120.0"),
        DeclareLaunchArgument("vlm_timeout_sec", default_value="75.0"),
        DeclareLaunchArgument("found_confidence_threshold", default_value="0.65"),
        DeclareLaunchArgument("telegram_poll_timeout_sec", default_value="20.0"),
        DeclareLaunchArgument("ollama_base_url", default_value="http://127.0.0.1:11434"),
        DeclareLaunchArgument("chat_model", default_value="qwen3.5:9b"),
    ]

    node = Node(
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
                "default_camera_name": LaunchConfiguration("default_camera_name"),
                "navigation_action_name": LaunchConfiguration("navigation_action_name"),
                "arm_pose_service": LaunchConfiguration("arm_pose_service"),
                "arm_detect_pose_name": LaunchConfiguration("arm_detect_pose_name"),
                "arm_zero_pose_name": LaunchConfiguration("arm_zero_pose_name"),
                "demo_mode": LaunchConfiguration("demo_mode"),
                "sim_mode": LaunchConfiguration("sim_mode"),
                "demo_navigation_delay_sec": LaunchConfiguration("demo_navigation_delay_sec"),
                "location_pose_map_json": LaunchConfiguration("location_pose_map_json"),
                "default_chat_id": LaunchConfiguration("default_chat_id"),
                "service_wait_timeout_sec": LaunchConfiguration("service_wait_timeout_sec"),
                "location_timeout_sec": LaunchConfiguration("location_timeout_sec"),
                "vlm_timeout_sec": LaunchConfiguration("vlm_timeout_sec"),
                "found_confidence_threshold": LaunchConfiguration("found_confidence_threshold"),
                "telegram_poll_timeout_sec": LaunchConfiguration("telegram_poll_timeout_sec"),
                "ollama_base_url": LaunchConfiguration("ollama_base_url"),
                "chat_model": LaunchConfiguration("chat_model"),
            }
        ],
    )

    return LaunchDescription(args + [node])

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
        DeclareLaunchArgument("capture_image_service", default_value="/camera/capture"),
        DeclareLaunchArgument("vlm_query_service", default_value="/vlm/query"),
        DeclareLaunchArgument("food_drink_sort_service", default_value="/food_drink/sort"),
        DeclareLaunchArgument("yoloe_detect_service", default_value="/yoloe/detect_prompt"),
        DeclareLaunchArgument("yoloe_prompt_text", default_value="can_coffee"),
        DeclareLaunchArgument("yoloe_save_image", default_value="true"),
        DeclareLaunchArgument("yoloe_detection_required", default_value="true"),
        DeclareLaunchArgument("yoloe_seg4_service_name", default_value="/yoloe/detect_prompt"),
        DeclareLaunchArgument("yoloe_seg4_model_path", default_value="/home/usern/Kevin_yolo/best_seg4.pt"),
        DeclareLaunchArgument("yoloe_seg4_save_dir", default_value="/home/usern/robocup_ws/yoloe_out"),
        DeclareLaunchArgument("navigation_action_name", default_value="/navigate_to_pose"),
        DeclareLaunchArgument("arm_pose_service", default_value="/arm_pose"),
        DeclareLaunchArgument("arm_center_pose_name", default_value="center"),
        DeclareLaunchArgument("demo_mode", default_value="true"),
        DeclareLaunchArgument("sim_mode", default_value="false"),
        DeclareLaunchArgument("demo_navigation_delay_sec", default_value="5.0"),
        DeclareLaunchArgument("location_pose_map_json", default_value=""),
        DeclareLaunchArgument("speak_action_name", default_value="/coqui_tts/speak"),
        DeclareLaunchArgument("default_chat_id", default_value="8725865338"),
        DeclareLaunchArgument("ollama_base_url", default_value="http://127.0.0.1:11434"),
        DeclareLaunchArgument("use_openai_vlm", default_value="true"),
        DeclareLaunchArgument("openai_api_url", default_value="https://api.openai.com/v1/responses"),
        DeclareLaunchArgument("openai_api_key_env", default_value="OPENAI_API_KEY"),
        DeclareLaunchArgument("openai_model", default_value="gpt-5.5"),
        DeclareLaunchArgument("vlm_model", default_value="qwen3.5:9b"),
        DeclareLaunchArgument("start_vlm_query_service", default_value="true"),
        DeclareLaunchArgument("start_camera_capture_service", default_value="true"),
    ]

    vlm_query_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("vlm_service"), "launch", "vlm_query_service.launch.py"]
            )
        ),
        condition=IfCondition(LaunchConfiguration("start_vlm_query_service")),
        launch_arguments={
            "service_name": LaunchConfiguration("vlm_query_service"),
            "ollama_base_url": LaunchConfiguration("ollama_base_url"),
            "use_openai_vlm": LaunchConfiguration("use_openai_vlm"),
            "openai_api_url": LaunchConfiguration("openai_api_url"),
            "openai_api_key_env": LaunchConfiguration("openai_api_key_env"),
            "openai_model": LaunchConfiguration("openai_model"),
            "vlm_model": LaunchConfiguration("vlm_model"),
            "default_camera_name": LaunchConfiguration("camera_name"),
            "default_camera_topic": LaunchConfiguration("camera_topic"),
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

    yoloe_seg4_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("yoloe_detection_service"), "launch", "yoloe_seg4_detection_service.launch.py"]
            )
        ),
        launch_arguments={
            "service_name": LaunchConfiguration("yoloe_seg4_service_name"),
            "model_path": LaunchConfiguration("yoloe_seg4_model_path"),
            "bag_model_path": LaunchConfiguration("yoloe_seg4_model_path"),
            "default_camera_name": LaunchConfiguration("camera_name"),
            "camera0_color_topic": LaunchConfiguration("camera_topic"),
            "save_dir": LaunchConfiguration("yoloe_seg4_save_dir"),
            "always_save_image": "true",
        }.items(),
    )

    food_drink_sort_node = Node(
        package="vlm_service",
        executable="food_drink_sort_service_node",
        name="food_drink_sort_service_node",
        output="screen",
        emulate_tty=True,
        parameters=[
            {
                "service_name": LaunchConfiguration("food_drink_sort_service"),
                "capture_image_service": LaunchConfiguration("capture_image_service"),
                "vlm_query_service": LaunchConfiguration("vlm_query_service"),
                "default_camera_name": LaunchConfiguration("camera_name"),
            }
        ],
    )

    telegram_node = Node(
        package="task_state_machine",
        executable="telegram_food_drink_state_machine_node",
        name="telegram_food_drink_state_machine_node",
        output="screen",
        emulate_tty=True,
        parameters=[
            {
                "bot_token": LaunchConfiguration("bot_token"),
                "bot_token_file": LaunchConfiguration("bot_token_file"),
                "allowed_user_ids_csv": LaunchConfiguration("allowed_user_ids"),
                "default_camera_name": LaunchConfiguration("camera_name"),
                "default_chat_id": LaunchConfiguration("default_chat_id"),
                "food_drink_sort_service": LaunchConfiguration("food_drink_sort_service"),
                "yoloe_detect_service": LaunchConfiguration("yoloe_detect_service"),
                "yoloe_prompt_text": LaunchConfiguration("yoloe_prompt_text"),
                "yoloe_save_image": LaunchConfiguration("yoloe_save_image"),
                "yoloe_detection_required": LaunchConfiguration("yoloe_detection_required"),
                "navigation_action_name": LaunchConfiguration("navigation_action_name"),
                "arm_pose_service": LaunchConfiguration("arm_pose_service"),
                "arm_center_pose_name": LaunchConfiguration("arm_center_pose_name"),
                "demo_mode": LaunchConfiguration("demo_mode"),
                "sim_mode": LaunchConfiguration("sim_mode"),
                "demo_navigation_delay_sec": LaunchConfiguration("demo_navigation_delay_sec"),
                "location_pose_map_json": LaunchConfiguration("location_pose_map_json"),
                "speak_action_name": LaunchConfiguration("speak_action_name"),
            }
        ],
    )

    return LaunchDescription(
        args
        + [
            vlm_query_launch,
            camera_capture_node,
            yoloe_seg4_launch,
            food_drink_sort_node,
            telegram_node,
        ]
    )

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def _include(package_name: str, launch_file: str, *, condition, launch_arguments=None):
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare(package_name), "launch", launch_file])
        ),
        condition=condition,
        launch_arguments=(launch_arguments or {}).items(),
    )


def generate_launch_description() -> LaunchDescription:
    default_camera_name = LaunchConfiguration("default_camera_name")
    color_topic = LaunchConfiguration("color_topic")
    depth_topic = LaunchConfiguration("depth_topic")
    camera_info_topic = LaunchConfiguration("camera_info_topic")
    camera_link_frame = LaunchConfiguration("camera_link_frame")

    ollama_base_url = LaunchConfiguration("ollama_base_url")
    vlm_model = LaunchConfiguration("vlm_model")
    yolo_model_path = LaunchConfiguration("yolo_model_path")
    yolo_bag_model_path = LaunchConfiguration("yolo_bag_model_path")
    yolo_device = LaunchConfiguration("yolo_device")
    yolo_python_site_packages = LaunchConfiguration("yolo_python_site_packages")

    launch_voice = LaunchConfiguration("launch_voice")
    launch_nanobot_agent = LaunchConfiguration("launch_nanobot_agent")
    launch_vlm = LaunchConfiguration("launch_vlm")
    launch_yolo = LaunchConfiguration("launch_yolo")
    launch_pointed_yolo = LaunchConfiguration("launch_pointed_yolo")
    launch_vlm_pointed_yolo = LaunchConfiguration("launch_vlm_pointed_yolo")
    launch_people_tracking = LaunchConfiguration("launch_people_tracking")

    voice_stack_launch = _include(
        "coqui_tts_service",
        "voice_interaction_stack.launch.py",
        condition=IfCondition(launch_voice),
        launch_arguments={
            "face_fullscreen": LaunchConfiguration("face_fullscreen"),
        },
    )

    nanobot_voice_agent_launch = _include(
        "coqui_tts_service",
        "nanobot_voice_agent.launch.py",
        condition=IfCondition(launch_nanobot_agent),
        launch_arguments={
            "awake_topic": "/awake",
            "awake_greeting_done_topic": "/awake_greeting_done",
            "get_command_service": "/get_command",
            "robot_status_service": "/robot_status",
            "speak_action_name": "/coqui_tts/speak",
            "nanobot_executable": LaunchConfiguration("nanobot_executable"),
            "nanobot_workdir": LaunchConfiguration("nanobot_workdir"),
            "nanobot_config_path": LaunchConfiguration("nanobot_config_path"),
            "nanobot_workspace": LaunchConfiguration("nanobot_workspace"),
            "nanobot_logs": LaunchConfiguration("nanobot_logs"),
        },
    )

    vlm_query_launch = _include(
        "vlm_service",
        "vlm_query_service.launch.py",
        condition=IfCondition(launch_vlm),
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
        },
    )

    yolo_detection_launch = _include(
        "yoloe_detection_service",
        "yoloe_detection_service.launch.py",
        condition=IfCondition(launch_yolo),
        launch_arguments={
            "service_name": "/yoloe/detect_prompt",
            "model_path": yolo_model_path,
            "bag_model_path": yolo_bag_model_path,
            "device": yolo_device,
            "default_camera_name": default_camera_name,
            "color_topic": color_topic,
            "depth_topic": depth_topic,
            "camera_info_topic": camera_info_topic,
            "camera_link_frame": camera_link_frame,
            "python_site_packages": yolo_python_site_packages,
        },
    )

    pointed_yolo_launch = _include(
        "yoloe_detection_service",
        "yoloe_pointed_detection_service.launch.py",
        condition=IfCondition(launch_pointed_yolo),
        launch_arguments={
            "service_name": "/yoloe/detect_pointed_prompt",
            "model_path": yolo_bag_model_path,
            "device": yolo_device,
            "color_topic": color_topic,
            "depth_topic": depth_topic,
            "camera_info_topic": camera_info_topic,
            "camera_link_frame": camera_link_frame,
            "show_ui": LaunchConfiguration("show_yolo_ui"),
        },
    )

    vlm_pointed_yolo_launch = _include(
        "yoloe_detection_service",
        "yoloe_vlm_pointed_detection_service.launch.py",
        condition=IfCondition(launch_vlm_pointed_yolo),
        launch_arguments={
            "service_name": "/yoloe/detect_pointed_prompt_vlm",
            "model_path": yolo_model_path,
            "device": yolo_device,
            "color_topic": color_topic,
            "depth_topic": depth_topic,
            "camera_info_topic": camera_info_topic,
            "camera_link_frame": camera_link_frame,
            "show_ui": LaunchConfiguration("show_yolo_ui"),
            "vlm_model": vlm_model,
            "ollama_base_url": ollama_base_url,
        },
    )

    people_tracking_launch = _include(
        "deepsort_people_follow",
        "deepsort_people_follow.launch.py",
        condition=IfCondition(launch_people_tracking),
        launch_arguments={
            "tracking_service_name": "/yoloe/set_tracking",
            "color_topic": color_topic,
            "depth_topic": depth_topic,
            "camera_info_topic": camera_info_topic,
            "camera_link_frame": camera_link_frame,
            "model_path": LaunchConfiguration("people_tracking_model_path"),
            "device": LaunchConfiguration("people_tracking_device"),
            "enable_ui": LaunchConfiguration("show_people_tracking_ui"),
            "launch_nav2_bridge": LaunchConfiguration("launch_people_follow_nav2_bridge"),
            "nav2_bridge_enabled": "false",
        },
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("default_camera_name", default_value="camera0"),
            DeclareLaunchArgument("color_topic", default_value="/camera0/color/image_raw"),
            DeclareLaunchArgument(
                "depth_topic", default_value="/camera0/realsense_splitter_node/output/depth"
            ),
            DeclareLaunchArgument("camera_info_topic", default_value="/camera0/color/camera_info"),
            DeclareLaunchArgument("camera_link_frame", default_value="camera0_link"),
            DeclareLaunchArgument("ollama_base_url", default_value="http://127.0.0.1:11434"),
            DeclareLaunchArgument("vlm_model", default_value="qwen3.5:9b"),
            DeclareLaunchArgument("yolo_model_path", default_value="/home/usern/yoloe-26l-seg.pt"),
            DeclareLaunchArgument(
                "yolo_bag_model_path",
                default_value="/home/usern/Kevin_yolo/best_latest.pt",
            ),
            DeclareLaunchArgument("yolo_device", default_value="auto"),
            DeclareLaunchArgument(
                "yolo_python_site_packages",
                default_value="/home/usern/coqui-venv/lib/python3.10/site-packages",
            ),
            DeclareLaunchArgument(
                "people_tracking_model_path",
                default_value="/home/usern/robocup_ws/yolo11n.pt",
            ),
            DeclareLaunchArgument("people_tracking_device", default_value="cuda:0"),
            DeclareLaunchArgument("nanobot_executable", default_value="/home/usern/nanobot/.venv/bin/nanobot"),
            DeclareLaunchArgument("nanobot_workdir", default_value="/home/usern/nanobot"),
            DeclareLaunchArgument("nanobot_config_path", default_value=""),
            DeclareLaunchArgument("nanobot_workspace", default_value=""),
            DeclareLaunchArgument("nanobot_logs", default_value="false"),
            DeclareLaunchArgument("face_fullscreen", default_value="false"),
            DeclareLaunchArgument("show_yolo_ui", default_value="false"),
            DeclareLaunchArgument("show_people_tracking_ui", default_value="false"),
            DeclareLaunchArgument("launch_voice", default_value="true"),
            DeclareLaunchArgument("launch_nanobot_agent", default_value="true"),
            DeclareLaunchArgument("launch_vlm", default_value="true"),
            DeclareLaunchArgument("launch_yolo", default_value="true"),
            DeclareLaunchArgument("launch_pointed_yolo", default_value="false"),
            DeclareLaunchArgument("launch_vlm_pointed_yolo", default_value="false"),
            DeclareLaunchArgument("launch_people_tracking", default_value="false"),
            DeclareLaunchArgument("launch_people_follow_nav2_bridge", default_value="false"),
            voice_stack_launch,
            nanobot_voice_agent_launch,
            vlm_query_launch,
            yolo_detection_launch,
            pointed_yolo_launch,
            vlm_pointed_yolo_launch,
            people_tracking_launch,
        ]
    )

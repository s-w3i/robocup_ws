from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    color_topic = LaunchConfiguration("color_topic")
    depth_topic = LaunchConfiguration("depth_topic")
    camera_info_topic = LaunchConfiguration("camera_info_topic")
    camera_link_frame = LaunchConfiguration("camera_link_frame")
    default_camera_name = LaunchConfiguration("default_camera_name")
    vlm_model = LaunchConfiguration("vlm_model")
    ollama_base_url = LaunchConfiguration("ollama_base_url")
    yolo_model_path = LaunchConfiguration("yolo_model_path")
    yolo_python_site_packages = LaunchConfiguration("yolo_python_site_packages")
    face_fullscreen = LaunchConfiguration("face_fullscreen")
    launch_whisper = LaunchConfiguration("launch_whisper")
    debug_text_input_mode = LaunchConfiguration("debug_text_input_mode")
    debug_text_input_prompt = LaunchConfiguration("debug_text_input_prompt")
    launch_state_machine = LaunchConfiguration("launch_state_machine")
    launch_bt_monitor = LaunchConfiguration("launch_bt_monitor")
    bt_use_ui = LaunchConfiguration("bt_use_ui")

    robot_status_node = Node(
        package="coqui_tts_service",
        executable="robot_status_node",
        name="robot_status_node",
        output="screen",
    )

    coqui_talking_face_action_node = Node(
        package="coqui_tts_service",
        executable="coqui_talking_face_action_node",
        name="coqui_talking_face_action_node",
        output="screen",
        parameters=[
            {
                "face_fullscreen": face_fullscreen,
            }
        ],
    )

    whisper_command_node = Node(
        package="coqui_tts_service",
        executable="whisper_command_node",
        name="whisper_command_node",
        output="screen",
        condition=IfCondition(launch_whisper),
    )

    vlm_query_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("vlm_service"), "launch", "vlm_query_service.launch.py"]
            )
        ),
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
        }.items(),
    )

    yolo_detection_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("yoloe_detection_service"), "launch", "yoloe_detection_service.launch.py"]
            )
        ),
        launch_arguments={
            "service_name": "/yoloe/detect_prompt",
            "model_path": yolo_model_path,
            "color_topic": color_topic,
            "depth_topic": depth_topic,
            "camera_info_topic": camera_info_topic,
            "camera_link_frame": camera_link_frame,
            "python_site_packages": yolo_python_site_packages,
        }.items(),
    )

    ask_name_and_drink_node = Node(
        package="vlm_service",
        executable="ask_name_and_drink_action_node",
        name="ask_name_and_drink_action_node",
        output="screen",
        emulate_tty=True,
        parameters=[
            {
                "action_name": "/ask_name_and_drink",
                "vlm_query_service": "/vlm/query",
                "get_command_service": "/get_command",
                "speak_action_name": "/coqui_tts/speak",
                "debug_text_input_mode": debug_text_input_mode,
                "debug_text_input_prompt": debug_text_input_prompt,
            }
        ],
    )

    describe_human_node = Node(
        package="vlm_service",
        executable="describe_human_action_node",
        name="describe_human_action_node",
        output="screen",
        emulate_tty=True,
        parameters=[
            {
                "action_name": "/describe_human",
                "vlm_query_service": "/vlm/query",
                "speak_action_name": "/coqui_tts/speak",
                "default_camera_name": default_camera_name,
            }
        ],
    )

    receptionist_state_machine_node = Node(
        package="task_state_machine",
        executable="receptionist_state_machine_node",
        name="receptionist_state_machine_node",
        output="screen",
        condition=IfCondition(launch_state_machine),
    )

    bt_monitor_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("bt_tree_view"), "launch", "bt_monitor.launch.py"]
            )
        ),
        condition=IfCondition(launch_bt_monitor),
        launch_arguments={
            "snapshot_topic": "/bt_tree/snapshots",
            "use_ui": bt_use_ui,
        }.items(),
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
            DeclareLaunchArgument("vlm_model", default_value="qwen3.5:9b"),
            DeclareLaunchArgument("ollama_base_url", default_value="http://127.0.0.1:11434"),
            DeclareLaunchArgument("yolo_model_path", default_value="/home/usern/yoloe-26l-seg.pt"),
            DeclareLaunchArgument(
                "yolo_python_site_packages",
                default_value="/home/usern/coqui-venv/lib/python3.10/site-packages",
            ),
            DeclareLaunchArgument("face_fullscreen", default_value="false"),
            DeclareLaunchArgument("launch_whisper", default_value="false"),
            DeclareLaunchArgument("debug_text_input_mode", default_value="true"),
            DeclareLaunchArgument("debug_text_input_prompt", default_value="guest"),
            DeclareLaunchArgument("launch_state_machine", default_value="false"),
            DeclareLaunchArgument("launch_bt_monitor", default_value="false"),
            DeclareLaunchArgument("bt_use_ui", default_value="false"),
            robot_status_node,
            coqui_talking_face_action_node,
            whisper_command_node,
            vlm_query_launch,
            yolo_detection_launch,
            ask_name_and_drink_node,
            describe_human_node,
            receptionist_state_machine_node,
            bt_monitor_launch,
        ]
    )

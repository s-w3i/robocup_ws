from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    face_fullscreen = LaunchConfiguration("face_fullscreen")

    laser_follow_rviz = LaunchConfiguration("laser_follow_rviz")

    pointed_color_topic = LaunchConfiguration("pointed_color_topic")
    pointed_depth_topic = LaunchConfiguration("pointed_depth_topic")
    pointed_camera_info_topic = LaunchConfiguration("pointed_camera_info_topic")
    pointed_camera_link_frame = LaunchConfiguration("pointed_camera_link_frame")
    pointed_model_path = LaunchConfiguration("pointed_model_path")
    pointed_device = LaunchConfiguration("pointed_device")
    pointed_show_ui = LaunchConfiguration("pointed_show_ui")
    pointed_save_dir = LaunchConfiguration("pointed_save_dir")
    gripper_detection_model_path = LaunchConfiguration("gripper_detection_model_path")
    gripper_detection_bag_model_path = LaunchConfiguration("gripper_detection_bag_model_path")
    gripper_detection_device = LaunchConfiguration("gripper_detection_device")
    gripper_detection_default_camera_name = LaunchConfiguration(
        "gripper_detection_default_camera_name"
    )
    gripper_color_topic = LaunchConfiguration("gripper_color_topic")
    gripper_depth_topic = LaunchConfiguration("gripper_depth_topic")
    gripper_camera_info_topic = LaunchConfiguration("gripper_camera_info_topic")
    gripper_camera_link_frame = LaunchConfiguration("gripper_camera_link_frame")

    voice_interaction_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("coqui_tts_service"), "launch", "voice_interaction_stack.launch.py"]
            )
        ),
        launch_arguments={
            "face_fullscreen": face_fullscreen,
        }.items(),
    )

    laser_people_follow_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("simple_follower_ros2"), "launch", "laser_follower.launch.py"]
            )
        ),
        launch_arguments={
            "enabled": "false",
            "vision_validation_enabled": "true",
            "vision_service_name": "/yoloe/detect_prompt",
            "vision_camera_name": "camera0",
            "vision_prompt": "person",
            "rviz": laser_follow_rviz,
        }.items(),
    )

    pointed_detection_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("yoloe_detection_service"), "launch", "yoloe_pointed_detection_service.launch.py"]
            )
        ),
        launch_arguments={
            "service_name": "/yoloe/detect_pointed_prompt",
            "model_path": pointed_model_path,
            "device": pointed_device,
            "color_topic": pointed_color_topic,
            "depth_topic": pointed_depth_topic,
            "camera_info_topic": pointed_camera_info_topic,
            "camera_link_frame": pointed_camera_link_frame,
            "show_ui": pointed_show_ui,
            "save_dir": pointed_save_dir,
        }.items(),
    )

    gripper_detection_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("yoloe_detection_service"), "launch", "yoloe_detection_service.launch.py"]
            )
        ),
        launch_arguments={
            "service_name": "/yoloe/detect_prompt",
            "model_path": gripper_detection_model_path,
            "bag_model_path": gripper_detection_bag_model_path,
            "device": gripper_detection_device,
            "default_camera_name": gripper_detection_default_camera_name,
            "color_topic": pointed_color_topic,
            "depth_topic": pointed_depth_topic,
            "camera_info_topic": pointed_camera_info_topic,
            "camera_link_frame": pointed_camera_link_frame,
            "camera_color_topic": gripper_color_topic,
            "camera_depth_topic": gripper_depth_topic,
            "camera_camera_info_topic": gripper_camera_info_topic,
            "camera_camera_link_frame": gripper_camera_link_frame,
            "save_dir": pointed_save_dir,
        }.items(),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("face_fullscreen", default_value="false"),
            DeclareLaunchArgument("laser_follow_rviz", default_value="false"),
            DeclareLaunchArgument("pointed_color_topic", default_value="/camera0/color/image_raw"),
            DeclareLaunchArgument("pointed_depth_topic", default_value="/camera0/depth/image_rect_raw"),
            DeclareLaunchArgument(
                "pointed_camera_info_topic",
                default_value="/camera0/color/camera_info",
            ),
            DeclareLaunchArgument(
                "pointed_camera_link_frame",
                default_value="camera0_color_optical_frame",
            ),
            DeclareLaunchArgument(
                "pointed_model_path",
                default_value="/home/usern/Kevin_yolo/best_latest.pt",
            ),
            DeclareLaunchArgument("pointed_device", default_value="auto"),
            DeclareLaunchArgument("pointed_show_ui", default_value="false"),
            DeclareLaunchArgument(
                "pointed_save_dir",
                default_value="/home/usern/robocup_ws/yoloe_out",
            ),
            DeclareLaunchArgument(
                "gripper_detection_model_path",
                default_value="/home/usern/yoloe-26l-seg.pt",
            ),
            DeclareLaunchArgument(
                "gripper_detection_bag_model_path",
                default_value="/home/usern/Kevin_yolo/best_latest.pt",
            ),
            DeclareLaunchArgument("gripper_detection_device", default_value="auto"),
            DeclareLaunchArgument(
                "gripper_detection_default_camera_name",
                default_value="camera0",
            ),
            DeclareLaunchArgument(
                "gripper_color_topic",
                default_value="/gripper_camera/color/image_raw",
            ),
            DeclareLaunchArgument(
                "gripper_depth_topic",
                default_value="/gripper_camera/depth/image_raw",
            ),
            DeclareLaunchArgument(
                "gripper_camera_info_topic",
                default_value="/gripper_camera/color/camera_info",
            ),
            DeclareLaunchArgument(
                "gripper_camera_link_frame",
                default_value="gripper_camera_link",
            ),
            voice_interaction_launch,
            laser_people_follow_launch,
            pointed_detection_launch,
            gripper_detection_launch,
        ]
    )

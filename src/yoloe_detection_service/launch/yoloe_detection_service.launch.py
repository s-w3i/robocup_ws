from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    args = [
        DeclareLaunchArgument("model_path", default_value="/home/usern/yoloe-26l-seg.pt"),
        DeclareLaunchArgument("device", default_value="auto"),
        DeclareLaunchArgument("service_name", default_value="/yoloe/detect_prompt"),
        DeclareLaunchArgument("color_topic", default_value="/camera/color/image_raw"),
        DeclareLaunchArgument("depth_topic", default_value="/camera/depth/image_raw"),
        DeclareLaunchArgument("camera_info_topic", default_value="/camera/color/camera_info"),
        DeclareLaunchArgument("camera_link_frame", default_value="camera_link"),
        DeclareLaunchArgument("save_dir", default_value="/home/usern/robocup_ws/yoloe_out"),
        DeclareLaunchArgument("always_save_image", default_value="false"),
    ]

    node = Node(
        package="yoloe_detection_service",
        executable="yoloe_detection_service_node",
        name="yoloe_detection_service_node",
        output="screen",
        parameters=[
            {
                "model_path": LaunchConfiguration("model_path"),
                "device": LaunchConfiguration("device"),
                "service_name": LaunchConfiguration("service_name"),
                "color_topic": LaunchConfiguration("color_topic"),
                "depth_topic": LaunchConfiguration("depth_topic"),
                "camera_info_topic": LaunchConfiguration("camera_info_topic"),
                "camera_link_frame": LaunchConfiguration("camera_link_frame"),
                "save_dir": LaunchConfiguration("save_dir"),
                "always_save_image": LaunchConfiguration("always_save_image"),
            }
        ],
    )

    return LaunchDescription(args + [node])

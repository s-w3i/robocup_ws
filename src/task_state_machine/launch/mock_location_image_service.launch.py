from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    args = [
        DeclareLaunchArgument("service_name", default_value="/lost_found/location_image"),
        DeclareLaunchArgument("image_path", default_value=""),
        DeclareLaunchArgument("camera_name", default_value="camera0"),
        DeclareLaunchArgument("success", default_value="true"),
    ]

    node = Node(
        package="task_state_machine",
        executable="mock_location_image_service_node",
        name="mock_location_image_service_node",
        output="screen",
        emulate_tty=True,
        parameters=[
            {
                "service_name": LaunchConfiguration("service_name"),
                "image_path": LaunchConfiguration("image_path"),
                "camera_name": LaunchConfiguration("camera_name"),
                "success": LaunchConfiguration("success"),
            }
        ],
    )

    return LaunchDescription(args + [node])

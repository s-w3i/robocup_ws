from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    sim = LaunchConfiguration("sim")
    default_camera_name = LaunchConfiguration("default_camera_name")

    return LaunchDescription(
        [
            DeclareLaunchArgument("sim", default_value="false"),
            DeclareLaunchArgument("default_camera_name", default_value="camera0"),
            Node(
                package="task_state_machine",
                executable="carry_my_luggage_state_machine_node",
                name="carry_my_luggage_state_machine_node",
                output="screen",
                additional_env={
                    "CARRY_MY_LUGGAGE_SIM": sim,
                    "YOLO_DETECTION_CAMERA_NAME": default_camera_name,
                },
            ),
        ]
    )

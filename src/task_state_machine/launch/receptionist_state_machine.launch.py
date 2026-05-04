from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    sim = LaunchConfiguration("sim")
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "sim",
                default_value=EnvironmentVariable("RECEPTIONIST_SIM", default_value="false"),
            ),
            Node(
                package="task_state_machine",
                executable="receptionist_state_machine_node",
                output="screen",
                additional_env={
                    "RECEPTIONIST_SIM": sim,
                },
            )
        ]
    )

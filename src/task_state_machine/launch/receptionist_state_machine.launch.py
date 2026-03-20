from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            Node(
                package="task_state_machine",
                executable="receptionist_state_machine_node",
                output="screen",
            )
        ]
    )

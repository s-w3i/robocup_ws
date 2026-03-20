from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    snapshot_topic = LaunchConfiguration("snapshot_topic")
    refresh_hz = LaunchConfiguration("refresh_hz")
    use_ui = LaunchConfiguration("use_ui")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "snapshot_topic",
                default_value="/bt_tree/snapshots",
                description="Topic carrying BT snapshot JSON strings.",
            ),
            DeclareLaunchArgument(
                "refresh_hz",
                default_value="2.0",
                description="Terminal refresh rate for the BT monitor.",
            ),
            DeclareLaunchArgument(
                "use_ui",
                default_value="false",
                description="Launch the PyQt UI instead of the terminal monitor.",
            ),
            Node(
                package="bt_tree_view",
                executable="bt_monitor_node",
                name="bt_monitor_node",
                output="screen",
                condition=UnlessCondition(use_ui),
                parameters=[
                    {
                        "snapshot_topic": snapshot_topic,
                        "refresh_hz": refresh_hz,
                    }
                ],
            ),
            Node(
                package="bt_tree_view",
                executable="bt_monitor_ui",
                name="bt_monitor_ui",
                output="screen",
                parameters=[
                    {
                        "snapshot_topic": snapshot_topic,
                    }
                ],
                condition=IfCondition(use_ui),
            ),
        ]
    )

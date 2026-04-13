from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    args = [
        DeclareLaunchArgument("enabled", default_value="false"),
        DeclareLaunchArgument("global_frame", default_value="map"),
        DeclareLaunchArgument("robot_base_frame", default_value="base_footprint"),
        DeclareLaunchArgument("tracked_frame", default_value="follow_target"),
        DeclareLaunchArgument("goal_update_topic", default_value="/goal_update"),
        DeclareLaunchArgument("navigate_to_pose_action", default_value="/navigate_to_pose"),
        DeclareLaunchArgument("enable_service_name", default_value="/people_follow_nav2/set_enabled"),
        DeclareLaunchArgument("goal_update_rate_hz", default_value="5.0"),
        DeclareLaunchArgument("transform_timeout_s", default_value="0.2"),
        DeclareLaunchArgument("target_lost_timeout_s", default_value="2.0"),
        DeclareLaunchArgument(
            "behavior_tree_path",
            default_value="/home/usern/robocup_ws/src/deepsort_people_follow/bt/follower_w_recovery.xml",
        ),
        DeclareLaunchArgument("behavior_tree_package", default_value="deepsort_people_follow"),
        DeclareLaunchArgument("behavior_tree_relative_path", default_value="bt/follower_w_recovery.xml"),
    ]

    node = Node(
        package="deepsort_people_follow",
        executable="people_follow_nav2_bridge",
        name="people_follow_nav2_bridge",
        output="screen",
        parameters=[
            {
                "enabled": LaunchConfiguration("enabled"),
                "global_frame": LaunchConfiguration("global_frame"),
                "robot_base_frame": LaunchConfiguration("robot_base_frame"),
                "tracked_frame": LaunchConfiguration("tracked_frame"),
                "goal_update_topic": LaunchConfiguration("goal_update_topic"),
                "navigate_to_pose_action": LaunchConfiguration("navigate_to_pose_action"),
                "enable_service_name": LaunchConfiguration("enable_service_name"),
                "goal_update_rate_hz": LaunchConfiguration("goal_update_rate_hz"),
                "transform_timeout_s": LaunchConfiguration("transform_timeout_s"),
                "target_lost_timeout_s": LaunchConfiguration("target_lost_timeout_s"),
                "behavior_tree_path": LaunchConfiguration("behavior_tree_path"),
                "behavior_tree_package": LaunchConfiguration("behavior_tree_package"),
                "behavior_tree_relative_path": LaunchConfiguration("behavior_tree_relative_path"),
            }
        ],
    )

    return LaunchDescription(args + [node])

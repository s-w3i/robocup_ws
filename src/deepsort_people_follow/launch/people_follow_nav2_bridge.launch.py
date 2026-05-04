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
        DeclareLaunchArgument("goal_command_mode", default_value="navigate_to_pose"),
        DeclareLaunchArgument("goal_pose_topic", default_value="/goal_pose"),
        DeclareLaunchArgument("goal_update_topic", default_value="/goal_update"),
        DeclareLaunchArgument("navigate_to_pose_action", default_value="/navigate_to_pose"),
        DeclareLaunchArgument("enable_service_name", default_value="/people_follow_nav2/set_enabled"),
        DeclareLaunchArgument("goal_update_rate_hz", default_value="10.0"),
        DeclareLaunchArgument("goal_republish_period_s", default_value="0.0"),
        DeclareLaunchArgument("min_goal_translation_delta_m", default_value="0.05"),
        DeclareLaunchArgument("min_goal_yaw_delta_rad", default_value="0.12"),
        DeclareLaunchArgument("follow_standoff_distance_m", default_value="0.7"),
        DeclareLaunchArgument("follow_distance_tolerance_m", default_value="0.10"),
        DeclareLaunchArgument("target_smoothing_alpha", default_value="0.65"),
        DeclareLaunchArgument("target_smoothing_reset_distance_m", default_value="0.40"),
        DeclareLaunchArgument("publish_hold_goal_on_disable", default_value="true"),
        DeclareLaunchArgument("transform_timeout_s", default_value="0.2"),
        DeclareLaunchArgument("target_lost_timeout_s", default_value="10.0"),
        DeclareLaunchArgument(
            "behavior_tree_path",
            default_value="/workspaces/isaac_ros-dev/src/wheeltec_robot_nav2/param/bt/follower_w_recovery.xml",
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
                "goal_command_mode": LaunchConfiguration("goal_command_mode"),
                "goal_pose_topic": LaunchConfiguration("goal_pose_topic"),
                "goal_update_topic": LaunchConfiguration("goal_update_topic"),
                "navigate_to_pose_action": LaunchConfiguration("navigate_to_pose_action"),
                "enable_service_name": LaunchConfiguration("enable_service_name"),
                "goal_update_rate_hz": LaunchConfiguration("goal_update_rate_hz"),
                "goal_republish_period_s": LaunchConfiguration("goal_republish_period_s"),
                "min_goal_translation_delta_m": LaunchConfiguration("min_goal_translation_delta_m"),
                "min_goal_yaw_delta_rad": LaunchConfiguration("min_goal_yaw_delta_rad"),
                "follow_standoff_distance_m": LaunchConfiguration("follow_standoff_distance_m"),
                "follow_distance_tolerance_m": LaunchConfiguration("follow_distance_tolerance_m"),
                "target_smoothing_alpha": LaunchConfiguration("target_smoothing_alpha"),
                "target_smoothing_reset_distance_m": LaunchConfiguration("target_smoothing_reset_distance_m"),
                "publish_hold_goal_on_disable": LaunchConfiguration("publish_hold_goal_on_disable"),
                "transform_timeout_s": LaunchConfiguration("transform_timeout_s"),
                "target_lost_timeout_s": LaunchConfiguration("target_lost_timeout_s"),
                "behavior_tree_path": LaunchConfiguration("behavior_tree_path"),
                "behavior_tree_package": LaunchConfiguration("behavior_tree_package"),
                "behavior_tree_relative_path": LaunchConfiguration("behavior_tree_relative_path"),
            }
        ],
    )

    return LaunchDescription(args + [node])

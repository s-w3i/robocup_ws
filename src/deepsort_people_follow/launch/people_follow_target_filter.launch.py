from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    args = [
        DeclareLaunchArgument("enabled", default_value="true"),
        DeclareLaunchArgument("global_frame", default_value="map"),
        DeclareLaunchArgument("source_frame", default_value="follow_target"),
        DeclareLaunchArgument("filtered_frame", default_value="follow_target_filtered"),
        DeclareLaunchArgument("filtered_pose_topic", default_value="/people/follow_target_filtered"),
        DeclareLaunchArgument("enable_service_name", default_value="/people_follow_target_filter/set_enabled"),
        DeclareLaunchArgument("publish_rate_hz", default_value="15.0"),
        DeclareLaunchArgument("transform_timeout_s", default_value="0.1"),
        DeclareLaunchArgument("target_lost_timeout_s", default_value="1.0"),
        DeclareLaunchArgument("prediction_horizon_s", default_value="0.25"),
        DeclareLaunchArgument("position_gain", default_value="0.65"),
        DeclareLaunchArgument("velocity_gain", default_value="0.20"),
        DeclareLaunchArgument("max_target_speed_mps", default_value="1.8"),
        DeclareLaunchArgument("reset_distance_m", default_value="1.2"),
        DeclareLaunchArgument("measurement_outlier_distance_m", default_value="1.0"),
        DeclareLaunchArgument("measurement_outlier_required_count", default_value="3"),
    ]

    node = Node(
        package="deepsort_people_follow",
        executable="people_follow_target_filter",
        name="people_follow_target_filter",
        output="screen",
        parameters=[
            {
                "enabled": LaunchConfiguration("enabled"),
                "global_frame": LaunchConfiguration("global_frame"),
                "source_frame": LaunchConfiguration("source_frame"),
                "filtered_frame": LaunchConfiguration("filtered_frame"),
                "filtered_pose_topic": LaunchConfiguration("filtered_pose_topic"),
                "enable_service_name": LaunchConfiguration("enable_service_name"),
                "publish_rate_hz": LaunchConfiguration("publish_rate_hz"),
                "transform_timeout_s": LaunchConfiguration("transform_timeout_s"),
                "target_lost_timeout_s": LaunchConfiguration("target_lost_timeout_s"),
                "prediction_horizon_s": LaunchConfiguration("prediction_horizon_s"),
                "position_gain": LaunchConfiguration("position_gain"),
                "velocity_gain": LaunchConfiguration("velocity_gain"),
                "max_target_speed_mps": LaunchConfiguration("max_target_speed_mps"),
                "reset_distance_m": LaunchConfiguration("reset_distance_m"),
                "measurement_outlier_distance_m": LaunchConfiguration("measurement_outlier_distance_m"),
                "measurement_outlier_required_count": LaunchConfiguration(
                    "measurement_outlier_required_count"
                ),
            }
        ],
    )

    return LaunchDescription(args + [node])

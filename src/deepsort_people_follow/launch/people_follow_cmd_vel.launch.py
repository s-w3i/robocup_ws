from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    args = [
        DeclareLaunchArgument("enabled", default_value="false"),
        DeclareLaunchArgument("robot_base_frame", default_value="base_footprint"),
        DeclareLaunchArgument("tracked_frame", default_value="follow_target"),
        DeclareLaunchArgument("cmd_vel_topic", default_value="/cmd_vel"),
        DeclareLaunchArgument("enable_service_name", default_value="/people_follow_cmd_vel/set_enabled"),
        DeclareLaunchArgument("control_rate_hz", default_value="15.0"),
        DeclareLaunchArgument("transform_timeout_s", default_value="0.1"),
        DeclareLaunchArgument("target_lost_timeout_s", default_value="0.75"),
        DeclareLaunchArgument("follow_standoff_distance_m", default_value="0.7"),
        DeclareLaunchArgument("follow_distance_tolerance_m", default_value="0.10"),
        DeclareLaunchArgument("bearing_deadband_rad", default_value="0.05"),
        DeclareLaunchArgument("rotate_in_place_min_angle_rad", default_value="0.45"),
        DeclareLaunchArgument("linear_heading_slowdown_angle_rad", default_value="0.80"),
        DeclareLaunchArgument("min_target_distance_m", default_value="0.20"),
        DeclareLaunchArgument("linear_kp", default_value="0.9"),
        DeclareLaunchArgument("angular_kp", default_value="1.8"),
        DeclareLaunchArgument("max_linear_speed_mps", default_value="0.35"),
        DeclareLaunchArgument("max_angular_speed_radps", default_value="1.2"),
        DeclareLaunchArgument("max_linear_accel_mps2", default_value="0.8"),
        DeclareLaunchArgument("max_angular_accel_radps2", default_value="2.5"),
    ]

    node = Node(
        package="deepsort_people_follow",
        executable="people_follow_cmd_vel",
        name="people_follow_cmd_vel",
        output="screen",
        parameters=[
            {
                "enabled": LaunchConfiguration("enabled"),
                "robot_base_frame": LaunchConfiguration("robot_base_frame"),
                "tracked_frame": LaunchConfiguration("tracked_frame"),
                "cmd_vel_topic": LaunchConfiguration("cmd_vel_topic"),
                "enable_service_name": LaunchConfiguration("enable_service_name"),
                "control_rate_hz": LaunchConfiguration("control_rate_hz"),
                "transform_timeout_s": LaunchConfiguration("transform_timeout_s"),
                "target_lost_timeout_s": LaunchConfiguration("target_lost_timeout_s"),
                "follow_standoff_distance_m": LaunchConfiguration("follow_standoff_distance_m"),
                "follow_distance_tolerance_m": LaunchConfiguration("follow_distance_tolerance_m"),
                "bearing_deadband_rad": LaunchConfiguration("bearing_deadband_rad"),
                "rotate_in_place_min_angle_rad": LaunchConfiguration("rotate_in_place_min_angle_rad"),
                "linear_heading_slowdown_angle_rad": LaunchConfiguration("linear_heading_slowdown_angle_rad"),
                "min_target_distance_m": LaunchConfiguration("min_target_distance_m"),
                "linear_kp": LaunchConfiguration("linear_kp"),
                "angular_kp": LaunchConfiguration("angular_kp"),
                "max_linear_speed_mps": LaunchConfiguration("max_linear_speed_mps"),
                "max_angular_speed_radps": LaunchConfiguration("max_angular_speed_radps"),
                "max_linear_accel_mps2": LaunchConfiguration("max_linear_accel_mps2"),
                "max_angular_accel_radps2": LaunchConfiguration("max_angular_accel_radps2"),
            }
        ],
    )

    return LaunchDescription(args + [node])

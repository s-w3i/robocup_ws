from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    package_share = FindPackageShare("deepsort_people_follow")

    tracker_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([package_share, "launch", "deepsort_people_follow.launch.py"])
        ),
        launch_arguments={
            "tracking_service_name": LaunchConfiguration("tracking_service_name"),
            "people_tracks_2d_topic": LaunchConfiguration("people_tracks_2d_topic"),
            "people_tracks_3d_topic": LaunchConfiguration("people_tracks_3d_topic"),
            "tracking_compat_topic": LaunchConfiguration("tracking_compat_topic"),
            "follow_pose_topic": LaunchConfiguration("follow_pose_topic"),
            "color_topic": LaunchConfiguration("color_topic"),
            "depth_topic": LaunchConfiguration("depth_topic"),
            "camera_info_topic": LaunchConfiguration("camera_info_topic"),
            "camera_link_frame": LaunchConfiguration("camera_link_frame"),
            "model_path": LaunchConfiguration("model_path"),
            "device": LaunchConfiguration("device"),
            "imgsz": LaunchConfiguration("imgsz"),
            "enable_ui": LaunchConfiguration("enable_ui"),
            "launch_nav2_bridge": "false",
        }.items(),
    )

    target_filter_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([package_share, "launch", "people_follow_target_filter.launch.py"])
        ),
        launch_arguments={
            "enabled": LaunchConfiguration("filter_enabled"),
            "global_frame": LaunchConfiguration("global_frame"),
            "source_frame": LaunchConfiguration("source_frame"),
            "filtered_frame": LaunchConfiguration("filtered_frame"),
            "filtered_pose_topic": LaunchConfiguration("filtered_pose_topic"),
            "enable_service_name": LaunchConfiguration("filter_enable_service_name"),
            "publish_rate_hz": LaunchConfiguration("filter_publish_rate_hz"),
            "transform_timeout_s": LaunchConfiguration("filter_transform_timeout_s"),
            "source_max_age_s": LaunchConfiguration("filter_source_max_age_s"),
            "target_lost_timeout_s": LaunchConfiguration("filter_target_lost_timeout_s"),
            "prediction_horizon_s": LaunchConfiguration("prediction_horizon_s"),
            "position_gain": LaunchConfiguration("position_gain"),
            "velocity_gain": LaunchConfiguration("velocity_gain"),
            "max_target_speed_mps": LaunchConfiguration("max_target_speed_mps"),
            "reset_distance_m": LaunchConfiguration("filter_reset_distance_m"),
            "measurement_outlier_distance_m": LaunchConfiguration("measurement_outlier_distance_m"),
            "measurement_outlier_required_count": LaunchConfiguration("measurement_outlier_required_count"),
            "measurement_hard_reject_distance_m": LaunchConfiguration("measurement_hard_reject_distance_m"),
        }.items(),
    )

    nav2_bridge_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([package_share, "launch", "people_follow_nav2_bridge.launch.py"])
        ),
        condition=IfCondition(LaunchConfiguration("launch_filtered_nav2_bridge")),
        launch_arguments={
            "enabled": LaunchConfiguration("nav2_bridge_enabled"),
            "global_frame": LaunchConfiguration("global_frame"),
            "robot_base_frame": LaunchConfiguration("robot_base_frame"),
            "tracked_frame": LaunchConfiguration("filtered_frame"),
            "goal_command_mode": LaunchConfiguration("goal_command_mode"),
            "goal_pose_topic": LaunchConfiguration("goal_pose_topic"),
            "goal_update_topic": LaunchConfiguration("goal_update_topic"),
            "navigate_to_pose_action": LaunchConfiguration("navigate_to_pose_action"),
            "enable_service_name": LaunchConfiguration("nav2_bridge_enable_service_name"),
            "goal_update_rate_hz": LaunchConfiguration("goal_update_rate_hz"),
            "goal_republish_period_s": LaunchConfiguration("goal_republish_period_s"),
            "min_goal_translation_delta_m": LaunchConfiguration("min_goal_translation_delta_m"),
            "min_goal_yaw_delta_rad": LaunchConfiguration("min_goal_yaw_delta_rad"),
            "follow_standoff_distance_m": LaunchConfiguration("follow_standoff_distance_m"),
            "follow_distance_tolerance_m": LaunchConfiguration("follow_distance_tolerance_m"),
            "target_smoothing_alpha": LaunchConfiguration("target_smoothing_alpha"),
            "target_smoothing_reset_distance_m": LaunchConfiguration("target_smoothing_reset_distance_m"),
            "publish_hold_goal_on_disable": LaunchConfiguration("publish_hold_goal_on_disable"),
            "transform_timeout_s": LaunchConfiguration("nav2_transform_timeout_s"),
            "target_lost_timeout_s": LaunchConfiguration("nav2_target_lost_timeout_s"),
            "behavior_tree_path": LaunchConfiguration("behavior_tree_path"),
            "behavior_tree_package": LaunchConfiguration("behavior_tree_package"),
            "behavior_tree_relative_path": LaunchConfiguration("behavior_tree_relative_path"),
        }.items(),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("tracking_service_name", default_value="/yoloe/set_tracking"),
            DeclareLaunchArgument("people_tracks_2d_topic", default_value="/people_tracks_2d"),
            DeclareLaunchArgument("people_tracks_3d_topic", default_value="/people_tracks_3d"),
            DeclareLaunchArgument(
                "tracking_compat_topic",
                default_value="/yoloe/tracking_detections",
            ),
            DeclareLaunchArgument("follow_pose_topic", default_value="/people/follow_target_pose"),
            DeclareLaunchArgument("color_topic", default_value="/camera0/color/image_raw"),
            DeclareLaunchArgument("depth_topic", default_value="/camera0/depth/image_rect_raw"),
            DeclareLaunchArgument("camera_info_topic", default_value="/camera0/color/camera_info"),
            DeclareLaunchArgument("camera_link_frame", default_value="camera0_link"),
            DeclareLaunchArgument("model_path", default_value="/home/usern/robocup_ws/yolo11n.pt"),
            DeclareLaunchArgument("device", default_value="cuda:0"),
            DeclareLaunchArgument("imgsz", default_value="416"),
            DeclareLaunchArgument("enable_ui", default_value="true"),
            DeclareLaunchArgument("launch_nav2_bridge", default_value="true"),
            DeclareLaunchArgument(
                "launch_filtered_nav2_bridge",
                default_value=LaunchConfiguration("launch_nav2_bridge"),
            ),
            DeclareLaunchArgument("filter_enabled", default_value="true"),
            DeclareLaunchArgument("global_frame", default_value="map"),
            DeclareLaunchArgument("source_frame", default_value="follow_target"),
            DeclareLaunchArgument("filtered_frame", default_value="follow_target_filtered"),
            DeclareLaunchArgument("filtered_pose_topic", default_value="/people/follow_target_filtered"),
            DeclareLaunchArgument(
                "filter_enable_service_name",
                default_value="/people_follow_target_filter/set_enabled",
            ),
            DeclareLaunchArgument("filter_publish_rate_hz", default_value="15.0"),
            DeclareLaunchArgument("filter_transform_timeout_s", default_value="0.1"),
            DeclareLaunchArgument("filter_source_max_age_s", default_value="0.35"),
            DeclareLaunchArgument("filter_target_lost_timeout_s", default_value="1.0"),
            DeclareLaunchArgument("prediction_horizon_s", default_value="0.25"),
            DeclareLaunchArgument("position_gain", default_value="0.65"),
            DeclareLaunchArgument("velocity_gain", default_value="0.20"),
            DeclareLaunchArgument("max_target_speed_mps", default_value="1.8"),
            DeclareLaunchArgument("filter_reset_distance_m", default_value="1.2"),
            DeclareLaunchArgument("measurement_outlier_distance_m", default_value="1.0"),
            DeclareLaunchArgument("measurement_outlier_required_count", default_value="5"),
            DeclareLaunchArgument("measurement_hard_reject_distance_m", default_value="1.5"),
            DeclareLaunchArgument("nav2_bridge_enabled", default_value="false"),
            DeclareLaunchArgument("robot_base_frame", default_value="base_footprint"),
            DeclareLaunchArgument("goal_command_mode", default_value="navigate_to_pose"),
            DeclareLaunchArgument("goal_pose_topic", default_value="/goal_pose"),
            DeclareLaunchArgument("goal_update_topic", default_value="/goal_update"),
            DeclareLaunchArgument("navigate_to_pose_action", default_value="/navigate_to_pose"),
            DeclareLaunchArgument(
                "nav2_bridge_enable_service_name",
                default_value="/people_follow_nav2/set_enabled",
            ),
            DeclareLaunchArgument("goal_update_rate_hz", default_value="10.0"),
            DeclareLaunchArgument("goal_republish_period_s", default_value="0.0"),
            DeclareLaunchArgument("min_goal_translation_delta_m", default_value="0.05"),
            DeclareLaunchArgument("min_goal_yaw_delta_rad", default_value="0.12"),
            DeclareLaunchArgument("follow_standoff_distance_m", default_value="0.7"),
            DeclareLaunchArgument("follow_distance_tolerance_m", default_value="0.10"),
            DeclareLaunchArgument("target_smoothing_alpha", default_value="0.65"),
            DeclareLaunchArgument("target_smoothing_reset_distance_m", default_value="0.40"),
            DeclareLaunchArgument("publish_hold_goal_on_disable", default_value="true"),
            DeclareLaunchArgument("nav2_transform_timeout_s", default_value="0.2"),
            DeclareLaunchArgument("nav2_target_lost_timeout_s", default_value="10.0"),
            DeclareLaunchArgument(
                "behavior_tree_path",
                default_value="/workspaces/isaac_ros-dev/src/wheeltec_robot_nav2/param/bt/follower_w_recovery.xml",
            ),
            DeclareLaunchArgument("behavior_tree_package", default_value="deepsort_people_follow"),
            DeclareLaunchArgument(
                "behavior_tree_relative_path",
                default_value="bt/follower_w_recovery.xml",
            ),
            tracker_launch,
            target_filter_launch,
            nav2_bridge_launch,
        ]
    )

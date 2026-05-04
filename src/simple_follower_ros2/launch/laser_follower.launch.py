import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
import launch_ros.actions

def generate_launch_description():
    enabled = LaunchConfiguration('enabled')
    avoidance_enabled = LaunchConfiguration('avoidance_enabled')
    rviz = LaunchConfiguration('rviz')
    target_radius = LaunchConfiguration('target_radius')
    lost_timeout = LaunchConfiguration('lost_timeout')
    max_target_jump = LaunchConfiguration('max_target_jump')
    target_smoothing_alpha = LaunchConfiguration('target_smoothing_alpha')
    max_target_width = LaunchConfiguration('max_target_width')
    max_cluster_depth = LaunchConfiguration('max_cluster_depth')
    max_linearity_ratio = LaunchConfiguration('max_linearity_ratio')
    max_speed = LaunchConfiguration('max_speed')
    target_distance = LaunchConfiguration('target_distance')
    vision_validation_enabled = LaunchConfiguration('vision_validation_enabled')
    vision_service_name = LaunchConfiguration('vision_service_name')
    vision_prompt = LaunchConfiguration('vision_prompt')
    vision_camera_name = LaunchConfiguration('vision_camera_name')
    vision_period = LaunchConfiguration('vision_period')
    vision_request_timeout = LaunchConfiguration('vision_request_timeout')
    vision_valid_timeout = LaunchConfiguration('vision_valid_timeout')
    vision_min_confidence = LaunchConfiguration('vision_min_confidence')
    vision_laser_angle_tolerance = LaunchConfiguration('vision_laser_angle_tolerance')
    vision_laser_distance_tolerance = LaunchConfiguration('vision_laser_distance_tolerance')
    require_vision_for_acquire = LaunchConfiguration('require_vision_for_acquire')
    rviz_config = os.path.join(
        get_package_share_directory('simple_follower_ros2'),
        'rviz',
        'laser_follower_debug.rviz',
    )
    return LaunchDescription([
        DeclareLaunchArgument(
            'enabled',
            default_value='true',
            description='Enable laser tracking and following at startup',
        ),
        DeclareLaunchArgument(
            'avoidance_enabled',
            default_value='true',
            description='Enable potential-field obstacle avoidance in the laser follower',
        ),
        DeclareLaunchArgument(
            'rviz',
            default_value='true',
            description='Start RViz with the laser follower debug config',
        ),
        DeclareLaunchArgument(
            'target_radius',
            default_value='0.35',
            description='Radius around previous target used for continuous laser target tracking',
        ),
        DeclareLaunchArgument(
            'lost_timeout',
            default_value='2.0',
            description='Seconds to keep target lock before reacquiring a new target',
        ),
        DeclareLaunchArgument(
            'max_target_jump',
            default_value='0.25',
            description='Reject target centroid updates that jump farther than this in meters',
        ),
        DeclareLaunchArgument(
            'target_smoothing_alpha',
            default_value='0.85',
            description='Low-pass filter alpha for target centroid updates; lower is smoother',
        ),
        DeclareLaunchArgument(
            'max_speed',
            default_value='0.4',
            description='Maximum linear and angular speed commanded by the laser follower',
        ),
        DeclareLaunchArgument(
            'target_distance',
            default_value='0.8',
            description='Target following distance in meters',
        ),
        DeclareLaunchArgument(
            'vision_validation_enabled',
            default_value='true',
            description='Use YOLO camera0 person detection to gate laser target acquisition',
        ),
        DeclareLaunchArgument(
            'vision_service_name',
            default_value='/yoloe/detect_prompt',
            description='YOLO prompt detection service name',
        ),
        DeclareLaunchArgument(
            'vision_prompt',
            default_value='person',
            description='Text prompt sent to the YOLO detection service',
        ),
        DeclareLaunchArgument(
            'vision_camera_name',
            default_value='camera0',
            description='Camera name sent to the YOLO detection service',
        ),
        DeclareLaunchArgument(
            'vision_period',
            default_value='0.5',
            description='Seconds between YOLO validation requests while tracking is enabled',
        ),
        DeclareLaunchArgument(
            'vision_request_timeout',
            default_value='5.0',
            description='Seconds to wait before dropping a slow YOLO validation request',
        ),
        DeclareLaunchArgument(
            'vision_valid_timeout',
            default_value='3.0',
            description='Seconds a YOLO person detection remains valid for laser acquisition',
        ),
        DeclareLaunchArgument(
            'vision_min_confidence',
            default_value='0.25',
            description='Minimum YOLO confidence accepted as person confirmation',
        ),
        DeclareLaunchArgument(
            'vision_laser_angle_tolerance',
            default_value='0.45',
            description='Maximum angle difference in radians between camera person pose and laser target',
        ),
        DeclareLaunchArgument(
            'vision_laser_distance_tolerance',
            default_value='0.8',
            description='Maximum distance difference in meters between camera person pose and laser target',
        ),
        DeclareLaunchArgument(
            'require_vision_for_acquire',
            default_value='true',
            description='Require recent YOLO person detection before laser acquire/reacquire',
        ),
        
        launch_ros.actions.Node(
            package='simple_follower_ros2', 
            executable='lasertracker', 
            name='lasertracker',
            parameters=[{
                'enabled': enabled,
                'target_radius': target_radius,
                'lost_timeout': lost_timeout,
                'max_target_jump': max_target_jump,
                'target_smoothing_alpha': target_smoothing_alpha,
                'vision_validation_enabled': vision_validation_enabled,
                'vision_service_name': vision_service_name,
                'vision_prompt': vision_prompt,
                'vision_camera_name': vision_camera_name,
                'vision_period': vision_period,
                'vision_request_timeout': vision_request_timeout,
                'vision_valid_timeout': vision_valid_timeout,
                'vision_min_confidence': vision_min_confidence,
                'vision_laser_angle_tolerance': vision_laser_angle_tolerance,
                'vision_laser_distance_tolerance': vision_laser_distance_tolerance,
                'require_vision_for_acquire': require_vision_for_acquire,
            }],
             ),
        launch_ros.actions.Node(
            package='simple_follower_ros2', 
            executable='laserfollower', 
            parameters=[{
                'enabled': enabled,
                'avoidance_enabled': avoidance_enabled,
                'max_speed': max_speed,
                'target_distance': target_distance,
            }],
            ),
        launch_ros.actions.Node(
            package='rviz2',
            executable='rviz2',
            name='laser_follower_rviz',
            arguments=['-d', rviz_config],
            condition=IfCondition(rviz),
            output='screen',
            ),]
    )

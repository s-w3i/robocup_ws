import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    wheeltec_nav_dir = get_package_share_directory('wheeltec_nav2')
    nvblox_examples_dir = get_package_share_directory('nvblox_examples_bringup')

    map_file = LaunchConfiguration('map')
    params_file = LaunchConfiguration('params')
    rviz_config = LaunchConfiguration('rviz_config')
    use_sim_time = LaunchConfiguration('use_sim_time')
    use_rviz = LaunchConfiguration('use_rviz')
    launch_nvblox = LaunchConfiguration('launch_nvblox')

    rosbag = LaunchConfiguration('rosbag')
    rosbag_args = LaunchConfiguration('rosbag_args')
    log_level = LaunchConfiguration('log_level')
    num_cameras = LaunchConfiguration('num_cameras')
    camera_serial_numbers = LaunchConfiguration('camera_serial_numbers')
    mode = LaunchConfiguration('mode')
    container_name = LaunchConfiguration('container_name')
    run_realsense = LaunchConfiguration('run_realsense')
    publish_fake_odom_tf = LaunchConfiguration('publish_fake_odom_tf')
    publish_camera_link_bridge_tf = LaunchConfiguration('publish_camera_link_bridge_tf')
    global_frame = LaunchConfiguration('global_frame')
    nvblox_base_frame = LaunchConfiguration('nvblox_base_frame')
    rviz_target_frame = LaunchConfiguration('rviz_target_frame')
    esdf_slice_height = LaunchConfiguration('esdf_slice_height')
    esdf_slice_min_height = LaunchConfiguration('esdf_slice_min_height')
    esdf_slice_max_height = LaunchConfiguration('esdf_slice_max_height')
    camera_link_bridge_parent_frame = LaunchConfiguration('camera_link_bridge_parent_frame')
    camera_link_bridge_child_frame = LaunchConfiguration('camera_link_bridge_child_frame')
    camera_link_bridge_translation = LaunchConfiguration('camera_link_bridge_translation')
    camera_link_bridge_orientation_rpy = LaunchConfiguration('camera_link_bridge_orientation_rpy')
    launch_tracking_relay = LaunchConfiguration('launch_tracking_relay')
    tracking_input_color_topic = LaunchConfiguration('tracking_input_color_topic')
    tracking_input_depth_topic = LaunchConfiguration('tracking_input_depth_topic')
    tracking_input_camera_info_topic = LaunchConfiguration('tracking_input_camera_info_topic')
    tracking_output_color_topic = LaunchConfiguration('tracking_output_color_topic')
    tracking_output_depth_topic = LaunchConfiguration('tracking_output_depth_topic')
    tracking_output_camera_info_topic = LaunchConfiguration('tracking_output_camera_info_topic')
    tracking_color_rate_hz = LaunchConfiguration('tracking_color_rate_hz')
    tracking_depth_rate_hz = LaunchConfiguration('tracking_depth_rate_hz')
    tracking_output_width = LaunchConfiguration('tracking_output_width')
    tracking_output_height = LaunchConfiguration('tracking_output_height')

    return LaunchDescription([
        DeclareLaunchArgument(
            'map',
            default_value=os.path.join(wheeltec_nav_dir, 'map', 'athome_1.yaml'),
            description='Full path to the static map YAML file used by Nav2.'),
        DeclareLaunchArgument(
            'params',
            default_value=os.path.join(
                wheeltec_nav_dir, 'param', 'wheeltec_params', 'param_senior_diff.yaml'),
            description='Full path to the Nav2 parameter file.'),
        DeclareLaunchArgument(
            'rviz_config',
            default_value=os.path.join(wheeltec_nav_dir, 'rviz', 'wheeltec_nvblox.rviz'),
            description='Full path to the RViz config file.'),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation time for Nav2.'),
        DeclareLaunchArgument(
            'use_rviz',
            default_value='true',
            description='Launch a single Wheeltec + NVBlox RViz instance.'),
        DeclareLaunchArgument(
            'launch_nvblox',
            default_value='false',
            description='Launch the nvblox mapping stack in addition to the camera bringup.'),
        DeclareLaunchArgument(
            'rosbag',
            default_value='None',
            description='Optional rosbag path for the nvblox bringup.'),
        DeclareLaunchArgument(
            'rosbag_args',
            default_value='',
            description='Additional rosbag playback arguments.'),
        DeclareLaunchArgument(
            'log_level',
            default_value='info',
            description='Shared log level for the bringup.'),
        DeclareLaunchArgument(
            'num_cameras',
            default_value='1',
            description='Number of Realsense cameras for nvblox.'),
        DeclareLaunchArgument(
            'camera_serial_numbers',
            default_value='',
            description='Comma-separated list of additional Realsense serial numbers.'),
        DeclareLaunchArgument(
            'mode',
            default_value='static',
            description='Nvblox mapping mode.'),
        DeclareLaunchArgument(
            'container_name',
            default_value='nvblox_container',
            description='Composable container name used by nvblox.'),
        DeclareLaunchArgument(
            'run_realsense',
            default_value='true',
            description='Launch the Realsense driver stack for nvblox.'),
        DeclareLaunchArgument(
            'publish_fake_odom_tf',
            default_value='false',
            description='Publish the fake odom TF used by camera-only demos.'),
        DeclareLaunchArgument(
            'publish_camera_link_bridge_tf',
            default_value='true',
            description='Bridge head_camera_link to camera0_link for the real robot TF tree.'),
        DeclareLaunchArgument(
            'global_frame',
            default_value='odom_combined',
            description='Global frame used by nvblox.'),
        DeclareLaunchArgument(
            'nvblox_base_frame',
            default_value='base_link',
            description='Base frame used by nvblox for map clearing and visualization.'),
        DeclareLaunchArgument(
            'rviz_target_frame',
            default_value='base_link',
            description='Target frame used by the RViz camera controls.'),
        DeclareLaunchArgument(
            'esdf_slice_height',
            default_value='0.0',
            description='Published nvblox ESDF slice height used by Nav2.'),
        DeclareLaunchArgument(
            'esdf_slice_min_height',
            default_value='0.1',
            description='Lower bound of the nvblox 2D ESDF band for obstacle projection.'),
        DeclareLaunchArgument(
            'esdf_slice_max_height',
            default_value='1.5',
            description='Upper bound of the nvblox 2D ESDF band for obstacle projection.'),
        DeclareLaunchArgument(
            'camera_link_bridge_parent_frame',
            default_value='head_camera_link',
            description='Parent frame for the static TF bridge to camera0_link.'),
        DeclareLaunchArgument(
            'camera_link_bridge_child_frame',
            default_value='camera0_link',
            description='Child frame for the static TF bridge.'),
        DeclareLaunchArgument(
            'camera_link_bridge_translation',
            default_value='0,0,0',
            description='XYZ translation for the static camera TF bridge.'),
        DeclareLaunchArgument(
            'camera_link_bridge_orientation_rpy',
            default_value='0,0,0',
            description='RPY orientation for the static camera TF bridge.'),
        DeclareLaunchArgument(
            'launch_tracking_relay',
            default_value='false',
            description='Republish reduced-rate tracking topics from the container side.'),
        DeclareLaunchArgument(
            'tracking_input_color_topic',
            default_value='/camera0/color/image_raw',
            description='Full-rate color topic consumed by the tracking relay.'),
        DeclareLaunchArgument(
            'tracking_input_depth_topic',
            default_value='/camera0/realsense_splitter_node/output/depth',
            description='Full-rate depth topic consumed by the tracking relay.'),
        DeclareLaunchArgument(
            'tracking_input_camera_info_topic',
            default_value='/camera0/color/camera_info',
            description='Full-rate camera info topic consumed by the tracking relay.'),
        DeclareLaunchArgument(
            'tracking_output_color_topic',
            default_value='/camera0/tracking/color/image_raw',
            description='Reduced-rate color topic exported for remote tracking clients.'),
        DeclareLaunchArgument(
            'tracking_output_depth_topic',
            default_value='/camera0/tracking/depth/image_raw',
            description='Reduced-rate depth topic exported for remote tracking clients.'),
        DeclareLaunchArgument(
            'tracking_output_camera_info_topic',
            default_value='/camera0/tracking/color/camera_info',
            description='Reduced-rate camera info topic exported for remote tracking clients.'),
        DeclareLaunchArgument(
            'tracking_color_rate_hz',
            default_value='8.0',
            description='Publish rate for the reduced-rate tracking color stream.'),
        DeclareLaunchArgument(
            'tracking_depth_rate_hz',
            default_value='4.0',
            description='Publish rate for the reduced-rate tracking depth stream.'),
        DeclareLaunchArgument(
            'tracking_output_width',
            default_value='320',
            description='Width of resized tracking images.'),
        DeclareLaunchArgument(
            'tracking_output_height',
            default_value='240',
            description='Height of resized tracking images.'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(nvblox_examples_dir, 'launch', 'realsense_nvblox_only.launch.py')),
            launch_arguments={
                'rosbag': rosbag,
                'rosbag_args': rosbag_args,
                'log_level': log_level,
                'num_cameras': num_cameras,
                'camera_serial_numbers': camera_serial_numbers,
                'mode': mode,
                'container_name': container_name,
                'run_realsense': run_realsense,
                'run_rviz': 'False',
                'publish_fake_odom_tf': publish_fake_odom_tf,
                'publish_camera_link_bridge_tf': publish_camera_link_bridge_tf,
                'global_frame': global_frame,
                'nvblox_base_frame': nvblox_base_frame,
                'rviz_target_frame': rviz_target_frame,
                'esdf_slice_height': esdf_slice_height,
                'esdf_slice_min_height': esdf_slice_min_height,
                'esdf_slice_max_height': esdf_slice_max_height,
                'camera_link_bridge_parent_frame': camera_link_bridge_parent_frame,
                'camera_link_bridge_child_frame': camera_link_bridge_child_frame,
                'camera_link_bridge_translation': camera_link_bridge_translation,
                'camera_link_bridge_orientation_rpy': camera_link_bridge_orientation_rpy,
            }.items(),
            condition=IfCondition(launch_nvblox)),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(nvblox_examples_dir, 'launch', 'sensors', 'realsense.launch.py')),
            launch_arguments={
                'container_name': container_name,
                'camera_serial_numbers': camera_serial_numbers,
                'num_cameras': num_cameras,
                'run_standalone': 'true',
            }.items(),
            condition=IfCondition(PythonExpression([
                '"', launch_nvblox, '" == "false" and "', run_realsense, '" == "true"'
            ]))),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='camera_link_bridge_tf_pub_camera_only',
            output='screen',
            arguments=[
                PythonExpression(['"', camera_link_bridge_translation, '".split(",")[0]']),
                PythonExpression(['"', camera_link_bridge_translation, '".split(",")[1]']),
                PythonExpression(['"', camera_link_bridge_translation, '".split(",")[2]']),
                PythonExpression(['"', camera_link_bridge_orientation_rpy, '".split(",")[0]']),
                PythonExpression(['"', camera_link_bridge_orientation_rpy, '".split(",")[1]']),
                PythonExpression(['"', camera_link_bridge_orientation_rpy, '".split(",")[2]']),
                camera_link_bridge_parent_frame,
                camera_link_bridge_child_frame,
            ],
            condition=IfCondition(PythonExpression([
                '"', launch_nvblox, '" == "false" and "',
                publish_camera_link_bridge_tf, '" == "true"'
            ]))),
        Node(
            package='wheeltec_nav2',
            executable='tracking_input_relay.py',
            name='tracking_input_relay',
            condition=IfCondition(launch_tracking_relay),
            output='screen',
            parameters=[{
                'input_color_topic': tracking_input_color_topic,
                'input_depth_topic': tracking_input_depth_topic,
                'input_camera_info_topic': tracking_input_camera_info_topic,
                'output_color_topic': tracking_output_color_topic,
                'output_depth_topic': tracking_output_depth_topic,
                'output_camera_info_topic': tracking_output_camera_info_topic,
                'color_rate_hz': tracking_color_rate_hz,
                'depth_rate_hz': tracking_depth_rate_hz,
                'output_width': tracking_output_width,
                'output_height': tracking_output_height,
            }]),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(wheeltec_nav_dir, 'launch', 'bringup_launch.py')),
            launch_arguments={
                'map': map_file,
                'use_sim_time': use_sim_time,
                'params_file': params_file,
                'log_level': log_level,
            }.items()),
        Node(
            package='rviz2',
            executable='rviz2',
            name='wheeltec_nav2_rviz',
            arguments=['-d', rviz_config],
            condition=IfCondition(use_rviz),
            output='screen'),
    ])

import os
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    use_rviz = LaunchConfiguration('use_rviz', default='false')

    # wheeltec_robot_dir = get_package_share_directory('turn_on_wheeltec_robot')
    # wheeltec_launch_dir = os.path.join(wheeltec_robot_dir, 'launch')
        
    wheeltec_nav_dir = get_package_share_directory('wheeltec_nav2')
    wheeltec_nav_launchr = os.path.join(wheeltec_nav_dir, 'launch')


    map_dir = os.path.join(wheeltec_nav_dir, 'map')
    map_file = LaunchConfiguration('map', default=os.path.join(
        map_dir, 'map.yaml'))


    #Modify the model parameter file, the options are:
    #param_mini_akm.yaml/param_mini_4wd.yaml/param_mini_diff.yaml/
    #param_mini_mec.yaml/param_mini_omni.yaml/param_mini_tank.yaml/
    #param_senior_akm.yaml/param_senior_diff.yaml/param_senior_mec_bs.yaml
    #param_senior_mec_dl.yaml/param_top_4wd_bs.yaml/param_top_4wd_dl.yaml
    #param_top_akm_dl.yaml/param_four_wheel_diff_dl.yaml/param_four_wheel_diff_bs.yaml

    param_dir = os.path.join(wheeltec_nav_dir, 'param','wheeltec_params')
    param_file = LaunchConfiguration('params', default=os.path.join(
        param_dir, 'nav2_param.yaml'))
    rviz_config = LaunchConfiguration('rviz_config', default=os.path.join(
        wheeltec_nav_dir, 'rviz', 'wheeltec_nvblox.rviz'))


    return LaunchDescription([
        DeclareLaunchArgument(
            'map',
            default_value=map_file,
            description='Full path to map file to load'),

        DeclareLaunchArgument(
            'params',
            default_value=param_file,
            description='Full path to param file to load'),
        DeclareLaunchArgument(
            'use_rviz',
            default_value=use_rviz,
            description='Whether to launch rviz2 with the Wheeltec + NVBlox config'),
        DeclareLaunchArgument(
            'rviz_config',
            default_value=rviz_config,
            description='Full path to the rviz config file to use'),
        # IncludeLaunchDescription(
        #     PythonLaunchDescriptionSource(
        #         [wheeltec_launch_dir, '/turn_on_wheeltec_robot.launch.py']),
        # ),
        # IncludeLaunchDescription(
        #     PythonLaunchDescriptionSource(
        #         [wheeltec_launch_dir, '/wheeltec_lidar.launch.py']),
        # ),        
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [wheeltec_nav_launchr, '/bringup_launch.py']),
            launch_arguments={
                'map': map_file,
                'use_sim_time': use_sim_time,
                'params_file': param_file}.items(),
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='wheeltec_nav2_rviz',
            arguments=['-d', rviz_config],
            condition=IfCondition(use_rviz),
            output='screen'),

    ])

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    default_site_packages = "/home/usern/coqui-venv/lib/python3.10/site-packages"
    args = [
        DeclareLaunchArgument("service_name", default_value="/yoloe/detect_prompt"),
        DeclareLaunchArgument("model_path", default_value="/home/usern/Kevin_yolo/best_seg4.pt"),
        DeclareLaunchArgument("bag_model_path", default_value="/home/usern/Kevin_yolo/best_seg4.pt"),
        DeclareLaunchArgument("device", default_value="auto"),
        DeclareLaunchArgument("imgsz", default_value="640"),
        DeclareLaunchArgument("conf", default_value="0.25"),
        DeclareLaunchArgument("iou", default_value="0.45"),
        DeclareLaunchArgument("max_det", default_value="300"),
        DeclareLaunchArgument("default_camera_name", default_value="camera0"),
        DeclareLaunchArgument("color_topic", default_value="/camera0/color/image_raw"),
        DeclareLaunchArgument("depth_topic", default_value="/camera0/realsense_splitter_node/output/depth"),
        DeclareLaunchArgument("camera_info_topic", default_value="/camera0/color/camera_info"),
        DeclareLaunchArgument("camera_link_frame", default_value="camera0_link"),
        DeclareLaunchArgument("camera0_color_topic", default_value="/camera0/color/image_raw"),
        DeclareLaunchArgument("camera0_depth_topic", default_value=""),
        DeclareLaunchArgument("camera0_camera_info_topic", default_value=""),
        DeclareLaunchArgument("camera0_camera_link_frame", default_value=""),
        DeclareLaunchArgument("camera_color_topic", default_value="/gripper_camera/color/image_raw"),
        DeclareLaunchArgument("camera_depth_topic", default_value="/gripper_camera/depth/image_raw"),
        DeclareLaunchArgument("camera_camera_info_topic", default_value="/gripper_camera/color/camera_info"),
        DeclareLaunchArgument("camera_camera_link_frame", default_value="gripper_camera_link"),
        DeclareLaunchArgument("save_dir", default_value="/home/usern/robocup_ws/yoloe_out"),
        DeclareLaunchArgument("always_save_image", default_value="true"),
        DeclareLaunchArgument("python_site_packages", default_value=default_site_packages),
    ]

    node = Node(
        package="yoloe_detection_service",
        executable="yoloe_seg4_detection_service_node",
        name="yoloe_seg4_detection_service_node",
        output="screen",
        additional_env={
            "PYTHONPATH": [
                LaunchConfiguration("python_site_packages"),
                ":",
                EnvironmentVariable("PYTHONPATH", default_value=""),
            ],
            "LD_LIBRARY_PATH": [
                LaunchConfiguration("python_site_packages"),
                "/nvidia/cublas/lib:",
                LaunchConfiguration("python_site_packages"),
                "/nvidia/cudnn/lib:",
                LaunchConfiguration("python_site_packages"),
                "/nvidia/cuda_cupti/lib:",
                LaunchConfiguration("python_site_packages"),
                "/nvidia/cuda_nvrtc/lib:",
                LaunchConfiguration("python_site_packages"),
                "/nvidia/cuda_runtime/lib:",
                "/usr/local/cuda/lib64:",
                "/usr/local/cuda-12.6/lib64:",
                "/usr/local/cuda/targets/aarch64-linux/lib:",
                "/usr/local/cuda-12.6/targets/aarch64-linux/lib:",
                "/lib/aarch64-linux-gnu:",
                "/usr/lib/aarch64-linux-gnu:",
                EnvironmentVariable("LD_LIBRARY_PATH", default_value=""),
            ],
        },
        parameters=[
            {
                "service_name": LaunchConfiguration("service_name"),
                "model_path": LaunchConfiguration("model_path"),
                "bag_model_path": LaunchConfiguration("bag_model_path"),
                "device": LaunchConfiguration("device"),
                "imgsz": LaunchConfiguration("imgsz"),
                "conf": LaunchConfiguration("conf"),
                "iou": LaunchConfiguration("iou"),
                "max_det": LaunchConfiguration("max_det"),
                "default_camera_name": LaunchConfiguration("default_camera_name"),
                "color_topic": LaunchConfiguration("color_topic"),
                "depth_topic": LaunchConfiguration("depth_topic"),
                "camera_info_topic": LaunchConfiguration("camera_info_topic"),
                "camera_link_frame": LaunchConfiguration("camera_link_frame"),
                "camera0_color_topic": LaunchConfiguration("camera0_color_topic"),
                "camera0_depth_topic": LaunchConfiguration("camera0_depth_topic"),
                "camera0_camera_info_topic": LaunchConfiguration("camera0_camera_info_topic"),
                "camera0_camera_link_frame": LaunchConfiguration("camera0_camera_link_frame"),
                "camera_color_topic": LaunchConfiguration("camera_color_topic"),
                "camera_depth_topic": LaunchConfiguration("camera_depth_topic"),
                "camera_camera_info_topic": LaunchConfiguration("camera_camera_info_topic"),
                "camera_camera_link_frame": LaunchConfiguration("camera_camera_link_frame"),
                "save_dir": LaunchConfiguration("save_dir"),
                "always_save_image": LaunchConfiguration("always_save_image"),
            }
        ],
    )

    return LaunchDescription(args + [node])

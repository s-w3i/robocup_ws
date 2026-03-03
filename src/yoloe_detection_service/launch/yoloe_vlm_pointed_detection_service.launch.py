from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pose_site = "/home/usern/pose-venv/lib/python3.10/site-packages"
    coqui_site = "/home/usern/coqui-venv/lib/python3.10/site-packages"

    args = [
        DeclareLaunchArgument("model_path", default_value="/home/usern/yoloe-26l-seg.pt"),
        DeclareLaunchArgument("device", default_value="auto"),
        DeclareLaunchArgument("service_name", default_value="/yoloe/detect_pointed_prompt_vlm"),
        DeclareLaunchArgument("color_topic", default_value="/camera0/color/image_raw"),
        DeclareLaunchArgument("depth_topic", default_value="/camera0/depth/image_rect_raw"),
        DeclareLaunchArgument("camera_info_topic", default_value="/camera0/color/camera_info"),
        DeclareLaunchArgument("camera_link_frame", default_value="camera0_link"),
        DeclareLaunchArgument("tf_ttl_sec", default_value="60.0"),
        DeclareLaunchArgument("tf_republish_hz", default_value="10.0"),
        DeclareLaunchArgument("show_ui", default_value="true"),
        DeclareLaunchArgument("save_dir", default_value="/home/usern/robocup_ws/yoloe_out"),
        DeclareLaunchArgument("vlm_model", default_value="qwen3-vl:8b"),
        DeclareLaunchArgument("ollama_base_url", default_value="http://127.0.0.1:11434"),
        DeclareLaunchArgument("vote_frames", default_value="5"),
        DeclareLaunchArgument("vlm_num_predict", default_value="128"),
        DeclareLaunchArgument("vlm_retry_num_predict", default_value="384"),
        DeclareLaunchArgument("vlm_max_retries", default_value="1"),
        DeclareLaunchArgument("vlm_max_candidates", default_value="12"),
        DeclareLaunchArgument("vlm_use_thinking_fallback", default_value="true"),
        DeclareLaunchArgument("vlm_image_max_edge", default_value="960"),
    ]

    node = Node(
        package="yoloe_detection_service",
        executable="yoloe_vlm_pointed_detection_service_node",
        name="yoloe_vlm_pointed_detection_service_node",
        output="screen",
        emulate_tty=True,
        additional_env={
            "PYTHONPATH": [
                pose_site,
                ":",
                coqui_site,
                ":",
                EnvironmentVariable("PYTHONPATH", default_value=""),
            ],
            "LD_LIBRARY_PATH": [
                f"{coqui_site}/nvidia/cublas/lib:",
                f"{coqui_site}/nvidia/cudnn/lib:",
                f"{coqui_site}/nvidia/cuda_cupti/lib:",
                f"{coqui_site}/nvidia/cuda_nvrtc/lib:",
                f"{coqui_site}/nvidia/cuda_runtime/lib:",
                "/usr/local/cuda/targets/aarch64-linux/lib:",
                "/usr/local/cuda-12.6/targets/aarch64-linux/lib:",
                "/lib/aarch64-linux-gnu:",
                "/usr/lib/aarch64-linux-gnu:",
                EnvironmentVariable("LD_LIBRARY_PATH", default_value=""),
            ],
        },
        parameters=[
            {
                "model_path": LaunchConfiguration("model_path"),
                "device": LaunchConfiguration("device"),
                "service_name": LaunchConfiguration("service_name"),
                "color_topic": LaunchConfiguration("color_topic"),
                "depth_topic": LaunchConfiguration("depth_topic"),
                "camera_info_topic": LaunchConfiguration("camera_info_topic"),
                "camera_link_frame": LaunchConfiguration("camera_link_frame"),
                "tf_ttl_sec": LaunchConfiguration("tf_ttl_sec"),
                "tf_republish_hz": LaunchConfiguration("tf_republish_hz"),
                "show_ui": LaunchConfiguration("show_ui"),
                "save_dir": LaunchConfiguration("save_dir"),
                "vlm_model": LaunchConfiguration("vlm_model"),
                "ollama_base_url": LaunchConfiguration("ollama_base_url"),
                "vote_frames": LaunchConfiguration("vote_frames"),
                "vlm_num_predict": LaunchConfiguration("vlm_num_predict"),
                "vlm_retry_num_predict": LaunchConfiguration("vlm_retry_num_predict"),
                "vlm_max_retries": LaunchConfiguration("vlm_max_retries"),
                "vlm_max_candidates": LaunchConfiguration("vlm_max_candidates"),
                "vlm_use_thinking_fallback": LaunchConfiguration("vlm_use_thinking_fallback"),
                "vlm_image_max_edge": LaunchConfiguration("vlm_image_max_edge"),
            }
        ],
    )

    return LaunchDescription(args + [node])

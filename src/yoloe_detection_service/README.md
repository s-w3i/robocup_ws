# YOLOE Prompt Detection Service (ROS2)

This package now provides only one-shot text-prompt detection.

## Service

- `/yoloe/detect_prompt`
- Type: `yoloe_detection_interfaces/srv/DetectObjectPrompt`

Request:

```text
string prompt_text   # e.g. "bottle" or "cup,bottle"
bool save_image
```

Response includes arrays for all valid detections:

- `detected_classes[]`
- `confidences[]`
- `poses_camera_link[]`
- `tf_child_frames[]` (e.g. `chair_1`, `chair_2`)

and `detections_in_frame`, `tf_published_count`, `saved_image_path`, `inference_ms`.

## Camera topics (default)

- Color: `/camera/color/image_raw`
- Depth: `/camera/depth/image_raw`
- Camera info: `/camera/color/camera_info`

## Build

```bash
cd /home/usern/robocup_ws
source /home/usern/coqui-venv/bin/activate
source /opt/ros/humble/setup.bash
colcon build --packages-select yoloe_detection_interfaces yoloe_detection_service --symlink-install
```

## Run

```bash
cd /home/usern/robocup_ws
./run_yoloe_detection_service_in_venv.sh
```

## Call

```bash
source /opt/ros/humble/setup.bash
source /home/usern/robocup_ws/install/setup.bash
ros2 service call /yoloe/detect_prompt yoloe_detection_interfaces/srv/DetectObjectPrompt "{prompt_text: 'bottle', save_image: true}"
```

Tracking service `/yoloe/set_tracking` is now provided by the `deepstream_people_tracking` package.

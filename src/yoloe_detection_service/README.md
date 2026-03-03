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

## Pointed-Object Service (Hand Pointing + YOLOE Prompt)

- `/yoloe/detect_pointed_prompt`
- Type: `yoloe_detection_interfaces/srv/DetectObjectPrompt`

Behavior:

- One inference per service request.
- Uses MediaPipe hand landmarks to detect left/right pointing gesture.
- Runs YOLOE with request prompt text.
- Returns only the object aligned with pointing ray (single centroid pose in `poses_camera_link[0]`).
- Publishes result UI image on `/yoloe/pointing_result_image` and optional OpenCV window.

Run:

```bash
cd /home/usern/robocup_ws
source /opt/ros/humble/setup.bash
source /home/usern/robocup_ws/install/setup.bash
ros2 launch yoloe_detection_service yoloe_pointed_detection_service.launch.py show_ui:=true
```

Call with ROS2 CLI:

```bash
source /opt/ros/humble/setup.bash
source /home/usern/robocup_ws/install/setup.bash
ros2 service call /yoloe/detect_pointed_prompt yoloe_detection_interfaces/srv/DetectObjectPrompt "{prompt_text: 'bottle', save_image: true}"
```

## VLM Alternative Pointed Service (YOLOE + Ollama VLM, VLM-Only Decision)

- `/yoloe/detect_pointed_prompt_vlm`
- Type: `yoloe_detection_interfaces/srv/DetectObjectPrompt`

Behavior:

- Runs YOLOE to produce candidate boxes for the requested prompt classes.
- Sends a candidate-labeled image to local Ollama VLM (`qwen3-vl` by default).
- VLM alone selects candidate ID (`-1` means no clear pointed target); no hand/arm cone fallback.
- Includes VLM response hardening: retries with higher `num_predict` and optional parsing from VLM `thinking` field when `content` is empty.
- Returns centroid pose and TF in `camera0_link` by default.
- Republishes detected object TF on `/tf` for a configurable TTL (`tf_ttl_sec`, default 60s), then stops.
- Publishes VLM candidate visualization on `/yoloe/vlm_pointing_query_image`.

Run:

```bash
source /opt/ros/humble/setup.bash
source /home/usern/robocup_ws/install/setup.bash
ros2 launch yoloe_detection_service yoloe_vlm_pointed_detection_service.launch.py show_ui:=false
```

Accuracy-first tuning (pure VLM):

```bash
ros2 launch yoloe_detection_service yoloe_vlm_pointed_detection_service.launch.py \
  show_ui:=false \
  vote_frames:=3 \
  vlm_num_predict:=256 \
  vlm_retry_num_predict:=512 \
  vlm_max_retries:=1 \
  vlm_max_candidates:=10 \
  vlm_image_max_edge:=1280
```

Call:

```bash
source /opt/ros/humble/setup.bash
source /home/usern/robocup_ws/install/setup.bash
ros2 service call /yoloe/detect_pointed_prompt_vlm yoloe_detection_interfaces/srv/DetectObjectPrompt "{prompt_text: 'bottle', save_image: true}"
```

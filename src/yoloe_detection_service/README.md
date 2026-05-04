# YOLOE Prompt Detection Service (ROS2)

This package now provides only one-shot text-prompt detection.

## Service

- `/yoloe/detect_prompt`
- Type: `yoloe_detection_interfaces/srv/DetectObjectPrompt`

Request:

```text
string prompt_text   # e.g. "bottle" or "cup,bottle"
bool save_image
string camera_name   # optional: "camera0" or "gripper_camera"; empty uses server default
```

Response includes arrays with at most one valid detection: the highest-confidence detection
that also has valid depth/TF:

- `detected_classes[]`
- `confidences[]`
- `poses_camera_link[]` with `header.frame_id` set to the configured final output frame
- `tf_child_frames[]` (e.g. `chair_1`)

The service definition still uses arrays, but only index `0` is populated when a valid result
is available. The response also includes `detections_in_frame`, `tf_published_count`,
`saved_image_path`, and `inference_ms`.

Model behavior:

- Default model path is `/home/usern/yoloe-26l-seg.pt`.
- Bag-specific model path is `/home/usern/Kevin_yolo/best_latest.pt`.
- Bag-only prompts such as `bag`, `paper bag`, and `brown paper bag` temporarily switch to
  the bag-specific model for that request, then unload it after detection.
- All other prompts stay on the default YOLOE model.
- Fixed-class YOLO/segmentation checkpoints use `prompt_text` only as a class-name filter
  against the checkpoint's built-in labels.

## Camera topics (default)

- Default request fallback: `camera0`
- `camera0` color: `/camera0/color/image_raw`
- `camera0` depth: `/camera0/realsense_splitter_node/output/depth`
- `camera0` camera info: `/camera0/color/camera_info`
- `gripper_camera` color: `/gripper_camera/color/image_raw`
- `gripper_camera` depth: `/gripper_camera/depth/image_raw`
- `gripper_camera` camera info: `/gripper_camera/color/camera_info`
- Final published pose/TF frame: `base_link`

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
source /opt/ros/humble/setup.bash
source /home/usern/robocup_ws/install/setup.bash
ros2 launch yoloe_detection_service yoloe_detection_service.launch.py
```

Use another venv without mixing packages by overriding `python_site_packages`:

```bash
ros2 launch yoloe_detection_service yoloe_detection_service.launch.py \
  python_site_packages:=/home/usern/yoloe-venv/lib/python3.10/site-packages
```

## Call

```bash
source /opt/ros/humble/setup.bash
source /home/usern/robocup_ws/install/setup.bash
ros2 service call /yoloe/detect_prompt yoloe_detection_interfaces/srv/DetectObjectPrompt "{prompt_text: 'bottle', save_image: true, camera_name: 'camera0'}"
```

Use the gripper camera stream:

```bash
ros2 service call /yoloe/detect_prompt yoloe_detection_interfaces/srv/DetectObjectPrompt "{prompt_text: 'bottle', save_image: true, camera_name: 'gripper_camera'}"
```

Tracking service `/yoloe/set_tracking` is now provided by the `deepstream_people_tracking` package.

## Pointed-Object Service (Hand Pointing + YOLOE Prompt)

- `/yoloe/detect_pointed_prompt`
- Type: `yoloe_detection_interfaces/srv/DetectObjectPrompt`

Behavior:

- One inference per service request.
- Uses MediaPipe hand landmarks to detect left/right pointing gesture.
- Uses the paper bag model `/home/usern/Kevin_yolo/best_latest.pt`.
- Accepts bag-only prompts such as `bag`, `paper bag`, or `brown paper bag`.
- Returns only the object aligned with pointing ray (single centroid pose in `poses_camera_link[0]`).
- Republishes the last successful TF continuously until the same child frame is updated by a
  newer successful detection.
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
ros2 service call /yoloe/detect_pointed_prompt yoloe_detection_interfaces/srv/DetectObjectPrompt "{prompt_text: 'brown paper bag', save_image: true}"
```

## VLM Alternative Pointed Service (YOLOE + Ollama VLM, VLM-Only Decision)

- `/yoloe/detect_pointed_prompt_vlm`
- Type: `yoloe_detection_interfaces/srv/DetectObjectPrompt`

Behavior:

- Supports `pure_vlm_mode` (default `true`) to ground objects directly with VLM JSON bbox/point output.
- When `pure_vlm_mode:=false`, runs YOLOE to produce candidate boxes and uses VLM to select candidate ID.
- Sends a query image to local Ollama VLM (`qwen3.5:9b` by default).
- Includes VLM response hardening: retries with higher `num_predict` and optional parsing from VLM `thinking` field when `content` is empty.
- Returns centroid pose and TF in `camera0_link` by default.
- Republishes detected object TF on `/tf` for a configurable TTL (`tf_ttl_sec`, default 60s), then stops.
- Publishes VLM query/selection visualization on `/yoloe/vlm_pointing_query_image`.

Run:

```bash
source /opt/ros/humble/setup.bash
source /home/usern/robocup_ws/install/setup.bash
ros2 launch yoloe_detection_service yoloe_vlm_pointed_detection_service.launch.py show_ui:=false
```

Stability-first tuning (pure VLM):

```bash
ros2 launch yoloe_detection_service yoloe_vlm_pointed_detection_service.launch.py \
  show_ui:=false \
  pure_vlm_mode:=true \
  vlm_model:=qwen3.5:9b \
  vote_frames:=3 \
  vlm_num_predict:=96 \
  vlm_retry_num_predict:=192 \
  vlm_max_retries:=1 \
  vlm_max_candidates:=8 \
  vlm_min_bbox_area_ratio:=0.008 \
  vlm_small_object_max_bbox_area_ratio:=0.03 \
  vlm_image_max_edge:=960
```

Call:

```bash
source /opt/ros/humble/setup.bash
source /home/usern/robocup_ws/install/setup.bash
ros2 service call /yoloe/detect_pointed_prompt_vlm yoloe_detection_interfaces/srv/DetectObjectPrompt "{prompt_text: 'bottle', save_image: true}"
```

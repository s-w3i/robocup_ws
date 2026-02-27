# DeepSORT People Follow (ROS2)

This package provides a DeepSORT-style people tracking pipeline with depth fusion and follow-target output, without DeepStream.

## Interfaces

- Service: `/yoloe/set_tracking` (`yoloe_detection_interfaces/srv/SetTracking`)
- Publishes:
  - `/people_tracks_2d` (`yoloe_detection_interfaces/msg/PeopleTrack2DArray`)
  - `/people_tracks_3d` (`yoloe_detection_interfaces/msg/PeopleTrack3DArray`)
  - `/yoloe/tracking_detections` (`yoloe_detection_interfaces/msg/Detection3DArray`)
  - `/people/follow_target_pose` (`geometry_msgs/PoseStamped`)
- TF:
  - `person_id_<track_id>`
  - `follow_target`

## Runtime dependencies

Install once in your venv:

```bash
python3 -m venv /home/usern/follow-venv
source /home/usern/follow-venv/bin/activate
pip install -U pip setuptools wheel
pip install "numpy<2" \
  "https://pypi.jetson-ai-lab.io/jp6/cu126/+f/62a/1beee9f2f1470/torch-2.8.0-cp310-cp310-linux_aarch64.whl" \
  nvidia-cuda-cupti-cu12==12.6.68 \
  nvidia-cusparselt-cu12==0.8.1 \
  nvidia-nvtx-cu12==12.6.68 \
  torchvision==0.23.0 \
  torchaudio==2.8.0 \
  "opencv-python<4.11" \
  scipy \
  deep-sort-realtime==1.3.2 \
  ultralytics==8.4.14 \
  ultralytics-thop \
  polars \
  Cython \
  gdown \
  tensorboard
pip install --no-build-isolation git+https://github.com/KaiyangZhou/deep-person-reid.git
```

Ultralytics + torch are required for the detector model.

## Build

```bash
cd /home/usern/robocup_ws
source /home/usern/follow-venv/bin/activate
source /opt/ros/humble/setup.bash
colcon build --packages-select yoloe_detection_interfaces deepsort_people_follow --symlink-install
```

## Run

```bash
cd /home/usern/robocup_ws
./run_deepsort_people_follow_in_venv.sh
```

Example override with ROS parameters:

```bash
./run_deepsort_people_follow_in_venv.sh --ros-args \
  -p model_path:=/home/usern/robocup_ws/yolo11s.pt \
  -p color_topic:=/camera/color/image_raw \
  -p depth_topic:=/camera/aligned_depth_to_color/image_raw \
  -p enable_ui:=true
```

## Control tracking

```bash
source /opt/ros/humble/setup.bash
source /home/usern/robocup_ws/install/setup.bash
ros2 service call /yoloe/set_tracking yoloe_detection_interfaces/srv/SetTracking "{enable: true, save_image: false, rate_hz: 0.0}"
```

Stop:

```bash
ros2 service call /yoloe/set_tracking yoloe_detection_interfaces/srv/SetTracking "{enable: false, save_image: false, rate_hz: 0.0}"
```

## Notes

- Tracker runs in human-only mode (`person`) for follow behavior.
- ReID backend is selectable with `reid_embedder:=auto|torchreid|clip|mobilenet|hsv` (`auto` keeps legacy behavior).
- Default ReID embedder is `torchreid` with `osnet_ain_x0_5`.
- For non-default `torchreid_model_name`, provide `embedder_weights_path` to use true ReID weights (otherwise torchreid loads ImageNet-pretrained weights).
- If selected embedder init fails, node can auto-fallback to lightweight HSV embedding (`fallback_to_hsv_embedder:=true`).
- `SetTracking` service no longer includes prompt fields; response returns a single `tracking_class`.
- `tracking_class` and `enable_open_vocab_prompt` parameters are retained only for backward compatibility and are ignored.
- `prefer_ultralytics_torch_nms:=true` keeps NMS on CUDA tensors using Ultralytics TorchNMS when torchvision CUDA NMS kernels are unavailable (recommended on Jetson).
- This package is intended to run in `/home/usern/follow-venv` to avoid dependency clashes with Whisper/TTS/Ollama environments.
- Live tracking window:
  - `enable_ui:=true` shows an OpenCV window with live camera preview; once tracking is enabled it overlays YOLO detections and DeepSORT IDs.
  - Overlay colors: blue=`YOLO detections`, green/orange/red=`tracked IDs/follow target`.
  - `ui_window_name` sets the window title (default: `DeepSORT Tracking`).
  - `ui_show_depth_text` toggles per-track depth labels.
  - Requires a desktop/X11/Wayland display; over SSH use X forwarding (`ssh -X`) or run without UI.
- For crowded scenes, tune:
  - `max_cosine_distance`
  - `max_age`
  - `n_init`
  - `reid_embedder` + model choice (`torchreid` or `clip`)
  - `torchreid_model_name` (`osnet_ain_x0_5` faster, `osnet_ain_x1_0` stronger baseline)
  - `embedder_weights_path` (required for stronger non-default torchreid checkpoints)
  - `reacquire_max_dist_m`
  - `reacquire_max_vel_mps`

Example stronger ReID configs:

```bash
# Option 1: stronger CLIP appearance features (slower, but often better in crowds)
./run_deepsort_people_follow_in_venv.sh --ros-args \
  -p reid_embedder:=clip \
  -p clip_model_name:=ViT-B/16 \
  -p max_cosine_distance:=0.18

# Option 2: torchreid with explicit checkpoint path
./run_deepsort_people_follow_in_venv.sh --ros-args \
  -p reid_embedder:=torchreid \
  -p torchreid_model_name:=osnet_ibn_x1_0 \
  -p embedder_weights_path:=/home/usern/models/osnet_ibn_x1_0_msmt17.pth.tar
```

# coqui_tts_service

ROS2 Coqui TTS nodes:

- Unified talking-face node:
  - Service: `/coqui_tts/synthesize` (`coqui_tts_interfaces/srv/SynthesizeSpeech`)
  - Action: `/coqui_tts/speak` (`coqui_tts_interfaces/action/SpeakText`)
- Legacy service-only node: `coqui_tts_service_node` (optional)

## Build

```bash
cd /home/usern/robocup_ws
colcon build --packages-select coqui_tts_interfaces coqui_tts_service
```

## Run Unified Node (Recommended)

```bash
source /home/usern/robocup_ws/install/setup.bash
ros2 run coqui_tts_service coqui_talking_face_action_node
```

## Auto-loading Coqui venv modules

Both nodes default to:

- `extra_site_packages=/home/usern/coqui-venv/lib/python3.10/site-packages`
- `isolate_site_packages=true`

This makes `ros2 run` work directly from `robocup_ws` without activating `coqui-venv`.

Override if needed:

```bash
ros2 run coqui_tts_service coqui_tts_service_node --ros-args \
  -p extra_site_packages:=/custom/venv/lib/python3.10/site-packages \
  -p isolate_site_packages:=true
```

## Test Synthesis

```bash
ros2 service call /coqui_tts/synthesize coqui_tts_interfaces/srv/SynthesizeSpeech \
"{text: 'hello from ros2', out_path: ''}"
```

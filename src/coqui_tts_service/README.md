# coqui_tts_service

ROS2 Coqui TTS nodes:

- Unified talking-face node:
  - Service: `/coqui_tts/synthesize` (`coqui_tts_interfaces/srv/SynthesizeSpeech`)
  - Action: `/coqui_tts/speak` (`coqui_tts_interfaces/action/SpeakText`)
- Robot status node:
  - Service: `/robot_status` (`coqui_tts_interfaces/srv/RobotStatus`)
  - Topic: `/robot_status` (`std_msgs/msg/String`)
  - Topic: `/awake` (`std_msgs/msg/Bool`) publishes `false` when status is set to `sleep`
  - Default status: `sleep`
- Whisper command node:
  - Service: `/get_command` (`std_srvs/srv/Trigger`)
  - Subscribes: `/robot_status` (`std_msgs/msg/String`)
  - Publishes: `/awake` (`std_msgs/msg/Bool`) as `true` when awake-word is detected
  - Transcription language is fixed to English (`en`)
  - Wake-word mode only when robot status is `sleep`
  - On wake word (`hi eva` by default): requests robot status `idle`
  - On `/get_command`: sets status to `listening`, returns transcript in `response.message`, then sets status back to `idle`
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

Tune face emotion-change speed:

```bash
ros2 run coqui_tts_service coqui_talking_face_action_node --ros-args \
  -p face_emotion_speed_scale:=1.5
```

`face_emotion_speed_scale` behavior:
- `1.0` default
- `>1.0` faster emotion changes
- `<1.0` slower emotion changes

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

## Robot Status Control

Run status node:

```bash
source /home/usern/robocup_ws/install/setup.bash
ros2 run coqui_tts_service robot_status_node
```

Run whisper command node:

```bash
source /home/usern/robocup_ws/install/setup.bash
ros2 run coqui_tts_service whisper_command_node
```

Override wake word / service timeout:

```bash
ros2 run coqui_tts_service whisper_command_node --ros-args \
  -p awake_word:="hi eva" \
  -p get_command_timeout_sec:=12.0 \
  -p whisper_device:=auto \
  -p model:=base
```

Set status:

```bash
ros2 service call /robot_status coqui_tts_interfaces/srv/RobotStatus "{status: 'sleep'}"
ros2 service call /robot_status coqui_tts_interfaces/srv/RobotStatus "{status: 'listening'}"
ros2 service call /robot_status coqui_tts_interfaces/srv/RobotStatus "{status: 'idle'}"
ros2 service call /robot_status coqui_tts_interfaces/srv/RobotStatus "{status: 'operating'}"
```

Toggle (cycle) status:

```bash
ros2 service call /robot_status coqui_tts_interfaces/srv/RobotStatus "{status: ''}"
```

Request one speech command transcript:

```bash
ros2 service call /get_command std_srvs/srv/Trigger "{}"
```

`/get_command` response:
- `success=true` and `message="<transcribed text>"` when transcription succeeds
- `success=false` and `message="<reason>"` on timeout/error

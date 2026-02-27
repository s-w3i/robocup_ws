# speech_transcription_service

ROS2 node that provides:

- `/request_speech` (`std_srvs/srv/Trigger`): starts a speech capture session and returns final transcription when silence is detected.
- `/awake` (`std_msgs/msg/Bool`): publishes `true` when wake phrase is detected (default: `hi eva`).

## Build

```bash
cd /home/usern/robocup_ws
colcon build --packages-select speech_transcription_service
```

## Run

```bash
source /home/usern/robocup_ws/install/setup.bash
ros2 run speech_transcription_service speech_transcription_node
```

Enable transcription debug logs:

```bash
source /home/usern/robocup_ws/install/setup.bash
ros2 run speech_transcription_service speech_transcription_node --ros-args --log-level speech_transcription_node:=debug
```

or

```bash
source /home/usern/robocup_ws/install/setup.bash
ros2 launch speech_transcription_service speech_transcription_service.launch.py
```

## Launch Full Voice Stack

Launches:
- `coqui_talking_face_action_node` (provides both `/coqui_tts/speak` and `/coqui_tts/synthesize`)
- `speech_transcription_node`

```bash
source /home/usern/robocup_ws/install/setup.bash
ros2 launch speech_transcription_service voice_interaction_stack.launch.py
```

Optional args:

```bash
ros2 launch speech_transcription_service voice_interaction_stack.launch.py \
  log_level:=debug \
  wake_phrase:="hi eva" \
  warmup_enabled:=true \
  warmup_text:="service warmup" \
  extra_site_packages:=/home/usern/coqui-venv/lib/python3.10/site-packages \
  isolate_site_packages:=true
```

Notes:
- `/coqui_tts/synthesize` only generates WAV and returns path (no playback).
- `/coqui_tts/speak` action generates and plays audio via talking-face node.

## Request Speech

```bash
ros2 service call /request_speech std_srvs/srv/Trigger "{}"
```

The transcription text is returned in `response.message`.

## Echo Awake Topic

```bash
ros2 topic echo /awake
```

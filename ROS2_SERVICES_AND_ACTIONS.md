# ROS 2 Services and Actions in This Workspace

This document summarizes ROS 2 service/action servers implemented under `src/` and what they do.

Note: endpoint names below are default parameter values and can be remapped at launch/runtime.

## Service Servers

| Default endpoint | Type | Implemented in | Function |
| --- | --- | --- | --- |
| `/coqui_tts/synthesize` | `coqui_tts_interfaces/srv/SynthesizeSpeech` | `coqui_tts_service/coqui_tts_service_node.py` | Synthesizes text to a WAV file using Coqui TTS and returns status, output path, elapsed time, and device used. |
| `/coqui_tts/synthesize` | `coqui_tts_interfaces/srv/SynthesizeSpeech` | `coqui_tts_service/coqui_talking_face_action_node.py` | Same synthesis service as above, but in the talking-face node. Rejects synthesis while `/coqui_tts/speak` action is active. |
| `/robot_status` | `coqui_tts_interfaces/srv/RobotStatus` | `coqui_tts_service/robot_status_node.py` | Sets or cycles robot status (`sleep`, `listening`, `idle`, `operating`), publishes status topic, and toggles `/awake` when transitioning from/to sleep. |
| `/get_command` | `std_srvs/srv/Trigger` | `coqui_tts_service/whisper_command_node.py` | Starts a one-shot voice command capture, waits for Whisper transcription, and returns recognized text in `response.message` (or timeout/error). |
| `/yoloe/detect_prompt` | `yoloe_detection_interfaces/srv/DetectObjectPrompt` | `yoloe_detection_service/yoloe_detection_service_node.py` | Runs one-shot prompt-based YOLOE detection on latest RGB-D frame; returns detected classes/confidences/3D poses and TF child frames for valid detections. |
| `/yoloe/detect_pointed_prompt` | `yoloe_detection_interfaces/srv/DetectObjectPrompt` | `yoloe_detection_service/yoloe_pointed_detection_service_node.py` | Runs one-shot prompt-based detection and selects the object being pointed at using hand/arm cues plus multi-frame voting; returns one selected object pose/TF. |
| `/yoloe/detect_pointed_prompt_vlm` | `yoloe_detection_interfaces/srv/DetectObjectPrompt` | `yoloe_detection_service/yoloe_vlm_pointed_detection_service_node.py` | Runs one-shot prompt-based detection and uses a VLM (Ollama) to select the pointed object among candidates, with voting and TF publication. |
| `/yoloe/set_tracking` | `yoloe_detection_interfaces/srv/SetTracking` | `deepsort_people_follow/deepsort_people_follow_node.py` | Enables/disables DeepSORT person tracking, sets publish rate/debug-image saving, and reports whether tracking is running. |

## Action Servers

| Default endpoint | Type | Implemented in | Function |
| --- | --- | --- | --- |
| `/coqui_tts/speak` | `coqui_tts_interfaces/action/SpeakText` | `coqui_tts_service/coqui_talking_face_action_node.py` | Accepts text goals, synthesizes speech, plays audio with talking-face animation, publishes stage/progress feedback (`synthesizing`, `playing`), and supports cancellation. |

## Custom Interface Definitions in This Workspace

### Services (`.srv`)

- `coqui_tts_interfaces/srv/SynthesizeSpeech.srv`
  - Request: `text`, `out_path`
  - Response: `success`, `wav_path`, `message`, `elapsed_seconds`, `device_used`
- `coqui_tts_interfaces/srv/RobotStatus.srv`
  - Request: `status`
  - Response: `success`, `status`, `message`
- `yoloe_detection_interfaces/srv/DetectObjectPrompt.srv`
  - Request: `prompt_text`, `save_image`
  - Response: `success`, `message`, `detected_classes`, `confidences`, `poses_camera_link`, `tf_child_frames`, `saved_image_path`, `detections_in_frame`, `tf_published_count`, `inference_ms`
- `yoloe_detection_interfaces/srv/SetTracking.srv`
  - Request: `enable`, `save_image`, `rate_hz`
  - Response: `success`, `message`, `running`, `tracking_class`

### Actions (`.action`)

- `coqui_tts_interfaces/action/SpeakText.action`
  - Goal: `text`
  - Result: `success`, `wav_path`, `message`, `synthesis_seconds`, `playback_seconds`, `device_used`
  - Feedback: `stage`, `progress`

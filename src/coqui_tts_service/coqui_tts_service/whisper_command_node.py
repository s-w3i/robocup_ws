#!/usr/bin/env python3

from __future__ import annotations

import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

DEFAULT_COQUI_SITE_PACKAGES = "/home/usern/coqui-venv/lib/python3.10/site-packages"
SYSTEM_SITE_PATH_PREFIXES = (
    "/usr/lib/python3/dist-packages",
    "/usr/local/lib/python3.10/dist-packages",
)
VALID_STATUSES = ("sleep", "listening", "idle", "operating")


def prepend_site_packages(path: str) -> bool:
    p = Path(path).expanduser().resolve()
    if not p.is_dir():
        return False
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)
    return True


def activate_coqui_site_packages(path: str, isolate: bool) -> tuple[bool, int]:
    added = prepend_site_packages(path)
    removed = 0
    if isolate:
        filtered = []
        for entry in sys.path:
            if entry.startswith(SYSTEM_SITE_PATH_PREFIXES):
                removed += 1
                continue
            filtered.append(entry)
        sys.path[:] = filtered
    return added, removed


prepend_site_packages(os.environ.get("COQUI_VENV_SITE_PACKAGES", DEFAULT_COQUI_SITE_PACKAGES))

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger

from coqui_tts_interfaces.srv import RobotStatus


@dataclass
class AudioSegment:
    start_time: float
    end_time: float
    samples: np.ndarray

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)


@dataclass
class PendingCommand:
    event: threading.Event
    text: str = ""


class EnergySegmenter:
    def __init__(
        self,
        rate: int,
        frame_ms: int,
        silence_seconds: float,
        pre_roll_seconds: float,
        min_speech_seconds: float,
        trigger_frames: int,
        energy_threshold: float,
        energy_multiplier: float,
        min_rms: float,
        calibration_seconds: float,
    ):
        self.rate = rate
        self.frame_ms = frame_ms
        self.frame_s = frame_ms / 1000.0
        self.silence_frames = max(1, int(round(silence_seconds / self.frame_s)))
        self.min_speech_seconds = max(0.05, min_speech_seconds)
        self.trigger_frames = max(1, trigger_frames)

        self.fixed_threshold = energy_threshold if energy_threshold > 0 else None
        self.energy_multiplier = max(1.0, energy_multiplier)
        self.min_rms = max(1.0, min_rms)
        self.pre_roll = deque(maxlen=max(1, int(round(pre_roll_seconds / self.frame_s))))
        self.calibration_target_frames = max(0, int(round(calibration_seconds / self.frame_s)))
        self.calibration_values: list[float] = []
        self.noise_rms: float | None = None
        self.calibration_done = self.fixed_threshold is not None or self.calibration_target_frames == 0

        self.in_speech = False
        self.speech_run = 0
        self.silence_run = 0
        self.frames: list[np.ndarray] = []
        self.segment_start_time = 0.0

    @staticmethod
    def rms(frame: np.ndarray) -> float:
        x = frame.astype(np.float32)
        return float(np.sqrt(np.mean(x * x)))

    def current_threshold(self) -> float:
        if self.fixed_threshold is not None:
            return self.fixed_threshold
        base = self.noise_rms if self.noise_rms is not None else self.min_rms
        return max(self.min_rms, base * self.energy_multiplier)

    def _update_noise_floor(self, rms: float) -> None:
        if self.fixed_threshold is not None:
            return
        if self.noise_rms is None:
            self.noise_rms = max(1.0, rms)
            return
        self.noise_rms = 0.98 * self.noise_rms + 0.02 * rms

    def _maybe_finish_calibration(self) -> bool:
        if self.fixed_threshold is not None:
            self.calibration_done = True
            return True
        if self.calibration_target_frames == 0:
            if self.noise_rms is None:
                self.noise_rms = self.min_rms
            self.calibration_done = True
            return True
        if len(self.calibration_values) < self.calibration_target_frames:
            return False
        avg = float(np.mean(self.calibration_values))
        self.noise_rms = max(1.0, avg)
        self.calibration_done = True
        return True

    def process(
        self, frame: np.ndarray, frame_end_time: float
    ) -> tuple[bool, AudioSegment | None, float, float]:
        frame_rms = self.rms(frame)
        threshold = self.current_threshold()

        if not self._maybe_finish_calibration() and not self.in_speech:
            self.calibration_values.append(frame_rms)
            self.pre_roll.append(frame.copy())
            return False, None, frame_rms, threshold

        if not self.in_speech and frame_rms < threshold:
            self._update_noise_floor(frame_rms)
            threshold = self.current_threshold()

        is_speech = frame_rms >= threshold
        started = False
        segment = None

        if not self.in_speech:
            self.pre_roll.append(frame.copy())
            if is_speech:
                self.speech_run += 1
            else:
                self.speech_run = 0

            if self.speech_run >= self.trigger_frames:
                self.in_speech = True
                started = True
                self.silence_run = 0
                self.frames = list(self.pre_roll)
                self.segment_start_time = frame_end_time - len(self.frames) * self.frame_s
                self.speech_run = 0
        else:
            self.frames.append(frame.copy())
            if is_speech:
                self.silence_run = 0
            else:
                self.silence_run += 1

            if self.silence_run >= self.silence_frames:
                keep_frames = (
                    self.frames[:-self.silence_run]
                    if self.silence_run < len(self.frames)
                    else []
                )
                segment_end_time = frame_end_time - self.silence_run * self.frame_s
                duration = len(keep_frames) * self.frame_s
                if keep_frames and duration >= self.min_speech_seconds:
                    segment = AudioSegment(
                        start_time=self.segment_start_time,
                        end_time=segment_end_time,
                        samples=np.concatenate(keep_frames),
                    )
                self._reset_after_segment()

        return started, segment, frame_rms, threshold

    def flush(self, now: float) -> AudioSegment | None:
        if not self.in_speech or not self.frames:
            return None
        duration = len(self.frames) * self.frame_s
        if duration < self.min_speech_seconds:
            self._reset_after_segment()
            return None
        segment = AudioSegment(
            start_time=self.segment_start_time,
            end_time=now,
            samples=np.concatenate(self.frames),
        )
        self._reset_after_segment()
        return segment

    def reset_activity(self) -> None:
        self._reset_after_segment()

    def _reset_after_segment(self) -> None:
        self.in_speech = False
        self.speech_run = 0
        self.silence_run = 0
        self.frames = []
        self.pre_roll.clear()


class WhisperTranscriber(threading.Thread):
    def __init__(
        self,
        *,
        segment_queue: queue.Queue[AudioSegment | None],
        model_name: str,
        language: str | None,
        task: str,
        whisper_device: str,
        no_fp16: bool,
        model_dir: str | None,
        result_cb,
        startup_cb,
    ):
        super().__init__(daemon=True)
        self._segment_queue = segment_queue
        self._model_name = model_name
        self._language = language
        self._task = task
        self._whisper_device = whisper_device
        self._no_fp16 = no_fp16
        self._model_dir = model_dir
        self._result_cb = result_cb
        self._startup_cb = startup_cb
        self._stop_requested = threading.Event()
        self._model = None
        self._device = "cpu"
        self._fp16 = False
        self.startup_error: str | None = None
        self.ready = threading.Event()

    def stop(self) -> None:
        self._stop_requested.set()
        try:
            self._segment_queue.put_nowait(None)
        except queue.Full:
            pass

    def _transcribe(self, segment: AudioSegment) -> str:
        audio = segment.samples.astype(np.float32) / 32768.0
        try:
            result = self._model.transcribe(
                audio,
                language=self._language,
                task=self._task,
                fp16=self._fp16,
                temperature=0.0,
                condition_on_previous_text=False,
                verbose=False,
            )
            return str(result.get("text", "")).strip()
        except TypeError:
            result = self._model.transcribe(
                audio,
                language=self._language,
                task=self._task,
                fp16=self._fp16,
            )
            return str(result.get("text", "")).strip()

    def run(self) -> None:
        try:
            import torch
            import whisper

            if self._whisper_device == "auto":
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self._device = self._whisper_device

            self._fp16 = self._device == "cuda" and not self._no_fp16
            self._model = whisper.load_model(
                self._model_name,
                device=self._device,
                download_root=self._model_dir,
            )
        except Exception as exc:  # pragma: no cover
            self.startup_error = str(exc)
            self.ready.set()
            self._startup_cb(False, self._device, self._fp16, self.startup_error)
            return

        self.ready.set()
        self._startup_cb(True, self._device, self._fp16, "")

        while not self._stop_requested.is_set():
            try:
                segment = self._segment_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if segment is None:
                break
            try:
                text = self._transcribe(segment)
            except Exception as exc:  # pragma: no cover
                self._result_cb(segment, "", str(exc))
                continue
            self._result_cb(segment, text, "")


def read_exact(pipe, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        data = pipe.read(remaining)
        if not data:
            break
        chunks.append(data)
        remaining -= len(data)
    return b"".join(chunks)


class WhisperCommandNode(Node):
    def __init__(self) -> None:
        super().__init__("whisper_command_node")

        self.declare_parameter("status_topic", "/robot_status")
        self.declare_parameter("awake_topic", "/awake")
        self.declare_parameter("status_service", "/robot_status")
        self.declare_parameter("get_command_service", "/get_command")
        self.declare_parameter("awake_word", "hi eva")
        self.declare_parameter("get_command_timeout_sec", 12.0)
        self.declare_parameter("audio_device", "default")
        self.declare_parameter("rate", 16000)
        self.declare_parameter("frame_ms", 30)
        self.declare_parameter("silence_seconds", 1.0)
        self.declare_parameter("pre_roll_seconds", 0.3)
        self.declare_parameter("min_speech_seconds", 0.4)
        self.declare_parameter("trigger_frames", 2)
        self.declare_parameter("energy_threshold", 0.0)
        self.declare_parameter("energy_multiplier", 1.0)
        self.declare_parameter("min_rms", 120.0)
        self.declare_parameter("calibration_seconds", 5.0)
        self.declare_parameter("model", "small")
        self.declare_parameter("language", "en")
        self.declare_parameter("task", "transcribe")
        self.declare_parameter("whisper_device", "auto")
        self.declare_parameter("no_fp16", False)
        self.declare_parameter("model_dir", "")
        self.declare_parameter("max_queue", 8)
        self.declare_parameter("extra_site_packages", DEFAULT_COQUI_SITE_PACKAGES)
        self.declare_parameter("isolate_site_packages", True)
        self.declare_parameter("log_transcripts", True)

        self.status_topic = str(self.get_parameter("status_topic").value)
        self.awake_topic = str(self.get_parameter("awake_topic").value)
        self.status_service = str(self.get_parameter("status_service").value)
        self.get_command_service = str(self.get_parameter("get_command_service").value)
        self.awake_word = str(self.get_parameter("awake_word").value).strip().lower()
        self.get_command_timeout_sec = float(self.get_parameter("get_command_timeout_sec").value)
        self.audio_device = str(self.get_parameter("audio_device").value)
        self.rate = int(self.get_parameter("rate").value)
        self.frame_ms = int(self.get_parameter("frame_ms").value)
        self.silence_seconds = float(self.get_parameter("silence_seconds").value)
        self.pre_roll_seconds = float(self.get_parameter("pre_roll_seconds").value)
        self.min_speech_seconds = float(self.get_parameter("min_speech_seconds").value)
        self.trigger_frames = int(self.get_parameter("trigger_frames").value)
        self.energy_threshold = float(self.get_parameter("energy_threshold").value)
        self.energy_multiplier = float(self.get_parameter("energy_multiplier").value)
        self.min_rms = float(self.get_parameter("min_rms").value)
        self.calibration_seconds = float(self.get_parameter("calibration_seconds").value)
        self.model_name = str(self.get_parameter("model").value)
        self.language = str(self.get_parameter("language").value).strip() or None
        self.task = str(self.get_parameter("task").value).strip().lower()
        self.whisper_device = str(self.get_parameter("whisper_device").value).strip().lower()
        self.no_fp16 = bool(self.get_parameter("no_fp16").value)
        self.model_dir = str(self.get_parameter("model_dir").value).strip() or None
        self.max_queue = int(self.get_parameter("max_queue").value)
        self.extra_site_packages = str(self.get_parameter("extra_site_packages").value)
        self.isolate_site_packages = bool(self.get_parameter("isolate_site_packages").value)
        self.log_transcripts = bool(self.get_parameter("log_transcripts").value)

        added, removed = activate_coqui_site_packages(
            self.extra_site_packages, self.isolate_site_packages
        )
        self._build_runtime_env(self.extra_site_packages)

        self._robot_status = "sleep"
        self._wake_word_armed = True
        self._status_lock = threading.Lock()
        self._pending_command: PendingCommand | None = None
        self._pending_lock = threading.Lock()
        self._stop_event = threading.Event()

        status_qos = QoSProfile(depth=1)
        status_qos.reliability = QoSReliabilityPolicy.RELIABLE
        status_qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        self._status_sub = self.create_subscription(
            String,
            self.status_topic,
            self._robot_status_callback,
            status_qos,
        )
        self._awake_pub = self.create_publisher(Bool, self.awake_topic, status_qos)

        self._status_client = self.create_client(RobotStatus, self.status_service)
        self._status_wait_warned = False

        self._command_srv = self.create_service(
            Trigger,
            self.get_command_service,
            self._handle_get_command,
        )

        if self.rate <= 0:
            raise ValueError("rate must be > 0")
        if self.frame_ms <= 0:
            raise ValueError("frame_ms must be > 0")
        if (self.rate * self.frame_ms) % 1000 != 0:
            raise ValueError("rate * frame_ms must be divisible by 1000")
        if self.max_queue <= 0:
            raise ValueError("max_queue must be > 0")
        if self.task not in ("transcribe", "translate"):
            raise ValueError("task must be transcribe or translate")
        if self.whisper_device not in ("auto", "cpu", "cuda"):
            raise ValueError("whisper_device must be auto/cpu/cuda")

        self._segmenter = EnergySegmenter(
            rate=self.rate,
            frame_ms=self.frame_ms,
            silence_seconds=self.silence_seconds,
            pre_roll_seconds=self.pre_roll_seconds,
            min_speech_seconds=self.min_speech_seconds,
            trigger_frames=self.trigger_frames,
            energy_threshold=self.energy_threshold,
            energy_multiplier=self.energy_multiplier,
            min_rms=self.min_rms,
            calibration_seconds=self.calibration_seconds,
        )
        self._frame_samples = self.rate * self.frame_ms // 1000
        self._frame_bytes = self._frame_samples * 2
        self._segments: queue.Queue[AudioSegment | None] = queue.Queue(maxsize=self.max_queue)
        self._transcriber = WhisperTranscriber(
            segment_queue=self._segments,
            model_name=self.model_name,
            language=self.language,
            task=self.task,
            whisper_device=self.whisper_device,
            no_fp16=self.no_fp16,
            model_dir=self.model_dir,
            result_cb=self._on_transcription_result,
            startup_cb=self._on_transcriber_startup,
        )
        self._transcriber.start()

        if shutil.which("arecord") is None:
            self.get_logger().error("arecord not found. Install ALSA utilities.")
            self._audio_thread = None
        else:
            self._audio_thread = threading.Thread(target=self._audio_loop, daemon=True)
            self._audio_thread.start()

        self.get_logger().info(
            f"Whisper command node ready. get_command={self.get_command_service} "
            f"awake_word='{self.awake_word}' awake_topic={self.awake_topic}"
        )
        self.get_logger().info(
            f"extra_site_packages={self.extra_site_packages} isolate_site_packages={self.isolate_site_packages} "
            f"(added={added}, removed={removed})"
        )

    def _build_runtime_env(self, site_packages: str) -> None:
        lib_paths: list[str] = []
        nvidia_root = Path(site_packages).expanduser().resolve() / "nvidia"
        if nvidia_root.is_dir():
            for lib_dir in nvidia_root.glob("*/lib"):
                lib_paths.append(str(lib_dir))
        for path in (
            "/usr/local/cuda/targets/aarch64-linux/lib",
            "/usr/local/cuda-12.6/targets/aarch64-linux/lib",
            "/lib/aarch64-linux-gnu",
            "/usr/lib/aarch64-linux-gnu",
        ):
            if Path(path).is_dir():
                lib_paths.append(path)
        current = os.environ.get("LD_LIBRARY_PATH", "")
        merged = [p for p in lib_paths if p]
        if current:
            merged.append(current)
        os.environ["LD_LIBRARY_PATH"] = ":".join(merged)

    @staticmethod
    def _normalize_text(text: str) -> str:
        cleaned = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
        return " ".join(cleaned.split())

    def _robot_status_callback(self, msg: String) -> None:
        status = str(msg.data).strip().lower()
        if status not in VALID_STATUSES:
            self.get_logger().warn(
                f"Ignoring invalid robot status '{status}' on {self.status_topic}."
            )
            return
        with self._status_lock:
            previous = self._robot_status
            self._robot_status = status
            self._wake_word_armed = status == "sleep"

        if status != previous:
            self.get_logger().info(f"Robot status changed to '{status}'.")
            if status == "sleep":
                self.get_logger().info("Wake-word mode enabled (waiting for awake word).")

    def _current_status(self) -> str:
        with self._status_lock:
            return self._robot_status

    def _is_wake_word_armed(self) -> bool:
        with self._status_lock:
            return self._wake_word_armed and self._robot_status == "sleep"

    def _set_robot_status_async(self, target: str) -> None:
        target = str(target).strip().lower()
        if target not in VALID_STATUSES:
            return
        if not self._status_client.service_is_ready():
            if not self._status_client.wait_for_service(timeout_sec=0.2):
                if not self._status_wait_warned:
                    self.get_logger().warn(
                        f"Status service {self.status_service} not ready; cannot set '{target}'."
                    )
                    self._status_wait_warned = True
                return
            self._status_wait_warned = False

        req = RobotStatus.Request()
        req.status = target
        try:
            future = self._status_client.call_async(req)
        except Exception as exc:  # pragma: no cover
            self.get_logger().error(f"Failed to send status request '{target}': {exc}")
            return

        def _done(fut):
            try:
                result = fut.result()
            except Exception as exc:  # pragma: no cover
                self.get_logger().error(f"Status request '{target}' failed: {exc}")
                return
            if result is None:
                self.get_logger().error(f"Status request '{target}' returned no result.")
                return
            if not result.success:
                self.get_logger().warn(
                    f"Status request '{target}' rejected: {result.message}"
                )

        future.add_done_callback(_done)

    def _publish_awake(self, value: bool) -> None:
        msg = Bool()
        msg.data = bool(value)
        self._awake_pub.publish(msg)

    def _should_listen(self) -> bool:
        with self._pending_lock:
            has_pending = self._pending_command is not None
        status = self._current_status()
        return has_pending or status == "sleep"

    def _enqueue_segment(self, segment: AudioSegment) -> None:
        try:
            self._segments.put(segment, timeout=0.1)
        except queue.Full:
            self.get_logger().warn("Dropping speech segment: transcription queue full.")

    def _audio_loop(self) -> None:
        cmd = [
            "arecord",
            "-D",
            self.audio_device,
            "-f",
            "S16_LE",
            "-c",
            "1",
            "-r",
            str(self.rate),
            "-t",
            "raw",
            "-q",
        ]

        announced_calibration = False
        was_listening = False
        while not self._stop_event.is_set():
            proc: subprocess.Popen[bytes] | None = None
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=0,
                )
                if proc.stdout is None:
                    self.get_logger().error("Failed to open arecord stdout.")
                    return

                while not self._stop_event.is_set():
                    chunk = read_exact(proc.stdout, self._frame_bytes)
                    if len(chunk) < self._frame_bytes:
                        break
                    now = time.time()
                    frame = np.frombuffer(chunk, dtype=np.int16).copy()
                    should_listen = self._should_listen()
                    if not should_listen:
                        if was_listening:
                            self._segmenter.reset_activity()
                            was_listening = False
                        continue
                    was_listening = True

                    started, segment, _, _ = self._segmenter.process(frame, now)

                    if (
                        self.energy_threshold <= 0.0
                        and self._segmenter.calibration_done
                        and not announced_calibration
                    ):
                        noise = self._segmenter.noise_rms if self._segmenter.noise_rms is not None else 0.0
                        threshold = self._segmenter.current_threshold()
                        self.get_logger().info(
                            f"Whisper VAD calibrated (noise_rms={noise:.1f}, threshold={threshold:.1f})"
                        )
                        announced_calibration = True

                    if started:
                        self.get_logger().debug("Speech detected.")
                    if segment is not None:
                        self._enqueue_segment(segment)
            except Exception as exc:  # pragma: no cover
                self.get_logger().error(f"Audio capture loop error: {exc}")
            finally:
                flushed = self._segmenter.flush(time.time())
                if flushed is not None and self._should_listen():
                    self._enqueue_segment(flushed)

                if proc is not None:
                    if proc.poll() is None:
                        proc.terminate()
                        try:
                            proc.wait(timeout=1.0)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                    if proc.stderr is not None:
                        err = proc.stderr.read().decode(errors="ignore").strip()
                        if err:
                            self.get_logger().warn(f"arecord: {err}")

            if not self._stop_event.is_set():
                self.get_logger().warn("Audio stream stopped. Restarting arecord in 1s.")
                time.sleep(1.0)

    def _on_transcriber_startup(
        self, ok: bool, device: str, fp16: bool, error_text: str
    ) -> None:
        if ok:
            self.get_logger().info(
                f"Whisper model loaded: {self.model_name} on {device} (fp16={'on' if fp16 else 'off'})"
            )
            return
        self.get_logger().error(
            "Whisper initialization failed: "
            f"{error_text}. Check COQUI_VENV_SITE_PACKAGES/extra_site_packages."
        )

    def _on_transcription_result(
        self, segment: AudioSegment, text: str, error_text: str
    ) -> None:
        if error_text:
            self.get_logger().error(f"Transcription failed: {error_text}")
            return

        cleaned = text.strip()
        if self.log_transcripts:
            stamp = time.strftime("%H:%M:%S", time.localtime(segment.end_time))
            if cleaned:
                self.get_logger().info(f"[{stamp}] {cleaned}")
            else:
                self.get_logger().info(f"[{stamp}] <no speech recognized>")

        with self._pending_lock:
            pending = self._pending_command
            if pending is not None and cleaned:
                pending.text = cleaned
                pending.event.set()
                return

        if not cleaned:
            return

        if not self._is_wake_word_armed():
            return

        normalized_text = self._normalize_text(cleaned)
        normalized_awake = self._normalize_text(self.awake_word)
        if normalized_awake and normalized_awake in normalized_text:
            with self._status_lock:
                self._wake_word_armed = False
            self._publish_awake(True)
            self.get_logger().info(
                f"Awake word '{self.awake_word}' detected. Requesting robot status 'idle'."
            )
            self._set_robot_status_async("idle")

    def _handle_get_command(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        del request

        if not self._transcriber.ready.is_set():
            response.success = False
            response.message = "Whisper model is still loading."
            return response
        if self._transcriber.startup_error:
            response.success = False
            response.message = f"Whisper unavailable: {self._transcriber.startup_error}"
            return response

        pending = PendingCommand(event=threading.Event())
        with self._pending_lock:
            if self._pending_command is not None:
                response.success = False
                response.message = "get_command is already in progress."
                return response
            self._pending_command = pending

        self._set_robot_status_async("listening")
        got_text = pending.event.wait(timeout=self.get_command_timeout_sec)

        with self._pending_lock:
            if self._pending_command is pending:
                self._pending_command = None
            text = pending.text.strip()

        self._set_robot_status_async("idle")

        if not got_text:
            response.success = False
            response.message = (
                f"Timed out after {self.get_command_timeout_sec:.1f}s waiting for speech."
            )
            return response

        if not text:
            response.success = False
            response.message = "No speech recognized."
            return response

        response.success = True
        response.message = text
        return response

    def destroy_node(self) -> bool:
        self._stop_event.set()
        try:
            self._segments.put_nowait(None)
        except queue.Full:
            pass

        self._transcriber.stop()
        if self._audio_thread is not None and self._audio_thread.is_alive():
            self._audio_thread.join(timeout=2.0)
        if self._transcriber.is_alive():
            self._transcriber.join(timeout=5.0)

        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = WhisperCommandNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

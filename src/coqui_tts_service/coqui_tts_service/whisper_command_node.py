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
VALID_STATUSES = ("sleep", "listening", "idle", "thinking", "operating")


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
from std_msgs.msg import String
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


@dataclass(frozen=True)
class CalibrationStats:
    median_rms: float
    p90_rms: float
    p95_rms: float
    p99_rms: float
    robust_sigma: float


@dataclass(frozen=True)
class SegmenterTuning:
    noise_rms: float
    threshold: float
    min_rms: float
    energy_multiplier: float
    trigger_frames: int
    silence_seconds: float
    pre_roll_seconds: float
    min_speech_seconds: float


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
        self._manual_silence_seconds = silence_seconds if silence_seconds > 0 else None
        self._manual_pre_roll_seconds = pre_roll_seconds if pre_roll_seconds > 0 else None
        self._manual_min_speech_seconds = min_speech_seconds if min_speech_seconds > 0 else None
        self._manual_trigger_frames = trigger_frames if trigger_frames > 0 else None
        self.fixed_threshold = energy_threshold if energy_threshold > 0 else None
        self._manual_energy_multiplier = energy_multiplier if energy_multiplier > 0 else None
        self._manual_min_rms = min_rms if min_rms > 0 else None
        self.calibration_seconds = max(0.0, calibration_seconds)
        self._auto_calibration_enabled = any(
            value is None
            for value in (
                self._manual_silence_seconds,
                self._manual_pre_roll_seconds,
                self._manual_min_speech_seconds,
                self._manual_trigger_frames,
                self._manual_energy_multiplier,
                self._manual_min_rms,
            )
        ) or self.fixed_threshold is None

        self.silence_frames = max(
            1,
            int(
                round(
                    (self._manual_silence_seconds or 0.8)
                    / self.frame_s
                )
            ),
        )
        self.min_speech_seconds = max(0.05, self._manual_min_speech_seconds or 0.25)
        self.trigger_frames = max(1, self._manual_trigger_frames or 1)
        self.energy_multiplier = max(1.0, self._manual_energy_multiplier or 1.1)
        self.min_rms = max(1.0, self._manual_min_rms or 1.0)
        self.pre_roll = deque(
            maxlen=max(1, int(round((self._manual_pre_roll_seconds or 0.6) / self.frame_s)))
        )
        self.calibration_target_frames = max(0, int(round(self.calibration_seconds / self.frame_s)))
        self.calibration_values: list[float] = []
        self.noise_rms: float | None = None
        self.calibration_stats: CalibrationStats | None = None
        self.tuning: SegmenterTuning | None = None
        self.calibration_done = (
            not self._auto_calibration_enabled or self.calibration_target_frames == 0
        )
        if self.calibration_done:
            noise_rms = max(1.0, self.min_rms)
            threshold = self.fixed_threshold if self.fixed_threshold is not None else max(
                self.min_rms, noise_rms * self.energy_multiplier
            )
            self.noise_rms = noise_rms
            self.tuning = SegmenterTuning(
                noise_rms=noise_rms,
                threshold=threshold,
                min_rms=self.min_rms,
                energy_multiplier=self.energy_multiplier,
                trigger_frames=self.trigger_frames,
                silence_seconds=self.silence_frames * self.frame_s,
                pre_roll_seconds=self.pre_roll.maxlen * self.frame_s,
                min_speech_seconds=self.min_speech_seconds,
            )

        self.in_speech = False
        self.speech_run = 0
        self.silence_run = 0
        self.frames: list[np.ndarray] = []
        self.segment_start_time = 0.0

    @staticmethod
    def _clamp(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))

    @staticmethod
    def rms(frame: np.ndarray) -> float:
        x = frame.astype(np.float32)
        return float(np.sqrt(np.mean(x * x)))

    @staticmethod
    def _build_stats(values: list[float]) -> CalibrationStats:
        arr = np.asarray(values, dtype=np.float32)
        median = float(np.median(arr))
        p90, p95, p99 = np.percentile(arr, [90, 95, 99]).astype(float)
        mad = float(np.median(np.abs(arr - median)))
        robust_sigma = max(1.0, 1.4826 * mad)
        return CalibrationStats(
            median_rms=max(1.0, median),
            p90_rms=max(1.0, p90),
            p95_rms=max(1.0, p95),
            p99_rms=max(1.0, p99),
            robust_sigma=robust_sigma,
        )

    def needs_calibration_audio(self) -> bool:
        return self._auto_calibration_enabled and not self.calibration_done

    def _derive_auto_tuning(self, stats: CalibrationStats) -> SegmenterTuning:
        noise_rms = max(1.0, stats.median_rms)
        transient_ratio = stats.p99_rms / max(noise_rms, 1.0)
        variability = stats.robust_sigma / max(noise_rms, 1.0)
        auto_threshold = max(
            noise_rms + 3.0 * stats.robust_sigma,
            stats.p95_rms + 1.5 * stats.robust_sigma,
            stats.p99_rms * 1.05,
            noise_rms * 1.12,
        )
        threshold = (
            self.fixed_threshold
            if self.fixed_threshold is not None
            else self._clamp(auto_threshold, noise_rms * 1.05, noise_rms * 3.5)
        )
        min_rms = (
            max(1.0, self._manual_min_rms)
            if self._manual_min_rms is not None
            else max(1.0, min(threshold * 0.95, max(stats.p90_rms, noise_rms * 1.02)))
        )
        energy_multiplier = (
            max(1.0, self._manual_energy_multiplier)
            if self._manual_energy_multiplier is not None
            else self._clamp(threshold / max(noise_rms, 1.0), 1.02, 3.5)
        )

        if self._manual_trigger_frames is not None:
            trigger_frames = max(1, self._manual_trigger_frames)
        elif variability > 0.18 or transient_ratio > 1.6:
            trigger_frames = 3
        elif variability > 0.08 or transient_ratio > 1.25:
            trigger_frames = 2
        else:
            trigger_frames = 1

        silence_seconds = (
            max(self.frame_s, self._manual_silence_seconds)
            if self._manual_silence_seconds is not None
            else self._clamp(
                0.55 + 0.7 * variability + 0.08 * (trigger_frames - 1),
                0.6,
                1.2,
            )
        )
        pre_roll_seconds = (
            max(self.frame_s, self._manual_pre_roll_seconds)
            if self._manual_pre_roll_seconds is not None
            else self._clamp(
                0.42 + 0.08 * trigger_frames + 0.25 * variability,
                0.45,
                0.9,
            )
        )
        min_speech_seconds = (
            max(0.05, self._manual_min_speech_seconds)
            if self._manual_min_speech_seconds is not None
            else self._clamp(
                0.18 + 0.06 * (trigger_frames - 1) + 0.2 * variability,
                0.18,
                0.45,
            )
        )

        return SegmenterTuning(
            noise_rms=noise_rms,
            threshold=threshold,
            min_rms=min_rms,
            energy_multiplier=energy_multiplier,
            trigger_frames=trigger_frames,
            silence_seconds=silence_seconds,
            pre_roll_seconds=pre_roll_seconds,
            min_speech_seconds=min_speech_seconds,
        )

    def _apply_tuning(self, tuning: SegmenterTuning) -> None:
        self.noise_rms = tuning.noise_rms
        self.min_rms = max(1.0, tuning.min_rms)
        self.energy_multiplier = max(1.0, tuning.energy_multiplier)
        self.trigger_frames = max(1, tuning.trigger_frames)
        self.min_speech_seconds = max(0.05, tuning.min_speech_seconds)
        self.silence_frames = max(1, int(round(tuning.silence_seconds / self.frame_s)))
        new_pre_roll_frames = max(1, int(round(tuning.pre_roll_seconds / self.frame_s)))
        self.pre_roll = deque(self.pre_roll, maxlen=new_pre_roll_frames)
        self.tuning = SegmenterTuning(
            noise_rms=tuning.noise_rms,
            threshold=tuning.threshold,
            min_rms=self.min_rms,
            energy_multiplier=self.energy_multiplier,
            trigger_frames=self.trigger_frames,
            silence_seconds=self.silence_frames * self.frame_s,
            pre_roll_seconds=new_pre_roll_frames * self.frame_s,
            min_speech_seconds=self.min_speech_seconds,
        )

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
        if self.calibration_done:
            return True
        if self.calibration_target_frames == 0 or not self._auto_calibration_enabled:
            self.calibration_stats = self._build_stats(self.calibration_values or [self.min_rms])
            tuning = self._derive_auto_tuning(self.calibration_stats)
            self._apply_tuning(tuning)
            self.calibration_done = True
            return True
        if len(self.calibration_values) < self.calibration_target_frames:
            return False
        self.calibration_stats = self._build_stats(self.calibration_values)
        self._apply_tuning(self._derive_auto_tuning(self.calibration_stats))
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
        self.declare_parameter("status_service", "/robot_status")
        self.declare_parameter("get_command_service", "/get_command")
        self.declare_parameter("awake_word", "hi")
        self.declare_parameter("awake_words", ["hi"])
        self.declare_parameter("get_command_timeout_sec", 10.0)
        self.declare_parameter("audio_device", "default")
        self.declare_parameter("rate", 16000)
        self.declare_parameter("frame_ms", 30)
        self.declare_parameter("silence_seconds", 0.0)
        self.declare_parameter("pre_roll_seconds", 0.0)
        self.declare_parameter("min_speech_seconds", 0.0)
        self.declare_parameter("trigger_frames", 0)
        self.declare_parameter("energy_threshold", 0.0)
        self.declare_parameter("energy_multiplier", 0.0)
        self.declare_parameter("min_rms", 0.0)
        self.declare_parameter("calibration_seconds", 10.0)
        self.declare_parameter("model", "medium")
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
        self.status_service = str(self.get_parameter("status_service").value)
        self.get_command_service = str(self.get_parameter("get_command_service").value)
        self.awake_word = str(self.get_parameter("awake_word").value).strip().lower()
        awake_words_value = self.get_parameter("awake_words").value
        self.awake_words = self._build_awake_words(awake_words_value, self.awake_word)
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
            f"awake_words={self.awake_words}"
        )
        self.get_logger().info(
            f"extra_site_packages={self.extra_site_packages} isolate_site_packages={self.isolate_site_packages} "
            f"(added={added}, removed={removed})"
        )
        if self._segmenter.needs_calibration_audio():
            self.get_logger().info(
                f"Collecting {self.calibration_seconds:.1f}s of ambient audio to auto-calibrate Whisper VAD. "
                "Keep the room quiet during startup for best results."
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

    @classmethod
    def _build_awake_words(cls, configured_words, legacy_awake_word: str) -> list[str]:
        words: list[str] = []

        if isinstance(configured_words, (list, tuple)):
            for value in configured_words:
                normalized = cls._normalize_text(str(value).strip().lower())
                if normalized and normalized not in words:
                    words.append(normalized)

        legacy_normalized = cls._normalize_text(str(legacy_awake_word).strip().lower())
        if legacy_normalized and legacy_normalized not in words:
            words.append(legacy_normalized)

        if not words:
            words = ["hi"]

        return words

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
                    should_process_audio = should_listen or self._segmenter.needs_calibration_audio()
                    if not should_process_audio:
                        if was_listening:
                            self._segmenter.reset_activity()
                            was_listening = False
                        continue
                    was_listening = should_listen

                    started, segment, _, _ = self._segmenter.process(frame, now)

                    if (
                        self._segmenter.calibration_stats is not None
                        and self._segmenter.tuning is not None
                        and self._segmenter.calibration_done
                        and not announced_calibration
                    ):
                        stats = self._segmenter.calibration_stats
                        tuning = self._segmenter.tuning
                        self.get_logger().info(
                            "Whisper VAD auto-calibrated "
                            f"(ambient={self.calibration_seconds:.1f}s, "
                            f"noise_rms={tuning.noise_rms:.1f}, p95={stats.p95_rms:.1f}, "
                            f"threshold={tuning.threshold:.1f}, min_rms={tuning.min_rms:.1f}, "
                            f"energy_multiplier={tuning.energy_multiplier:.2f}, "
                            f"trigger_frames={tuning.trigger_frames}, "
                            f"pre_roll={tuning.pre_roll_seconds:.2f}s, "
                            f"min_speech={tuning.min_speech_seconds:.2f}s, "
                            f"silence={tuning.silence_seconds:.2f}s)"
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
        matched_awake_word = next(
            (awake_word for awake_word in self.awake_words if awake_word in normalized_text),
            "",
        )
        if matched_awake_word:
            with self._status_lock:
                self._wake_word_armed = False
            self.get_logger().info(
                f"Awake word '{matched_awake_word}' detected. Requesting robot status 'idle'."
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

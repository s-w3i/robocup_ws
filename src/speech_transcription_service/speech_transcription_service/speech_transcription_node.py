#!/usr/bin/env python3
"""ROS2 Whisper speech transcription service with wake phrase detection.

Features:
- Service `/request_speech` (std_srvs/Trigger):
  - Starts a speech session when called.
  - Returns final transcription when 2s silence is detected (configurable).
- Always-on wake phrase detection:
  - Listens continuously for "Hi Alice" (configurable).
  - Publishes `std_msgs/Bool(data=True)` on `/awake` when detected.
"""

from __future__ import annotations

import ctypes
import os
import pathlib
import queue
import re
import shutil
import subprocess
import sys
import sysconfig
import threading
import time
import types
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from std_srvs.srv import Trigger

np: Any = None


def ensure_torch_runtime_libs() -> None:
    """Expose CUDA libs from wheels/system paths before importing torch."""
    purelib = pathlib.Path(sysconfig.get_paths().get("purelib", ""))
    lib_dirs: list[str] = []

    nvidia_root = purelib / "nvidia"
    if nvidia_root.is_dir():
        for lib_dir in nvidia_root.glob("*/lib"):
            if lib_dir.is_dir():
                lib_dirs.append(str(lib_dir))

    for path in (
        "/usr/local/cuda/lib64",
        "/usr/local/cuda-12.6/lib64",
        "/usr/local/cuda/targets/aarch64-linux/lib",
        "/usr/local/cuda-12.6/targets/aarch64-linux/lib",
        "/usr/lib/aarch64-linux-gnu",
        "/lib/aarch64-linux-gnu",
    ):
        if pathlib.Path(path).is_dir():
            lib_dirs.append(path)

    current = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [entry for entry in current.split(":") if entry]
    for lib_dir in lib_dirs:
        if lib_dir not in parts:
            parts.insert(0, lib_dir)
    os.environ["LD_LIBRARY_PATH"] = ":".join(parts)


def preload_cupti_if_needed() -> None:
    """Preload CUPTI when available (common Jetson runtime requirement)."""
    candidates = [
        pathlib.Path(
            "/home/usern/coqui-venv/lib/python3.10/site-packages/nvidia/cuda_cupti/lib/libcupti.so.12"
        ),
        pathlib.Path("/usr/local/cuda-12.6/extras/CUPTI/lib64/libcupti.so.12"),
        pathlib.Path("/usr/local/cuda/extras/CUPTI/lib64/libcupti.so.12"),
    ]
    for candidate in candidates:
        if candidate.exists():
            ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
            return


def maybe_add_site_packages(path: str) -> bool:
    """Add external site-packages path at runtime if available."""
    if not path:
        return False
    p = pathlib.Path(path).expanduser().resolve()
    if not p.is_dir():
        return False
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)
    return True


def ensure_numpy(extra_site_packages: str = "") -> Any:
    """Import numpy after optional external site-packages path setup."""
    global np
    if np is not None:
        return np
    maybe_add_site_packages(extra_site_packages)
    import numpy as _np

    np = _np
    return np


def ensure_coverage_stub_for_numba() -> None:
    """Provide compatibility for environments where coverage.types is missing.

    Some numba builds expect coverage.types.* symbols at import time. In mixed
    Python environments this can fail if an older coverage module is imported.
    """
    try:
        import coverage as cov  # type: ignore

        if hasattr(cov, "types") and hasattr(cov.types, "Tracer"):
            return
    except Exception:
        pass

    module = types.ModuleType("coverage")

    class _Coverage:
        @staticmethod
        def current() -> None:
            return None

    class _Types:
        class Tracer:
            pass

        TTraceData = dict
        TShouldTraceFn = Any
        TFileDisposition = Any
        TShouldStartContextFn = Any
        TWarnFn = Any
        TTraceFn = Any

    module.Coverage = _Coverage
    module.types = _Types
    sys.modules["coverage"] = module


def read_exact(pipe, size: int) -> bytes:
    """Read exactly `size` bytes unless EOF."""
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        data = pipe.read(remaining)
        if not data:
            break
        chunks.append(data)
        remaining -= len(data)
    return b"".join(chunks)


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@dataclass
class AudioSegment:
    start_time: float
    end_time: float
    samples: np.ndarray

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)


class EnergySegmenter:
    """Simple RMS VAD-based segmenter with silence-based end detection."""

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
        self.min_speech_seconds = min_speech_seconds
        self.trigger_frames = max(1, trigger_frames)

        self.fixed_threshold = energy_threshold if energy_threshold > 0 else None
        self.energy_multiplier = max(1.0, energy_multiplier)
        self.min_rms = max(1.0, min_rms)

        self.pre_roll = deque(
            maxlen=max(1, int(round(pre_roll_seconds / self.frame_s)))
        )
        self.calibration_target_frames = max(
            0, int(round(calibration_seconds / self.frame_s))
        )
        self.calibration_values: list[float] = []
        self.noise_rms: float | None = None
        self.calibration_done = (
            self.fixed_threshold is not None or self.calibration_target_frames == 0
        )

        self.in_speech = False
        self.speech_run = 0
        self.silence_run = 0
        self.frames: list[np.ndarray] = []
        self.segment_start_time = 0.0

    def current_threshold(self) -> float:
        if self.fixed_threshold is not None:
            return self.fixed_threshold
        base = self.noise_rms if self.noise_rms is not None else self.min_rms
        return max(self.min_rms, base * self.energy_multiplier)

    @staticmethod
    def rms(frame: np.ndarray) -> float:
        x = frame.astype(np.float32)
        return float(np.sqrt(np.mean(x * x)))

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
        self.noise_rms = max(1.0, float(np.mean(self.calibration_values)))
        self.calibration_done = True
        return True

    def process(
        self, frame: np.ndarray, frame_end_time: float
    ) -> tuple[bool, AudioSegment | None]:
        frame_rms = self.rms(frame)
        threshold = self.current_threshold()

        if not self._maybe_finish_calibration() and not self.in_speech:
            self.calibration_values.append(frame_rms)
            self.pre_roll.append(frame.copy())
            return False, None

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

        return started, segment

    def reset(self) -> None:
        self._reset_after_segment()
        self.calibration_values.clear()
        self.noise_rms = None
        self.calibration_done = (
            self.fixed_threshold is not None or self.calibration_target_frames == 0
        )

    def _reset_after_segment(self) -> None:
        self.in_speech = False
        self.speech_run = 0
        self.silence_run = 0
        self.frames = []
        self.pre_roll.clear()


@dataclass
class RequestSession:
    done_event: threading.Event = field(default_factory=threading.Event)
    text: str = ""
    error: str = ""


class SpeechTranscriptionNode(Node):
    def __init__(self) -> None:
        super().__init__("speech_transcription_node")

        self.declare_parameter("service_name", "/request_speech")
        self.declare_parameter("awake_topic", "/awake")
        self.declare_parameter("wake_phrase", "hi eva")
        self.declare_parameter("wake_cooldown_sec", 2.0)

        self.declare_parameter("audio_device", "default")
        self.declare_parameter("sample_rate", 16000)
        self.declare_parameter("frame_ms", 30)
        self.declare_parameter("audio_restart_delay_sec", 1.0)

        self.declare_parameter("whisper_model", "tiny.en")
        self.declare_parameter("whisper_language", "en")
        self.declare_parameter("whisper_task", "transcribe")
        self.declare_parameter("whisper_device", "cuda")
        self.declare_parameter("whisper_no_fp16", False)
        self.declare_parameter("whisper_model_dir", "")
        self.declare_parameter(
            "extra_site_packages",
            "/home/usern/coqui-venv/lib/python3.10/site-packages",
        )

        self.declare_parameter("request_timeout_seconds", 30.0)
        self.declare_parameter("request_silence_seconds", 2.0)
        self.declare_parameter("request_pre_roll_seconds", 0.3)
        self.declare_parameter("request_min_speech_seconds", 0.2)
        self.declare_parameter("request_trigger_frames", 2)
        self.declare_parameter("request_energy_threshold", 0.0)
        self.declare_parameter("request_energy_multiplier", 2.0)
        self.declare_parameter("request_min_rms", 120.0)
        self.declare_parameter("request_calibration_seconds", 0.0)

        self.declare_parameter("wake_silence_seconds", 0.8)
        self.declare_parameter("wake_pre_roll_seconds", 0.2)
        self.declare_parameter("wake_min_speech_seconds", 0.2)
        self.declare_parameter("wake_trigger_frames", 2)
        self.declare_parameter("wake_energy_threshold", 0.0)
        self.declare_parameter("wake_energy_multiplier", 2.0)
        self.declare_parameter("wake_min_rms", 120.0)
        self.declare_parameter("wake_calibration_seconds", 1.0)
        self.declare_parameter("wake_max_segment_seconds", 4.0)
        self.declare_parameter("wake_queue_size", 4)

        self.service_name = str(self.get_parameter("service_name").value)
        self.awake_topic = str(self.get_parameter("awake_topic").value)
        self.wake_phrase = str(self.get_parameter("wake_phrase").value)
        self.wake_phrase_norm = normalize_text(self.wake_phrase)
        self.wake_cooldown_sec = float(self.get_parameter("wake_cooldown_sec").value)

        self.audio_device = str(self.get_parameter("audio_device").value)
        self.sample_rate = int(self.get_parameter("sample_rate").value)
        self.frame_ms = int(self.get_parameter("frame_ms").value)
        self.audio_restart_delay_sec = float(
            self.get_parameter("audio_restart_delay_sec").value
        )

        self.whisper_model = str(self.get_parameter("whisper_model").value)
        lang = str(self.get_parameter("whisper_language").value).strip()
        self.whisper_language = lang if lang else None
        self.whisper_task = str(self.get_parameter("whisper_task").value)
        self.whisper_device_req = str(self.get_parameter("whisper_device").value)
        self.whisper_no_fp16 = bool(self.get_parameter("whisper_no_fp16").value)
        model_dir = str(self.get_parameter("whisper_model_dir").value).strip()
        self.whisper_model_dir = model_dir if model_dir else None
        self.extra_site_packages = str(self.get_parameter("extra_site_packages").value).strip()

        self.request_timeout_seconds = float(
            self.get_parameter("request_timeout_seconds").value
        )

        if shutil.which("arecord") is None:
            raise RuntimeError("arecord not found. Install ALSA utilities first.")
        if self.sample_rate <= 0 or self.frame_ms <= 0:
            raise RuntimeError("sample_rate and frame_ms must be > 0.")
        if (self.sample_rate * self.frame_ms) % 1000 != 0:
            raise RuntimeError("sample_rate * frame_ms must be divisible by 1000.")

        self.frame_samples = self.sample_rate * self.frame_ms // 1000
        self.frame_bytes = self.frame_samples * 2

        self._awake_pub = self.create_publisher(Bool, self.awake_topic, 10)
        self.create_service(Trigger, self.service_name, self._handle_request_speech)

        self._state_lock = threading.Lock()
        self._model_lock = threading.Lock()
        self._proc_lock = threading.Lock()

        self._active_request: RequestSession | None = None
        self._request_segmenter: EnergySegmenter | None = None
        self._wake_last_pub_time = 0.0

        self._arecord_proc: subprocess.Popen[bytes] | None = None
        self._stop_event = threading.Event()
        self._wake_queue: queue.Queue[np.ndarray] = queue.Queue(
            maxsize=max(1, int(self.get_parameter("wake_queue_size").value))
        )

        self._wake_segmenter = EnergySegmenter(
            rate=self.sample_rate,
            frame_ms=self.frame_ms,
            silence_seconds=float(self.get_parameter("wake_silence_seconds").value),
            pre_roll_seconds=float(self.get_parameter("wake_pre_roll_seconds").value),
            min_speech_seconds=float(
                self.get_parameter("wake_min_speech_seconds").value
            ),
            trigger_frames=int(self.get_parameter("wake_trigger_frames").value),
            energy_threshold=float(self.get_parameter("wake_energy_threshold").value),
            energy_multiplier=float(self.get_parameter("wake_energy_multiplier").value),
            min_rms=float(self.get_parameter("wake_min_rms").value),
            calibration_seconds=float(
                self.get_parameter("wake_calibration_seconds").value
            ),
        )
        self._wake_max_segment_seconds = float(
            self.get_parameter("wake_max_segment_seconds").value
        )

        self._load_whisper_model()

        self._audio_thread = threading.Thread(target=self._audio_loop, daemon=True)
        self._wake_thread = threading.Thread(target=self._wake_loop, daemon=True)
        self._audio_thread.start()
        self._wake_thread.start()

        self.get_logger().info(
            f"Speech service ready on {self.service_name}, awake topic: {self.awake_topic}"
        )
        self.get_logger().info(
            f"Wake phrase: '{self.wake_phrase}' | whisper={self.whisper_model} on {self._whisper_device}"
        )

    def _new_request_segmenter(self) -> EnergySegmenter:
        return EnergySegmenter(
            rate=self.sample_rate,
            frame_ms=self.frame_ms,
            silence_seconds=float(self.get_parameter("request_silence_seconds").value),
            pre_roll_seconds=float(self.get_parameter("request_pre_roll_seconds").value),
            min_speech_seconds=float(
                self.get_parameter("request_min_speech_seconds").value
            ),
            trigger_frames=int(self.get_parameter("request_trigger_frames").value),
            energy_threshold=float(self.get_parameter("request_energy_threshold").value),
            energy_multiplier=float(
                self.get_parameter("request_energy_multiplier").value
            ),
            min_rms=float(self.get_parameter("request_min_rms").value),
            calibration_seconds=float(
                self.get_parameter("request_calibration_seconds").value
            ),
        )

    def _load_whisper_model(self) -> None:
        ensure_torch_runtime_libs()
        preload_cupti_if_needed()
        maybe_add_site_packages(self.extra_site_packages)
        ensure_numpy(self.extra_site_packages)
        ensure_coverage_stub_for_numba()

        try:
            import torch
            import whisper
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Whisper dependencies not found in current ROS Python env. "
                "Set parameter 'extra_site_packages' to your venv site-packages "
                f"(current: '{self.extra_site_packages}'). Missing: {exc}"
            ) from exc

        requested = self.whisper_device_req.lower()
        if requested == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif requested == "cuda":
            if torch.cuda.is_available():
                device = "cuda"
            else:
                self.get_logger().warn(
                    "whisper_device=cuda requested but CUDA unavailable; falling back to cpu."
                )
                device = "cpu"
        else:
            device = "cpu"

        self._whisper_device = device
        self._use_fp16 = self._whisper_device == "cuda" and not self.whisper_no_fp16

        started = time.perf_counter()
        self._whisper_model = whisper.load_model(
            self.whisper_model,
            device=self._whisper_device,
            download_root=self.whisper_model_dir,
        )
        elapsed = time.perf_counter() - started
        self.get_logger().info(
            f"Loaded Whisper model '{self.whisper_model}' in {elapsed:.2f}s "
            f"(device={self._whisper_device}, fp16={self._use_fp16})"
        )

    def _transcribe_samples(self, samples: np.ndarray) -> str:
        audio = samples.astype(np.float32) / 32768.0
        with self._model_lock:
            try:
                result = self._whisper_model.transcribe(
                    audio,
                    language=self.whisper_language,
                    task=self.whisper_task,
                    fp16=self._use_fp16,
                    temperature=0.0,
                    condition_on_previous_text=False,
                    verbose=False,
                )
            except TypeError:
                result = self._whisper_model.transcribe(
                    audio,
                    language=self.whisper_language,
                    task=self.whisper_task,
                    fp16=self._use_fp16,
                )
        return str(result.get("text", "")).strip()

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
            str(self.sample_rate),
            "-t",
            "raw",
            "-q",
        ]

        while not self._stop_event.is_set():
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
            with self._proc_lock:
                self._arecord_proc = proc

            if proc.stdout is None:
                self.get_logger().error("Failed to open arecord stdout.")
                break

            try:
                while not self._stop_event.is_set():
                    chunk = read_exact(proc.stdout, self.frame_bytes)
                    if len(chunk) < self.frame_bytes:
                        break

                    frame = np.frombuffer(chunk, dtype=np.int16).copy()
                    now = time.time()
                    self._process_wake_frame(frame, now)
                    self._process_request_frame(frame, now)
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        proc.kill()

                if proc.stderr is not None:
                    stderr_text = proc.stderr.read().decode(errors="ignore").strip()
                    if stderr_text and not self._stop_event.is_set():
                        self.get_logger().warn(f"arecord stderr: {stderr_text}")

                with self._proc_lock:
                    if self._arecord_proc is proc:
                        self._arecord_proc = None

            if not self._stop_event.is_set():
                time.sleep(self.audio_restart_delay_sec)

    def _process_wake_frame(self, frame: np.ndarray, now: float) -> None:
        _started, segment = self._wake_segmenter.process(frame, now)
        if segment is None:
            return
        if segment.duration > self._wake_max_segment_seconds:
            return
        try:
            self._wake_queue.put_nowait(segment.samples)
        except queue.Full:
            try:
                _ = self._wake_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._wake_queue.put_nowait(segment.samples)
            except queue.Full:
                pass

    def _process_request_frame(self, frame: np.ndarray, now: float) -> None:
        with self._state_lock:
            session = self._active_request
            segmenter = self._request_segmenter

        if session is None or segmenter is None:
            return

        started, segment = segmenter.process(frame, now)
        if started:
            self.get_logger().info("Request speech: speech started.")

        if segment is None:
            return

        try:
            text = self._transcribe_samples(segment.samples)
            if text:
                self.get_logger().debug(f"Request transcription text: {text!r}")
            else:
                self.get_logger().debug("Request transcription text: <empty>")
            if not text:
                error = "No speech recognized."
            else:
                error = ""
        except Exception as exc:
            text = ""
            error = f"Transcription failed: {exc}"

        with self._state_lock:
            if self._active_request is session:
                self._active_request = None
                self._request_segmenter = None

        session.text = text
        session.error = error
        session.done_event.set()

    def _wake_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                samples = self._wake_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                text = self._transcribe_samples(samples)
            except Exception as exc:
                self.get_logger().warn(f"Wake transcription failed: {exc}")
                continue

            if not text:
                self.get_logger().debug("Wake transcription text: <empty>")
                continue

            normalized = normalize_text(text)
            self.get_logger().debug(
                f"Wake transcription text: {text!r} | normalized={normalized!r}"
            )
            if self.wake_phrase_norm not in normalized:
                continue

            now = time.time()
            if (now - self._wake_last_pub_time) < self.wake_cooldown_sec:
                continue

            self._wake_last_pub_time = now
            msg = Bool()
            msg.data = True
            self._awake_pub.publish(msg)
            self.get_logger().info(f"Wake phrase detected: '{text}'")

    def _handle_request_speech(
        self, _request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        session = RequestSession()
        with self._state_lock:
            if self._active_request is not None:
                response.success = False
                response.message = "Another speech request is already running."
                return response
            self._active_request = session
            self._request_segmenter = self._new_request_segmenter()

        self.get_logger().info("Request speech: waiting for utterance.")
        done = session.done_event.wait(timeout=self.request_timeout_seconds)

        with self._state_lock:
            if self._active_request is session:
                self._active_request = None
                self._request_segmenter = None

        if not done:
            response.success = False
            response.message = "Timed out waiting for speech to finish."
            return response

        if session.error:
            response.success = False
            response.message = session.error
            return response

        response.success = True
        response.message = session.text
        return response

    def destroy_node(self) -> bool:
        self._stop_event.set()
        with self._proc_lock:
            proc = self._arecord_proc
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                proc.kill()

        if hasattr(self, "_audio_thread"):
            self._audio_thread.join(timeout=2.0)
        if hasattr(self, "_wake_thread"):
            self._wake_thread.join(timeout=2.0)
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = SpeechTranscriptionNode()
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

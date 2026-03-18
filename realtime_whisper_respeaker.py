#!/usr/bin/env python3
"""
Realtime speech-to-text from ReSpeaker microphone with Whisper.

Behavior:
- Starts an utterance automatically when speech energy is detected.
- Ends an utterance after N seconds of silence (default 2.0).
- Transcribes each utterance in a background worker.
"""

from __future__ import annotations

import argparse
import queue
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np


DEFAULT_ALSA_DEVICE = "default"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Realtime speech-to-text with ReSpeaker + Whisper."
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_ALSA_DEVICE,
        help=f"ALSA input device passed to arecord (default: {DEFAULT_ALSA_DEVICE})",
    )
    parser.add_argument(
        "-r",
        "--rate",
        type=int,
        default=16000,
        help="Sample rate in Hz (default: %(default)s)",
    )
    parser.add_argument(
        "--frame-ms",
        type=int,
        default=30,
        help="Frame size in milliseconds for VAD (default: %(default)s)",
    )
    parser.add_argument(
        "--silence-seconds",
        type=float,
        default=2.0,
        help="End utterance after this much silence (default: %(default)s)",
    )
    parser.add_argument(
        "--pre-roll-seconds",
        type=float,
        default=0.30,
        help="Audio to prepend before trigger (default: %(default)s)",
    )
    parser.add_argument(
        "--min-speech-seconds",
        type=float,
        default=0.40,
        help="Drop utterances shorter than this (default: %(default)s)",
    )
    parser.add_argument(
        "--trigger-frames",
        type=int,
        default=2,
        help="Consecutive speech frames required to start (default: %(default)s)",
    )
    parser.add_argument(
        "--energy-threshold",
        type=float,
        default=0.0,
        help="Fixed RMS threshold; 0 enables adaptive thresholding (default: %(default)s)",
    )
    parser.add_argument(
        "--energy-multiplier",
        type=float,
        default=2.0,
        help="Adaptive threshold multiplier over noise floor (default: %(default)s)",
    )
    parser.add_argument(
        "--min-rms",
        type=float,
        default=120.0,
        help="Lower bound for adaptive RMS threshold (default: %(default)s)",
    )
    parser.add_argument(
        "--calibration-seconds",
        type=float,
        default=1.0,
        help="Initial noise-floor calibration seconds in adaptive mode (default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        default="base",
        help="Whisper model name/path (default: %(default)s)",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Whisper language code, e.g. en, zh, ja (default: auto)",
    )
    parser.add_argument(
        "--task",
        choices=["transcribe", "translate"],
        default="transcribe",
        help="Whisper task (default: %(default)s)",
    )
    parser.add_argument(
        "--whisper-device",
        choices=["auto", "cpu", "cuda"],
        default="cuda",
        help="Compute device for Whisper (default: %(default)s)",
    )
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable fp16 decode on CUDA.",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Directory for Whisper model downloads/cache.",
    )
    parser.add_argument(
        "--max-queue",
        type=int,
        default=8,
        help="Max pending utterances for transcription (default: %(default)s)",
    )
    parser.add_argument(
        "--partial-interval",
        type=float,
        default=0.6,
        help="Seconds between partial realtime transcriptions (default: %(default)s)",
    )
    parser.add_argument(
        "--partial-min-seconds",
        type=float,
        default=0.5,
        help="Minimum active speech seconds before partial decode (default: %(default)s)",
    )
    parser.add_argument(
        "--partial-window-seconds",
        type=float,
        default=6.0,
        help="Max recent audio window for partial decode, 0 means full utterance (default: %(default)s)",
    )
    parser.add_argument(
        "--manual-end",
        action="store_true",
        help="Enable terminal command '/end' + Enter to force-finish current utterance.",
    )
    parser.add_argument(
        "--debug-vad",
        action="store_true",
        help="Print RMS and threshold periodically for tuning.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Show capture devices from arecord and exit.",
    )
    return parser


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True)


@dataclass
class AudioSegment:
    index: int
    start_time: float
    end_time: float
    samples: np.ndarray

    @property
    def duration(self) -> float:
        return max(0.0, self.end_time - self.start_time)


@dataclass
class TranscriptionJob:
    kind: str  # "partial" | "final"
    segment: AudioSegment


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
        self.calibration_done = self.fixed_threshold is not None or self.calibration_target_frames == 0

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
                        index=-1,
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
            index=-1,
            start_time=self.segment_start_time,
            end_time=now,
            samples=np.concatenate(self.frames),
        )
        self._reset_after_segment()
        return segment

    def snapshot_active(self, now: float, max_seconds: float = 0.0) -> AudioSegment | None:
        if not self.in_speech or not self.frames:
            return None
        frames = self.frames
        start = self.segment_start_time
        if max_seconds > 0:
            max_frames = max(1, int(round(max_seconds / self.frame_s)))
            if len(frames) > max_frames:
                frames = frames[-max_frames:]
                start = now - len(frames) * self.frame_s
        return AudioSegment(
            index=-1,
            start_time=start,
            end_time=now,
            samples=np.concatenate(frames),
        )

    def _reset_after_segment(self) -> None:
        self.in_speech = False
        self.speech_run = 0
        self.silence_run = 0
        self.frames = []
        self.pre_roll.clear()


class WhisperWorker(threading.Thread):
    def __init__(
        self,
        max_queue: int,
        model_name: str,
        language: str | None,
        task: str,
        whisper_device: str,
        no_fp16: bool,
        model_dir: str | None,
    ):
        super().__init__(daemon=False)
        self.final_queue: queue.Queue[TranscriptionJob | None] = queue.Queue(
            maxsize=max_queue
        )
        self.model_name = model_name
        self.language = language
        self.task = task
        self.whisper_device = whisper_device
        self.no_fp16 = no_fp16
        self.model_dir = model_dir
        self.ready = threading.Event()
        self.startup_error: str | None = None
        self._model = None
        self._device = "cpu"
        self._fp16 = False
        self._stop_requested = False
        self._partial_lock = threading.Lock()
        self._latest_partial: TranscriptionJob | None = None
        self._last_partial_text: dict[int, str] = {}

    def submit_final(self, segment: AudioSegment, timeout: float = 0.5) -> bool:
        try:
            self.final_queue.put(TranscriptionJob(kind="final", segment=segment), timeout=timeout)
            return True
        except queue.Full:
            return False

    def submit_partial(self, segment: AudioSegment) -> None:
        with self._partial_lock:
            self._latest_partial = TranscriptionJob(kind="partial", segment=segment)

    def stop(self) -> None:
        self._stop_requested = True
        self.final_queue.put(None)

    def _transcribe(self, segment: AudioSegment) -> str:
        audio = segment.samples.astype(np.float32) / 32768.0
        try:
            result = self._model.transcribe(
                audio,
                language=self.language,
                task=self.task,
                fp16=self._fp16,
                temperature=0.0,
                condition_on_previous_text=False,
                verbose=False,
            )
            return str(result.get("text", "")).strip()
        except TypeError:
            result = self._model.transcribe(
                audio,
                language=self.language,
                task=self.task,
                fp16=self._fp16,
            )
            return str(result.get("text", "")).strip()

    def run(self) -> None:
        try:
            import torch
            import whisper

            if self.whisper_device == "auto":
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self._device = self.whisper_device

            self._fp16 = self._device == "cuda" and not self.no_fp16
            self._model = whisper.load_model(
                self.model_name,
                device=self._device,
                download_root=self.model_dir,
            )
            print(
                f"Whisper model loaded: {self.model_name} on {self._device} "
                f"(fp16={'on' if self._fp16 else 'off'})",
                flush=True,
            )
        except Exception as exc:
            self.startup_error = str(exc)
            self.ready.set()
            return

        self.ready.set()

        while True:
            job: TranscriptionJob | None = None
            try:
                job = self.final_queue.get(timeout=0.05)
            except queue.Empty:
                pass

            if job is None and self._stop_requested:
                break

            if job is not None:
                try:
                    text = self._transcribe(job.segment)
                except Exception as exc:
                    print(
                        f"[utterance {job.segment.index}] final transcription failed: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                else:
                    stamp = time.strftime("%H:%M:%S", time.localtime(job.segment.end_time))
                    if text:
                        print(f"[{stamp}] {text}", flush=True)
                    else:
                        print(f"[{stamp}] <no speech recognized>", flush=True)
                self._last_partial_text.pop(job.segment.index, None)

            with self._partial_lock:
                partial_job = self._latest_partial
                self._latest_partial = None

            if partial_job is None:
                continue

            try:
                text = self._transcribe(partial_job.segment)
            except Exception as exc:
                print(
                    f"[utterance {partial_job.segment.index}] partial transcription failed: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                continue

            if not text:
                continue
            prev = self._last_partial_text.get(partial_job.segment.index)
            if text != prev:
                print(f"[partial {partial_job.segment.index}] {text}", flush=True)
                self._last_partial_text[partial_job.segment.index] = text


def read_exact(pipe, size: int) -> bytes:
    """Read exactly size bytes from a pipe unless EOF is reached."""
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        data = pipe.read(remaining)
        if not data:
            break
        chunks.append(data)
        remaining -= len(data)
    return b"".join(chunks)


class ManualCommandReader(threading.Thread):
    def __init__(self, out_queue: queue.Queue[str]):
        super().__init__(daemon=True)
        self.out_queue = out_queue

    def run(self) -> None:
        while True:
            line = sys.stdin.readline()
            if not line:
                return
            cmd = line.strip().lower()
            if cmd in {"/end", "end"}:
                self.out_queue.put("end")
            elif cmd in {"/quit", "/exit", "quit", "exit"}:
                self.out_queue.put("quit")
                return


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if shutil.which("arecord") is None:
        print("Error: arecord not found. Install ALSA utilities.", file=sys.stderr)
        return 1

    if args.list_devices:
        result = run(["arecord", "-l"])
        if result.returncode != 0:
            print(result.stderr.strip(), file=sys.stderr)
            return result.returncode
        print(result.stdout.rstrip())
        return 0

    if args.rate <= 0:
        print("Error: --rate must be > 0.", file=sys.stderr)
        return 2
    if args.frame_ms <= 0:
        print("Error: --frame-ms must be > 0.", file=sys.stderr)
        return 2
    if (args.rate * args.frame_ms) % 1000 != 0:
        print(
            "Error: --rate * --frame-ms must be divisible by 1000 "
            f"(got {args.rate} * {args.frame_ms}).",
            file=sys.stderr,
        )
        return 2
    if args.silence_seconds <= 0:
        print("Error: --silence-seconds must be > 0.", file=sys.stderr)
        return 2
    if args.max_queue <= 0:
        print("Error: --max-queue must be > 0.", file=sys.stderr)
        return 2
    if args.partial_interval <= 0:
        print("Error: --partial-interval must be > 0.", file=sys.stderr)
        return 2
    if args.partial_min_seconds <= 0:
        print("Error: --partial-min-seconds must be > 0.", file=sys.stderr)
        return 2
    if args.partial_window_seconds < 0:
        print("Error: --partial-window-seconds must be >= 0.", file=sys.stderr)
        return 2

    frame_samples = args.rate * args.frame_ms // 1000
    frame_bytes = frame_samples * 2

    segmenter = EnergySegmenter(
        rate=args.rate,
        frame_ms=args.frame_ms,
        silence_seconds=args.silence_seconds,
        pre_roll_seconds=args.pre_roll_seconds,
        min_speech_seconds=args.min_speech_seconds,
        trigger_frames=args.trigger_frames,
        energy_threshold=args.energy_threshold,
        energy_multiplier=args.energy_multiplier,
        min_rms=args.min_rms,
        calibration_seconds=args.calibration_seconds,
    )

    worker = WhisperWorker(
        max_queue=args.max_queue,
        model_name=args.model,
        language=args.language,
        task=args.task,
        whisper_device=args.whisper_device,
        no_fp16=args.no_fp16,
        model_dir=args.model_dir,
    )
    worker.start()
    worker.ready.wait(timeout=120.0)
    if worker.startup_error:
        print("Error: failed to initialize Whisper worker.", file=sys.stderr)
        print(f"Details: {worker.startup_error}", file=sys.stderr)
        print(
            "Tip: if you see missing CUDA libs (e.g. libcupti.so), either install "
            "CUDA runtime libs or use a CPU-only PyTorch build in your venv.",
            file=sys.stderr,
        )
        return 1
    if not worker.ready.is_set():
        print("Error: timed out while loading Whisper model.", file=sys.stderr)
        return 1

    cmd = [
        "arecord",
        "-D",
        args.device,
        "-f",
        "S16_LE",
        "-c",
        "1",
        "-r",
        str(args.rate),
        "-t",
        "raw",
        "-q",
    ]

    print(
        "Listening... "
        f"(device={args.device}, rate={args.rate}, silence_stop={args.silence_seconds}s)",
        flush=True,
    )
    if args.energy_threshold <= 0:
        print(
            "Using adaptive VAD threshold. Speak after startup calibration.",
            flush=True,
        )
        print(
            f"Calibrating for {args.calibration_seconds:.1f}s...",
            flush=True,
        )

    next_utterance_id = 1
    active_utterance_id: int | None = None
    last_partial_submit = 0.0
    command_queue: queue.Queue[str] = queue.Queue()
    command_reader: ManualCommandReader | None = None
    if args.manual_end:
        if sys.stdin.isatty():
            command_reader = ManualCommandReader(command_queue)
            command_reader.start()
            print(
                "Manual command enabled: type '/end' + Enter to finalize current "
                "utterance, '/quit' to stop.",
                flush=True,
            )
        else:
            print(
                "Manual end requested, but stdin is not a TTY. Ignoring.",
                file=sys.stderr,
            )

    proc: subprocess.Popen[bytes] | None = None
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        if proc.stdout is None:
            print("Error: failed to open arecord stdout pipe.", file=sys.stderr)
            return 1

        last_debug = 0.0
        calibration_announced = False
        while True:
            chunk = read_exact(proc.stdout, frame_bytes)
            if len(chunk) < frame_bytes:
                break

            frame = np.frombuffer(chunk, dtype=np.int16).copy()
            now = time.time()
            started, segment, rms, threshold = segmenter.process(frame, now)

            while True:
                try:
                    cmd_text = command_queue.get_nowait()
                except queue.Empty:
                    break
                if cmd_text == "quit":
                    raise KeyboardInterrupt
                if cmd_text == "end":
                    forced = segmenter.flush(now)
                    if forced is None:
                        print("No active utterance to force-end.", flush=True)
                    else:
                        if active_utterance_id is None:
                            active_utterance_id = next_utterance_id
                            next_utterance_id += 1
                        forced.index = active_utterance_id
                        if worker.submit_final(forced):
                            print(
                                f"[utterance {forced.index}] forced final "
                                f"({forced.duration:.2f}s)",
                                flush=True,
                            )
                        else:
                            print(
                                f"[utterance {forced.index}] dropped: final queue full",
                                file=sys.stderr,
                                flush=True,
                            )
                        active_utterance_id = None

            if (
                args.energy_threshold <= 0
                and segmenter.calibration_done
                and not calibration_announced
            ):
                noise = segmenter.noise_rms if segmenter.noise_rms is not None else 0.0
                print(
                    "Calibration done: "
                    f"noise_rms={noise:.1f}, threshold={segmenter.current_threshold():.1f}",
                    flush=True,
                )
                calibration_announced = True

            if started:
                if active_utterance_id is None:
                    active_utterance_id = next_utterance_id
                    next_utterance_id += 1
                print(f"Speech detected... [utterance {active_utterance_id}]", flush=True)
                last_partial_submit = 0.0

            if args.debug_vad and (now - last_debug) > 1.0:
                state = "speech" if segmenter.in_speech else "idle"
                if not segmenter.calibration_done and args.energy_threshold <= 0:
                    state = "calibrating"
                print(
                    f"VAD state={state} rms={rms:.1f} threshold={threshold:.1f}",
                    flush=True,
                )
                last_debug = now

            if (
                segmenter.in_speech
                and active_utterance_id is not None
                and (now - last_partial_submit) >= args.partial_interval
            ):
                partial_segment = segmenter.snapshot_active(
                    now, max_seconds=args.partial_window_seconds
                )
                if (
                    partial_segment is not None
                    and partial_segment.duration >= args.partial_min_seconds
                ):
                    partial_segment.index = active_utterance_id
                    worker.submit_partial(partial_segment)
                    last_partial_submit = now

            if segment is not None:
                if active_utterance_id is None:
                    active_utterance_id = next_utterance_id
                    next_utterance_id += 1
                segment.index = active_utterance_id
                if worker.submit_final(segment):
                    print(
                        f"[utterance {segment.index}] final queued "
                        f"({segment.duration:.2f}s)",
                        flush=True,
                    )
                else:
                    print(
                        f"[utterance {segment.index}] dropped: final queue full",
                        file=sys.stderr,
                        flush=True,
                    )
                active_utterance_id = None
    except KeyboardInterrupt:
        print("\nStopping...", flush=True)
    finally:
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
                    print(f"arecord: {err}", file=sys.stderr)

        last_segment = segmenter.flush(time.time())
        if last_segment is not None:
            if active_utterance_id is None:
                active_utterance_id = next_utterance_id
                next_utterance_id += 1
            last_segment.index = active_utterance_id
            if worker.submit_final(last_segment):
                print(
                    f"[utterance {last_segment.index}] final queued "
                    f"({last_segment.duration:.2f}s)",
                    flush=True,
                )
            else:
                print(
                    f"[utterance {last_segment.index}] dropped: final queue full",
                    file=sys.stderr,
                    flush=True,
                )

        worker.stop()
        worker.join()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

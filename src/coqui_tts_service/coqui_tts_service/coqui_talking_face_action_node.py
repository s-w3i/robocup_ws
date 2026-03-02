#!/usr/bin/env python3

import math
import os
import random
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path
from typing import Callable, List, Tuple

DEFAULT_COQUI_SITE_PACKAGES = "/home/usern/coqui-venv/lib/python3.10/site-packages"
SYSTEM_SITE_PATH_PREFIXES = (
    "/usr/lib/python3/dist-packages",
    "/usr/local/lib/python3.10/dist-packages",
)


def prepend_site_packages(path: str) -> bool:
    p = Path(path).expanduser().resolve()
    if not p.is_dir():
        return False
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)
    return True


def activate_coqui_site_packages(path: str, isolate: bool) -> tuple[bool, int]:
    """Prepend Coqui site-packages and optionally isolate from system site-packages."""
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
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import String

from coqui_tts_interfaces.action import SpeakText
from coqui_tts_interfaces.srv import SynthesizeSpeech


class CuteFacePlayer:
    def __init__(
        self,
        logger,
        width: int,
        height: int,
        fps: int,
        emotion_speed_scale: float = 1.0,
    ) -> None:
        self._logger = logger
        self.width = max(640, int(width))
        self.height = max(480, int(height))
        self.fps = max(60, fps)
        self._emotion_speed_scale = max(0.1, float(emotion_speed_scale))
        self._window_flags = 0

        self._pygame = None
        self._screen = None
        self._clock = None
        self._text_font = None
        self._ready = False
        self._init_error = ""
        self._display_ready = False
        self._ui_failed = False
        self._render_thread = None
        self._render_stop = threading.Event()
        self._display_init_done = threading.Event()
        self._state_lock = threading.Lock()
        self._mouth_target = 0.12
        self._mouth_display = 0.12
        self._is_talking = False
        self._robot_status = "sleep"
        self._idle_emotion = "sleepy"
        self._idle_emotion_until = float("inf")
        self._text_preview = ""
        self._next_blink = time.perf_counter() + random.uniform(1.2, 2.8)
        self._blink_end = 0.0

    def _scale_interval(self, seconds: float, min_seconds: float = 0.1) -> float:
        return max(min_seconds, float(seconds) / self._emotion_speed_scale)

    @staticmethod
    def _sanitize_text(text: str) -> str:
        cleaned = " ".join(text.split()).strip()
        return cleaned[:140]

    def _ensure_ready(self) -> bool:
        if self._ready:
            return True

        try:
            import pygame  # pylint: disable=import-outside-toplevel
        except Exception as exc:  # pragma: no cover
            self._init_error = f"pygame import failed: {exc}"
            self._logger.error(self._init_error)
            return False

        self._pygame = pygame
        try:
            pygame.mixer.pre_init(22050, -16, 1, 512)
            pygame.init()
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=22050, channels=1)
        except Exception as exc:  # pragma: no cover
            self._init_error = f"pygame audio init failed: {exc}"
            self._logger.error(self._init_error)
            return False

        self._clock = pygame.time.Clock()
        self._ready = True
        return True

    def start(self, wait_for_ui: bool = False, timeout_sec: float = 2.0) -> bool:
        if not self._ensure_ready():
            return False
        if self._ui_failed:
            return True
        if self._render_thread is not None and self._render_thread.is_alive():
            if wait_for_ui and not self._display_init_done.is_set():
                self._display_init_done.wait(timeout=max(0.1, timeout_sec))
            return True

        self._render_stop.clear()
        self._display_init_done.clear()
        self._render_thread = threading.Thread(target=self._render_loop, daemon=True)
        self._render_thread.start()
        if wait_for_ui:
            self._display_init_done.wait(timeout=max(0.1, timeout_sec))
        return True

    def stop(self) -> None:
        self._render_stop.set()
        if self._render_thread is not None and self._render_thread.is_alive():
            self._render_thread.join(timeout=2.0)
        if self._ready and self._pygame is not None:
            try:
                self._pygame.mixer.music.stop()
            except Exception:
                pass
            try:
                self._pygame.quit()
            except Exception:
                pass

    def _init_display(self) -> None:
        if self._display_ready or self._ui_failed:
            self._display_init_done.set()
            return
        try:
            self._window_flags = int(getattr(self._pygame, "RESIZABLE", 0))

            info = self._pygame.display.Info()
            display_w = int(getattr(info, "current_w", 0) or 0)
            display_h = int(getattr(info, "current_h", 0) or 0)
            if display_w > 0 and display_h > 0:
                # Fit to display size with margins so it appears as a large, sharp window.
                self.width = max(640, int(display_w * 0.96))
                self.height = max(480, int(display_h * 0.90))

            self._screen = self._pygame.display.set_mode((self.width, self.height), self._window_flags)
            self._pygame.display.set_caption("EVA")
            self._text_font = self._pygame.font.SysFont(
                "DejaVu Sans",
                max(18, min(34, self.width // 22)),
                bold=True,
            )
            self._display_ready = True
            self._logger.info("Talking-face UI is active.")
        except Exception as exc:
            self._ui_failed = True
            self._display_ready = False
            self._init_error = f"Display unavailable ({exc}). Running audio-only."
            self._logger.warning(self._init_error)
        finally:
            self._display_init_done.set()

    def _resize_window(self, width: int, height: int) -> None:
        if not self._display_ready:
            return

        new_w = max(640, int(width))
        new_h = max(480, int(height))
        if new_w == self.width and new_h == self.height:
            return

        self.width = new_w
        self.height = new_h
        self._screen = self._pygame.display.set_mode((self.width, self.height), self._window_flags)
        self._text_font = self._pygame.font.SysFont(
            "DejaVu Sans",
            max(18, min(34, self.width // 22)),
            bold=True,
        )

    def set_idle(self, text: str = "") -> None:
        with self._state_lock:
            self._mouth_target = 0.12
            self._is_talking = False
            now = time.perf_counter()
            if self._robot_status in ("idle", "operating"):
                self._idle_emotion_until = min(
                    self._idle_emotion_until,
                    now + self._scale_interval(random.uniform(0.4, 1.1), min_seconds=0.08),
                )
            else:
                self._idle_emotion_until = now
            self._apply_robot_status_locked(now)
            self._text_preview = self._sanitize_text(text)

    def set_robot_status(self, status: str) -> bool:
        normalized = status.strip().lower()
        if normalized not in ("sleep", "listening", "idle", "operating"):
            return False
        with self._state_lock:
            if self._robot_status == normalized:
                return False
            self._robot_status = normalized
            now = time.perf_counter()
            self._idle_emotion_until = now
            self._apply_robot_status_locked(now)
        return True

    def _apply_robot_status_locked(self, now: float) -> None:
        if self._robot_status == "sleep":
            self._idle_emotion = "sleepy"
            self._idle_emotion_until = float("inf")
            return
        if self._robot_status == "listening":
            self._idle_emotion = "listening"
            self._idle_emotion_until = float("inf")
            return
        if now >= self._idle_emotion_until:
            self._choose_next_idle_emotion_locked(now)

    def _set_speaking_mouth(self, text: str, mouth_open: float) -> None:
        with self._state_lock:
            self._is_talking = True
            self._text_preview = self._sanitize_text(text)
            self._mouth_target = float(max(0.12, min(1.0, mouth_open)))

    def _choose_next_idle_emotion_locked(self, now: float) -> None:
        emotions = [
            "neutral",
            "smile_big",
            "happy_open",
            "surprised",
            "wink_left",
            "wink_right",
        ]
        pool = [emotion for emotion in emotions if emotion != self._idle_emotion]
        self._idle_emotion = random.choice(pool) if pool else "neutral"

        duration_map = {
            "neutral": 2.8,
            "smile_big": 2.2,
            "happy_open": 1.7,
            "surprised": 1.2,
            "wink_left": 0.85,
            "wink_right": 0.85,
        }
        duration = duration_map.get(self._idle_emotion, 2.0) + random.uniform(-0.25, 0.45)
        self._idle_emotion_until = now + self._scale_interval(duration, min_seconds=0.25)

    def _extract_envelope(self, wav_path: str) -> Tuple[List[float], float]:
        try:
            with wave.open(wav_path, "rb") as wav_file:
                sample_rate = wav_file.getframerate()
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                frame_count = wav_file.getnframes()
                raw = wav_file.readframes(frame_count)
        except Exception:  # pragma: no cover
            return [0.2], 0.0

        if sample_rate <= 0 or frame_count <= 0:
            return [0.2], 0.0

        if sample_width == 1:
            samples = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
            samples = (samples - 128.0) / 128.0
        elif sample_width == 2:
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sample_width == 4:
            samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            return [0.2], frame_count / float(sample_rate)

        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1)

        duration = samples.size / float(sample_rate)
        hop = max(1, int(sample_rate / self.fps))
        chunks = max(1, samples.size // hop)
        trimmed = samples[: chunks * hop]
        if trimmed.size == 0:
            return [0.2], duration

        blocks = trimmed.reshape(chunks, hop)
        rms = np.sqrt(np.mean(np.square(blocks), axis=1))
        peak = float(np.max(rms))
        if peak > 1e-7:
            rms = rms / peak

        envelope = np.clip(0.12 + (0.88 * rms), 0.12, 1.0).tolist()
        return envelope, duration

    @staticmethod
    def _split_lines(text: str, max_chars: int, max_lines: int) -> List[str]:
        words = text.split()
        if not words:
            return []

        lines = []
        current = ""
        idx = 0
        while idx < len(words) and len(lines) < max_lines:
            word = words[idx]
            candidate = f"{current} {word}".strip()
            if current and len(candidate) > max_chars:
                lines.append(current)
                current = ""
                if len(lines) == max_lines:
                    break
                continue
            if len(word) > max_chars and not current:
                lines.append(word[: max_chars - 3] + "...")
                idx += 1
                continue
            current = candidate
            idx += 1

        if len(lines) < max_lines and current:
            lines.append(current)

        if idx < len(words) and lines:
            tail = lines[-1]
            lines[-1] = (tail[: max(1, max_chars - 3)] + "...") if len(tail) >= max_chars else (tail + "...")

        return lines[:max_lines]

    def _draw_face(
        self,
        mouth_open: float,
        blinking: bool,
        is_talking: bool,
        idle_emotion: str,
        text_preview: str,
    ) -> None:
        if not self._display_ready:
            return

        pg = self._pygame
        screen = self._screen
        center_x = self.width // 2
        center_y = self.height // 2
        now_t = time.perf_counter()

        # Flat Kirby-like face background with subtle polish.
        base_color = (231, 169, 176)
        screen.fill(base_color)
        face_polish = pg.Surface((self.width, self.height), pg.SRCALPHA)
        pg.draw.ellipse(
            face_polish,
            (255, 238, 242, 58),
            pg.Rect(
                -int(self.width * 0.08),
                -int(self.height * 0.10),
                int(self.width * 0.90),
                int(self.height * 0.72),
            ),
        )
        pg.draw.ellipse(
            face_polish,
            (183, 102, 128, 34),
            pg.Rect(
                int(self.width * 0.52),
                int(self.height * 0.50),
                int(self.width * 0.62),
                int(self.height * 0.58),
            ),
        )
        screen.blit(face_polish, (0, 0))

        eye_y = center_y - int(self.height * 0.11)
        eye_offset = int(self.width * 0.18)
        eye_w = max(54, int(self.width * 0.14))
        eye_h = max(110, int(self.height * 0.34))
        sleep_breathe_offset = 0

        mood = idle_emotion if not is_talking else "talking"
        if mood == "sleepy":
            eye_h = max(72, int(eye_h * 0.68))
            eye_y += int(self.height * 0.02)
            sleep_breathe_offset = int(self.height * 0.008 * math.sin(now_t * 1.2))
        elif mood == "listening":
            eye_h = max(82, int(eye_h * 0.86))
            eye_w = max(48, int(eye_w * 0.94))
            eye_y -= int(self.height * 0.01)
            # Listening mode: subtle micro motion makes the status easier to recognize.
            eye_y += int(self.height * 0.004 * math.sin(now_t * 4.2))
        elif mood == "surprised":
            eye_h = int(eye_h * 1.12)
            eye_w = int(eye_w * 0.95)

        def draw_closed_eye(x_center: int, y_center: int, width_px: int) -> None:
            pg.draw.line(
                screen,
                (15, 15, 20),
                (x_center - width_px // 2, y_center),
                (x_center + width_px // 2, y_center),
                9,
            )

        def draw_sleep_eye(x_center: int, y_center: int, width_px: int, height_px: int) -> None:
            curve_h = max(14, int(height_px * 0.16))
            eye_rect = pg.Rect(0, 0, width_px, curve_h)
            eye_rect.center = (x_center, y_center + max(2, curve_h // 4))
            pg.draw.arc(screen, (20, 20, 24), eye_rect, 0.08 * math.pi, 0.92 * math.pi, 8)

        def draw_open_eye(x_center: int, y_center: int, width_px: int, height_px: int) -> None:
            eye_rect = pg.Rect(0, 0, width_px, height_px)
            eye_rect.center = (x_center, y_center)
            pg.draw.ellipse(screen, (8, 8, 10), eye_rect)
            inner_shadow = eye_rect.inflate(-max(6, width_px // 8), -max(8, height_px // 9))
            pg.draw.ellipse(screen, (18, 16, 23), inner_shadow)

            highlight_w = max(24, int(width_px * 0.55))
            highlight_h = max(44, int(height_px * 0.44))
            highlight = pg.Rect(0, 0, highlight_w, highlight_h)
            highlight.center = (
                eye_rect.centerx - int(width_px * 0.08),
                eye_rect.top + int(height_px * 0.23),
            )
            pg.draw.ellipse(screen, (236, 236, 236), highlight)
            small_glint = pg.Rect(0, 0, max(8, highlight_w // 4), max(10, highlight_h // 4))
            small_glint.center = (highlight.centerx + max(3, width_px // 18), highlight.centery + max(4, height_px // 16))
            pg.draw.ellipse(screen, (255, 255, 255), small_glint)

            blue_w = max(20, int(width_px * 0.60))
            blue_h = max(34, int(height_px * 0.34))
            blue_reflect = pg.Rect(0, 0, blue_w, blue_h)
            blue_reflect.center = (
                eye_rect.centerx,
                eye_rect.bottom - int(height_px * 0.16),
            )
            pg.draw.ellipse(screen, (6, 120, 201), blue_reflect)
            blue_reflect_inner = blue_reflect.inflate(-max(4, blue_w // 5), -max(4, blue_h // 5))
            pg.draw.ellipse(screen, (22, 158, 233), blue_reflect_inner)

        left_x = center_x - eye_offset
        right_x = center_x + eye_offset
        if mood == "sleepy":
            draw_sleep_eye(left_x, eye_y + sleep_breathe_offset, eye_w, eye_h)
            draw_sleep_eye(right_x, eye_y + sleep_breathe_offset, eye_w, eye_h)
        elif mood == "wink_left":
            draw_closed_eye(left_x, eye_y, eye_w)
            draw_open_eye(right_x, eye_y, eye_w, eye_h)
        elif mood == "wink_right":
            draw_open_eye(left_x, eye_y, eye_w, eye_h)
            draw_closed_eye(right_x, eye_y, eye_w)
        elif blinking:
            draw_closed_eye(left_x, eye_y, eye_w)
            draw_closed_eye(right_x, eye_y, eye_w)
        else:
            draw_open_eye(left_x, eye_y, eye_w, eye_h)
            draw_open_eye(right_x, eye_y, eye_w, eye_h)

        if mood == "listening":
            listen_overlay = pg.Surface((self.width, self.height), pg.SRCALPHA)
            pulse = 0.5 + 0.5 * math.sin(now_t * 3.0)
            forehead_x = center_x
            forehead_y = center_y - int(self.height * 0.30)
            glow_w = max(90, int(self.width * (0.19 + 0.03 * pulse)))
            glow_h = max(48, int(self.height * (0.09 + 0.02 * pulse)))
            glow_rect = pg.Rect(0, 0, glow_w, glow_h)
            glow_rect.center = (forehead_x, forehead_y)
            pg.draw.ellipse(listen_overlay, (84, 144, 220, int(46 + 48 * pulse)), glow_rect)
            halo_rect = glow_rect.inflate(max(16, glow_w // 3), max(12, glow_h // 2))
            pg.draw.ellipse(
                listen_overlay,
                (150, 208, 255, int(26 + 34 * pulse)),
                halo_rect,
                width=max(2, int(self.width * 0.003)),
            )
            screen.blit(listen_overlay, (0, 0))

            ring_inner = max(14, int(self.width * 0.018))
            ring_outer = max(22, int(self.width * 0.028))
            ring_width = max(3, int(self.width * 0.003))
            ring_color = (24, 84, 152)
            ring_highlight = (116, 170, 226)
            ear_offset = int(self.width * 0.31)
            ear_y = eye_y + int(self.height * 0.01)
            for side in (-1, 1):
                ear_x = center_x + (side * ear_offset)
                pg.draw.circle(screen, ring_color, (ear_x, ear_y), ring_inner, ring_width)
                pg.draw.circle(screen, ring_highlight, (ear_x, ear_y), ring_outer, ring_width)
                for idx in range(4):
                    phase = (now_t * 0.65 + idx * 0.23) % 1.0
                    radius = max(ring_outer + 2, ring_outer + int(self.width * (0.02 + 0.10 * phase)))
                    alpha = int(170 * (1.0 - phase))
                    wave = pg.Surface((self.width, self.height), pg.SRCALPHA)
                    pg.draw.circle(
                        wave,
                        (112, 176, 245, alpha),
                        (ear_x, ear_y),
                        radius,
                        max(2, ring_width - 1),
                    )
                    screen.blit(wave, (0, 0))

            # Animated listening bars on forehead for a clear "audio input" cue.
            bars_color = (42, 102, 176)
            bars_glow = (150, 208, 255)
            bar_w = max(7, int(self.width * 0.009))
            bar_gap = max(6, int(self.width * 0.006))
            bar_count = 5
            base_y = forehead_y + int(self.height * 0.09)
            total_w = (bar_count * bar_w) + ((bar_count - 1) * bar_gap)
            start_x = center_x - (total_w // 2)
            for idx in range(bar_count):
                osc = 0.5 + 0.5 * math.sin((now_t * 7.0) + idx * 0.9)
                bar_h = max(12, int(self.height * (0.028 + 0.05 * osc)))
                rect = pg.Rect(
                    start_x + idx * (bar_w + bar_gap),
                    base_y - bar_h,
                    bar_w,
                    bar_h,
                )
                pg.draw.rect(screen, bars_color, rect, border_radius=max(3, bar_w // 2))
                glow_rect = rect.inflate(max(2, bar_w // 3), max(2, bar_w // 4))
                pg.draw.rect(screen, bars_glow, glow_rect, width=1, border_radius=max(3, bar_w // 2))

        if mood == "sleepy":
            sleep_overlay = pg.Surface((self.width, self.height), pg.SRCALPHA)
            breathe = 0.5 + 0.5 * math.sin(now_t * 1.2)
            bubble_x = center_x + int(self.width * 0.13)
            bubble_y = center_y + int(self.height * 0.02) + sleep_breathe_offset
            bubble_r = max(10, int(self.width * (0.013 + 0.010 * breathe)))
            pg.draw.circle(sleep_overlay, (220, 241, 255, 180), (bubble_x, bubble_y), bubble_r)
            highlight_r = max(3, bubble_r // 3)
            pg.draw.circle(
                sleep_overlay,
                (255, 255, 255, 190),
                (bubble_x - max(3, bubble_r // 4), bubble_y - max(2, bubble_r // 4)),
                highlight_r,
            )
            for idx in range(3):
                phase = (now_t * 0.22 + idx * 0.34) % 1.0
                alpha = int(60 + (140 * (1.0 - phase)))
                z_w = max(10, int(self.width * (0.011 + 0.004 * (1.0 - phase))))
                z_h = max(12, int(self.height * (0.022 + 0.006 * (1.0 - phase))))
                z_x = bubble_x + int(self.width * (0.04 + idx * 0.03))
                z_y = bubble_y - int(self.height * (0.10 + idx * 0.05 + phase * 0.18))
                z_color = (96, 136, 180, alpha)
                stroke = max(2, int(self.width * 0.0025))
                pg.draw.line(sleep_overlay, z_color, (z_x, z_y), (z_x + z_w, z_y), stroke)
                pg.draw.line(
                    sleep_overlay,
                    z_color,
                    (z_x + z_w, z_y),
                    (z_x, z_y + z_h),
                    stroke,
                )
                pg.draw.line(
                    sleep_overlay,
                    z_color,
                    (z_x, z_y + z_h),
                    (z_x + z_w, z_y + z_h),
                    stroke,
                )
            screen.blit(sleep_overlay, (0, 0))

        # Cheeks.
        cheek_color = (226, 73, 104)
        cheek_w = int(self.width * 0.13)
        cheek_h = int(self.height * 0.085)
        cheek_y = center_y + int(self.height * 0.14) + sleep_breathe_offset
        cheek_offset = int(self.width * 0.27)
        left_cheek = pg.Rect(0, 0, cheek_w, cheek_h)
        right_cheek = pg.Rect(0, 0, cheek_w, cheek_h)
        left_cheek.center = (center_x - cheek_offset, cheek_y)
        right_cheek.center = (center_x + cheek_offset, cheek_y)
        pg.draw.ellipse(screen, cheek_color, left_cheek)
        pg.draw.ellipse(screen, cheek_color, right_cheek)
        left_cheek_shine = left_cheek.inflate(-max(8, cheek_w // 3), -max(8, cheek_h // 3))
        right_cheek_shine = right_cheek.inflate(-max(8, cheek_w // 3), -max(8, cheek_h // 3))
        pg.draw.ellipse(screen, (241, 104, 132), left_cheek_shine)
        pg.draw.ellipse(screen, (241, 104, 132), right_cheek_shine)

        mouth_center_y = center_y + int(self.height * 0.21) + sleep_breathe_offset
        if is_talking:
            mouth_norm = max(0.0, min(1.0, (mouth_open - 0.12) / 0.88))
        else:
            base_idle_mouth = {
                "neutral": 0.10,
                "smile_big": 0.18,
                "happy_open": 0.38,
                "surprised": 0.72,
                "sleepy": 0.05,
                "listening": 0.08,
                "wink_left": 0.16,
                "wink_right": 0.16,
            }.get(idle_emotion, 0.10)
            wobble = 0.04 * math.sin(time.perf_counter() * 2.7)
            mouth_norm = max(0.0, min(1.0, base_idle_mouth + wobble))

        if not is_talking and idle_emotion == "surprised":
            o_width = int(self.width * 0.12)
            o_height = int(self.height * 0.18)
            mouth_rect = pg.Rect(0, 0, o_width, o_height)
            mouth_rect.center = (center_x, mouth_center_y + int(self.height * 0.01))
            pg.draw.ellipse(screen, (105, 30, 35), mouth_rect)
            inner = mouth_rect.inflate(-max(6, o_width // 3), -max(8, o_height // 3))
            if inner.width > 0 and inner.height > 0:
                pg.draw.ellipse(screen, (199, 57, 62), inner)
                inner_shine = inner.inflate(-max(4, inner.width // 4), -max(4, inner.height // 4))
                inner_shine.centery += max(2, inner.height // 10)
                pg.draw.ellipse(screen, (220, 84, 95), inner_shine)
        elif mouth_norm < 0.18:
            smile_rect = pg.Rect(0, 0, int(self.width * 0.18), int(self.height * 0.10))
            smile_rect.center = (center_x, mouth_center_y)
            pg.draw.arc(screen, (115, 33, 40), smile_rect, 0.08 * math.pi, 0.92 * math.pi, 7)
            smile_glow = pg.Surface((self.width, self.height), pg.SRCALPHA)
            pg.draw.arc(smile_glow, (196, 84, 102, 120), smile_rect.inflate(8, 6), 0.08 * math.pi, 0.92 * math.pi, 3)
            screen.blit(smile_glow, (0, 0))
        else:
            mouth_width = int(self.width * (0.14 + (0.05 * mouth_norm)))
            mouth_height = int(self.height * (0.11 + (0.10 * mouth_norm)))
            mouth_rect = pg.Rect(0, 0, mouth_width, mouth_height)
            mouth_rect.center = (center_x, mouth_center_y)
            pg.draw.ellipse(screen, (105, 30, 35), mouth_rect)

            inner = mouth_rect.inflate(-max(8, mouth_width // 5), -max(8, mouth_height // 2))
            if inner.width > 0 and inner.height > 0:
                inner.centery += max(2, mouth_height // 4)
                pg.draw.ellipse(screen, (199, 57, 62), inner)
                tongue_highlight = inner.inflate(-max(4, inner.width // 5), -max(4, inner.height // 3))
                tongue_highlight.centery += max(1, inner.height // 10)
                pg.draw.ellipse(screen, (221, 90, 108), tongue_highlight)
                mouth_top_shadow = inner.inflate(-max(3, inner.width // 8), -max(8, inner.height // 2))
                mouth_top_shadow.centery -= max(4, inner.height // 6)
                if mouth_top_shadow.width > 0 and mouth_top_shadow.height > 0:
                    pg.draw.ellipse(screen, (83, 21, 30), mouth_top_shadow)

        # Show transcript only while robot is speaking.
        if is_talking and text_preview:
            lines = self._split_lines(text_preview, max_chars=max(22, self.width // 22), max_lines=2)
            if lines:
                rendered = [self._text_font.render(line, True, (255, 255, 255)) for line in lines]
                line_h = rendered[0].get_height()
                box_w = max(surface.get_width() for surface in rendered) + 36
                box_h = (line_h * len(rendered)) + (len(rendered) - 1) * 4 + 22
                box_rect = pg.Rect(0, 0, box_w, box_h)
                box_rect.centerx = center_x
                box_rect.bottom = self.height - 20

                overlay = pg.Surface((self.width, self.height), pg.SRCALPHA)
                pg.draw.rect(overlay, (48, 20, 36, 158), box_rect, border_radius=14)
                top_band = box_rect.inflate(0, -int(box_rect.height * 0.45))
                top_band.height = max(8, box_rect.height // 3)
                pg.draw.rect(overlay, (88, 38, 63, 145), top_band, border_radius=14)
                screen.blit(overlay, (0, 0))

                y = box_rect.y + 11
                for surface in rendered:
                    x = center_x - (surface.get_width() // 2)
                    screen.blit(surface, (x, y))
                    y += surface.get_height() + 4

        pg.display.flip()

    def _render_loop(self) -> None:
        self._init_display()
        if not self._display_ready:
            return

        pg = self._pygame
        while not self._render_stop.is_set():
            try:
                for event in pg.event.get():
                    if event.type == pg.QUIT:
                        pg.display.iconify()
                    elif event.type == pg.VIDEORESIZE:
                        self._resize_window(event.w, event.h)
                    elif hasattr(pg, "WINDOWSIZECHANGED") and event.type == pg.WINDOWSIZECHANGED:
                        # SDL2 path on some platforms.
                        new_w = int(
                            getattr(event, "x", getattr(event, "w", getattr(event, "width", self.width)))
                        )
                        new_h = int(
                            getattr(event, "y", getattr(event, "h", getattr(event, "height", self.height)))
                        )
                        self._resize_window(new_w, new_h)

                now = time.perf_counter()
                if now >= self._next_blink:
                    self._blink_end = now + 0.14
                    self._next_blink = now + random.uniform(1.5, 3.5)
                blinking = now <= self._blink_end

                with self._state_lock:
                    if not self._is_talking:
                        self._apply_robot_status_locked(now)
                    mouth_target = self._mouth_target
                    is_talking = self._is_talking
                    idle_emotion = self._idle_emotion
                    text_preview = self._text_preview

                self._mouth_display = (0.68 * self._mouth_display) + (0.32 * mouth_target)
                self._draw_face(self._mouth_display, blinking, is_talking, idle_emotion, text_preview)
            except Exception as exc:
                self._ui_failed = True
                self._display_ready = False
                self._logger.warning(f"Talking-face UI stopped: {exc}. Continuing audio-only.")
                return

            if self._clock is not None:
                self._clock.tick(self.fps)
            else:
                time.sleep(0.02)

    def _play_audio_only(
        self,
        wav_path: str,
        text: str,
        should_cancel: Callable[[], bool],
        feedback_cb: Callable[[str, float], None],
    ) -> Tuple[bool, float, str]:
        envelope, duration = self._extract_envelope(wav_path)
        self._set_speaking_mouth(text, 0.16)
        start_time = time.perf_counter()
        try:
            process = subprocess.Popen(  # pylint: disable=consider-using-with
                ["aplay", "-q", wav_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
        except Exception as exc:
            return False, 0.0, f"Audio playback failed: {exc}"

        while process.poll() is None:
            elapsed = time.perf_counter() - start_time
            progress = min(1.0, elapsed / duration) if duration > 0.0 else 0.0
            if duration > 0.0 and envelope:
                idx = int((elapsed / duration) * (len(envelope) - 1))
                idx = max(0, min(idx, len(envelope) - 1))
                self._set_speaking_mouth(text, float(envelope[idx]))
            feedback_cb("playing", progress)
            if should_cancel():
                process.terminate()
                self.set_idle("")
                return False, elapsed, "Playback canceled."
            time.sleep(0.03)

        elapsed = time.perf_counter() - start_time
        self.set_idle("")
        if process.returncode != 0:
            error = (process.stderr.read() or "").strip()
            return False, elapsed, f"Audio playback failed: {error}"

        feedback_cb("playing", 1.0)
        return True, elapsed, "Playback finished."

    def play(
        self,
        wav_path: str,
        text: str,
        should_cancel: Callable[[], bool],
        feedback_cb: Callable[[str, float], None],
    ) -> Tuple[bool, float, str]:
        if not self.start():
            return self._play_audio_only(wav_path, text, should_cancel, feedback_cb)

        envelope, duration = self._extract_envelope(wav_path)
        self._set_speaking_mouth(text, 0.16)

        pg = self._pygame
        try:
            pg.mixer.music.load(wav_path)
            pg.mixer.music.play()
        except Exception:
            return self._play_audio_only(wav_path, text, should_cancel, feedback_cb)

        start_time = time.perf_counter()
        next_feedback_at = 0.0

        while pg.mixer.music.get_busy():
            elapsed = time.perf_counter() - start_time
            if should_cancel():
                pg.mixer.music.stop()
                self.set_idle("")
                return False, elapsed, "Playback canceled."

            if duration > 0.0 and envelope:
                idx = int((elapsed / duration) * (len(envelope) - 1))
                idx = max(0, min(idx, len(envelope) - 1))
                mouth_open = float(envelope[idx])
                progress = min(1.0, elapsed / duration)
            else:
                mouth_open = 0.16
                progress = 0.0

            self._set_speaking_mouth(text, mouth_open)

            if elapsed >= next_feedback_at:
                feedback_cb("playing", progress)
                next_feedback_at = elapsed + 0.10

            time.sleep(0.01)

        elapsed = time.perf_counter() - start_time
        feedback_cb("playing", 1.0)
        self.set_idle("")
        return True, elapsed, "Playback finished."


class CoquiTalkingFaceActionNode(Node):
    def __init__(self) -> None:
        super().__init__("coqui_talking_face_action_node")

        default_model = "tts_models/en/ljspeech/vits"
        self.declare_parameter("tts_executable", "/home/usern/coqui-venv/bin/tts")
        self.declare_parameter("fixed_output_path", "/home/usern/coqui-venv/tts_output/speech.wav")
        self.declare_parameter("model_name", default_model)
        self.declare_parameter("speaker_wav", "")
        self.declare_parameter("speaker_idx", "")
        self.declare_parameter("language_idx", "")
        self.declare_parameter("tts_device", "auto")
        self.declare_parameter("split_sentences", False)
        self.declare_parameter("warmup_enabled", True)
        self.declare_parameter("warmup_text", "service warmup")
        self.declare_parameter("face_width", 1280)
        self.declare_parameter("face_height", 720)
        self.declare_parameter("face_fps", 30)
        self.declare_parameter("face_emotion_speed_scale", 0.8)
        self.declare_parameter("extra_site_packages", DEFAULT_COQUI_SITE_PACKAGES)
        self.declare_parameter("isolate_site_packages", True)
        self.declare_parameter("robot_status_topic", "/robot_status")

        self.tts_executable = self.get_parameter("tts_executable").get_parameter_value().string_value
        self.fixed_output_path = self.get_parameter("fixed_output_path").get_parameter_value().string_value
        self.model_name = self.get_parameter("model_name").get_parameter_value().string_value
        self.speaker_wav = self.get_parameter("speaker_wav").get_parameter_value().string_value
        self.speaker_idx = self.get_parameter("speaker_idx").get_parameter_value().string_value
        self.language_idx = self.get_parameter("language_idx").get_parameter_value().string_value
        self.tts_device = self.get_parameter("tts_device").get_parameter_value().string_value.lower()
        self.split_sentences = self.get_parameter("split_sentences").get_parameter_value().bool_value
        self.warmup_enabled = self.get_parameter("warmup_enabled").get_parameter_value().bool_value
        self.warmup_text = self.get_parameter("warmup_text").get_parameter_value().string_value
        self.face_width = self.get_parameter("face_width").get_parameter_value().integer_value
        self.face_height = self.get_parameter("face_height").get_parameter_value().integer_value
        self.face_fps = self.get_parameter("face_fps").get_parameter_value().integer_value
        self.face_emotion_speed_scale = (
            self.get_parameter("face_emotion_speed_scale").get_parameter_value().double_value
        )
        self.extra_site_packages = self.get_parameter("extra_site_packages").get_parameter_value().string_value
        self.isolate_site_packages = (
            self.get_parameter("isolate_site_packages").get_parameter_value().bool_value
        )
        self.robot_status_topic = self.get_parameter("robot_status_topic").get_parameter_value().string_value
        added, removed = activate_coqui_site_packages(
            self.extra_site_packages, self.isolate_site_packages
        )

        self.coqui_python = str(Path(self.tts_executable).expanduser().resolve().parent / "python")
        self._build_runtime_env()
        self.runtime_device_hint = self._detect_runtime_device_hint()
        self.requested_device = self._resolve_requested_device()

        self.face_player = CuteFacePlayer(
            logger=self.get_logger(),
            width=self.face_width,
            height=self.face_height,
            fps=self.face_fps,
            emotion_speed_scale=self.face_emotion_speed_scale,
        )
        self.face_player.start(wait_for_ui=True, timeout_sec=4.0)
        self.face_player.set_idle("")
        self._robot_status = "sleep"
        status_qos = QoSProfile(depth=1)
        status_qos.reliability = QoSReliabilityPolicy.RELIABLE
        status_qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        self._robot_status_sub = self.create_subscription(
            String,
            self.robot_status_topic,
            self._robot_status_callback,
            status_qos,
        )
        self.face_player.set_robot_status(self._robot_status)

        self.tts_engine = None
        self.engine_error = ""
        self._load_engine()

        self._goal_lock = threading.Lock()
        self._goal_active = False
        self._synthesis_lock = threading.Lock()
        self._callback_group = ReentrantCallbackGroup()
        self._action_server = ActionServer(
            self,
            SpeakText,
            "/coqui_tts/speak",
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self._callback_group,
        )
        self.create_service(
            SynthesizeSpeech,
            "/coqui_tts/synthesize",
            self.handle_synthesize_request,
        )

        self.get_logger().info("Coqui talking-face action ready on /coqui_tts/speak")
        self.get_logger().info("Coqui TTS service ready on /coqui_tts/synthesize")
        self.get_logger().info(f"Fixed output path: {self.fixed_output_path}")
        self.get_logger().info(f"Resolved device: {self.requested_device}")
        self.get_logger().info(f"Model: {self.model_name}")
        self.get_logger().info(
            f"face_emotion_speed_scale: {self.face_emotion_speed_scale}"
        )
        self.get_logger().info(f"extra_site_packages: {self.extra_site_packages}")
        self.get_logger().info(
            f"isolate_site_packages: {self.isolate_site_packages} "
            f"(added={added}, removed={removed})"
        )
        self.get_logger().info(f"Robot status topic: {self.robot_status_topic}")

    def _robot_status_callback(self, msg: String) -> None:
        status = str(msg.data).strip().lower()
        if status not in ("sleep", "listening", "idle", "operating"):
            self.get_logger().warn(
                f"Ignoring invalid robot status '{status}' on {self.robot_status_topic}."
            )
            return
        if status == self._robot_status:
            return
        changed = self.face_player.set_robot_status(status)
        if changed:
            self._robot_status = status
            self.get_logger().info(f"Robot status changed to '{status}'.")

    def _build_runtime_env(self) -> None:
        lib_paths = []
        nvidia_lib_root = (
            Path(self.coqui_python).resolve().parent.parent
            / "lib"
            / "python3.10"
            / "site-packages"
            / "nvidia"
        )
        if nvidia_lib_root.exists():
            for lib_dir in nvidia_lib_root.glob("*/lib"):
                lib_paths.append(str(lib_dir))

        lib_paths.extend(
            [
                "/usr/local/cuda/targets/aarch64-linux/lib",
                "/usr/local/cuda-12.6/targets/aarch64-linux/lib",
                "/lib/aarch64-linux-gnu",
                "/usr/lib/aarch64-linux-gnu",
            ]
        )

        current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        merged = [p for p in lib_paths if p]
        if current_ld_path:
            merged.append(current_ld_path)
        os.environ["LD_LIBRARY_PATH"] = ":".join(merged)

    def _detect_runtime_device_hint(self) -> str:
        try:
            import torch  # pylint: disable=import-outside-toplevel

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "unknown"

    def _resolve_requested_device(self) -> str:
        if self.tts_device in ("cpu", "cuda"):
            return self.tts_device
        if self.runtime_device_hint in ("cpu", "cuda"):
            return self.runtime_device_hint
        return "cpu"

    def _detect_engine_device(self) -> str:
        if self.tts_engine is None:
            return "none"
        try:
            synth = getattr(self.tts_engine, "synthesizer", None)
            tts_model = getattr(synth, "tts_model", None)
            if tts_model is None:
                return "unknown"
            params = tts_model.parameters()
            first_param = next(params, None)
            if first_param is None:
                return "unknown"
            return str(first_param.device)
        except Exception:
            return "unknown"

    def _apply_engine_device(self) -> None:
        if self.tts_engine is None:
            raise RuntimeError("TTS engine is not initialized.")
        target = self.requested_device
        if hasattr(self.tts_engine, "to"):
            self.tts_engine.to(target)
            actual = self._detect_engine_device()
            self.get_logger().info(f"TTS engine actual device: {actual}")
            if target == "cuda" and actual != "unknown" and not actual.startswith("cuda"):
                raise RuntimeError(
                    f"Requested CUDA but TTS engine reports device '{actual}'."
                )
            return
        self.get_logger().warn(
            "TTS engine .to(device) API not available; cannot explicitly enforce device."
        )

    def _load_engine(self) -> None:
        try:
            from TTS.api import TTS  # pylint: disable=import-outside-toplevel

            try:
                self.tts_engine = TTS(
                    model_name=self.model_name.strip(),
                    progress_bar=False,
                )
                self._apply_engine_device()
            except TypeError:
                # Backward compatibility for older Coqui TTS APIs.
                self.get_logger().warn(
                    "Falling back to deprecated TTS(..., gpu=...) API."
                )
                self.tts_engine = TTS(
                    model_name=self.model_name.strip(),
                    progress_bar=False,
                    gpu=(self.requested_device == "cuda"),
                )

            if self.warmup_enabled and self.warmup_text.strip():
                warmup_path = self._resolve_output_path().replace(".wav", "_warmup.wav")
                self.tts_engine.tts_to_file(
                    text=self.warmup_text,
                    file_path=warmup_path,
                    speaker=self.speaker_idx or None,
                    language=self.language_idx or None,
                    speaker_wav=self.speaker_wav or None,
                    split_sentences=False,
                )
                Path(warmup_path).unlink(missing_ok=True)
        except Exception as exc:  # pragma: no cover
            self.tts_engine = None
            self.engine_error = str(exc)
            self.get_logger().error(f"Failed to load TTS engine: {self.engine_error}")

    def _resolve_output_path(self) -> str:
        out_path = Path(self.fixed_output_path).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        return str(out_path)

    def _synthesize_to_file(self, text: str, output_path: str) -> Tuple[bool, float, str]:
        if self.tts_engine is None:
            return False, 0.0, f"TTS engine is not ready: {self.engine_error}"

        start = time.perf_counter()
        try:
            with self._synthesis_lock:
                self.tts_engine.tts_to_file(
                    text=text,
                    file_path=output_path,
                    speaker=self.speaker_idx or None,
                    language=self.language_idx or None,
                    speaker_wav=self.speaker_wav or None,
                    split_sentences=self.split_sentences,
                )
        except Exception as exc:
            return False, float(time.perf_counter() - start), f"TTS generation failed: {exc}"
        return True, float(time.perf_counter() - start), ""

    def handle_synthesize_request(self, request, response):
        response.elapsed_seconds = 0.0
        response.device_used = self.requested_device

        text = request.text.strip()
        if not text:
            response.success = False
            response.wav_path = ""
            response.message = "Request text is empty."
            return response

        with self._goal_lock:
            if self._goal_active:
                response.success = False
                response.wav_path = ""
                response.message = "Cannot synthesize while /coqui_tts/speak action is active."
                return response

        if request.out_path.strip():
            self.get_logger().warn(
                "Request out_path is ignored. Using fixed_output_path parameter instead."
            )

        output_path = self._resolve_output_path()
        Path(output_path).unlink(missing_ok=True)

        ok, elapsed, err = self._synthesize_to_file(text, output_path)
        response.elapsed_seconds = float(elapsed)
        if not ok:
            response.success = False
            response.wav_path = ""
            response.message = err
            return response

        if not os.path.isfile(output_path):
            response.success = False
            response.wav_path = ""
            response.message = "TTS finished but output file was not created."
            return response

        response.success = True
        response.wav_path = output_path
        response.message = (
            f"Speech generated successfully in {elapsed:.2f}s using {response.device_used}."
        )
        return response

    def _publish_feedback(self, goal_handle, stage: str, progress: float) -> None:
        feedback = SpeakText.Feedback()
        feedback.stage = stage
        feedback.progress = float(max(0.0, min(1.0, progress)))
        goal_handle.publish_feedback(feedback)

    def goal_callback(self, goal_request) -> GoalResponse:
        if self.tts_engine is None:
            self.get_logger().warning("Rejecting goal: TTS engine is not ready.")
            return GoalResponse.REJECT

        if not goal_request.text.strip():
            self.get_logger().warning("Rejecting goal: text is empty.")
            return GoalResponse.REJECT

        with self._goal_lock:
            if self._goal_active:
                self.get_logger().warning("Rejecting goal: already speaking.")
                return GoalResponse.REJECT
            self._goal_active = True

        return GoalResponse.ACCEPT

    def cancel_callback(self, _goal_handle) -> CancelResponse:
        self.get_logger().info("Received cancel request for /coqui_tts/speak.")
        return CancelResponse.ACCEPT

    def execute_callback(self, goal_handle):
        result = SpeakText.Result()
        result.success = False
        result.wav_path = ""
        result.message = ""
        result.synthesis_seconds = 0.0
        result.playback_seconds = 0.0
        result.device_used = self.requested_device

        synthesis_seconds = 0.0
        playback_seconds = 0.0
        try:
            text = goal_handle.request.text.strip()
            self.face_player.set_idle(text)
            output_path = self._resolve_output_path()
            Path(output_path).unlink(missing_ok=True)

            self._publish_feedback(goal_handle, "synthesizing", 0.0)
            ok, synthesis_seconds, synth_error = self._synthesize_to_file(text, output_path)
            if not ok:
                result.message = synth_error
                goal_handle.abort()
                return result

            if not os.path.isfile(output_path):
                result.message = "TTS finished but output file was not created."
                goal_handle.abort()
                return result

            if goal_handle.is_cancel_requested:
                result.wav_path = output_path
                result.synthesis_seconds = float(synthesis_seconds)
                result.message = "Goal canceled after synthesis."
                goal_handle.canceled()
                return result

            self._publish_feedback(goal_handle, "playing", 0.0)

            def playback_feedback(stage: str, progress: float) -> None:
                self._publish_feedback(goal_handle, stage, progress)

            play_ok, playback_seconds, play_message = self.face_player.play(
                wav_path=output_path,
                text=text,
                should_cancel=lambda: goal_handle.is_cancel_requested,
                feedback_cb=playback_feedback,
            )

            result.wav_path = output_path
            result.synthesis_seconds = float(synthesis_seconds)
            result.playback_seconds = float(playback_seconds)

            if goal_handle.is_cancel_requested:
                result.message = "Goal canceled during playback."
                goal_handle.canceled()
                return result

            if not play_ok:
                result.message = play_message
                goal_handle.abort()
                return result

            result.success = True
            result.message = (
                f"Synthesized in {synthesis_seconds:.2f}s and played in {playback_seconds:.2f}s "
                f"using {self.requested_device}."
            )
            goal_handle.succeed()
            return result
        except Exception as exc:
            result.wav_path = self._resolve_output_path()
            result.synthesis_seconds = float(synthesis_seconds)
            result.playback_seconds = float(playback_seconds)
            result.message = f"Speak action failed: {exc}"
            goal_handle.abort()
            return result
        finally:
            self.face_player.set_idle("")
            with self._goal_lock:
                self._goal_active = False

    def destroy_node(self):
        self.face_player.stop()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CoquiTalkingFaceActionNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

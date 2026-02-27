#!/usr/bin/env python3

import os
import sys
import tempfile
import time
from pathlib import Path

import rclpy
from rclpy.node import Node

from coqui_tts_interfaces.srv import SynthesizeSpeech

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


class CoquiTtsServiceNode(Node):
    def __init__(self) -> None:
        super().__init__("coqui_tts_service_node")

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
        self.declare_parameter("timeout_sec", 180.0)
        self.declare_parameter("extra_site_packages", DEFAULT_COQUI_SITE_PACKAGES)
        self.declare_parameter("isolate_site_packages", True)

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
        self.timeout_sec = self.get_parameter("timeout_sec").get_parameter_value().double_value
        self.extra_site_packages = self.get_parameter("extra_site_packages").get_parameter_value().string_value
        self.isolate_site_packages = (
            self.get_parameter("isolate_site_packages").get_parameter_value().bool_value
        )
        added, removed = activate_coqui_site_packages(
            self.extra_site_packages, self.isolate_site_packages
        )
        self.coqui_python = str(Path(self.tts_executable).expanduser().resolve().parent / "python")
        self._build_runtime_env()
        self.runtime_device_hint = self._detect_runtime_device_hint()
        self.requested_device = self._resolve_requested_device()
        self.tts_engine = None
        self.engine_error = ""
        self._load_engine()

        self.create_service(
            SynthesizeSpeech,
            "/coqui_tts/synthesize",
            self.handle_synthesize_request,
        )

        self.get_logger().info("Coqui TTS service ready on /coqui_tts/synthesize")
        self.get_logger().info(f"Using tts executable: {self.tts_executable}")
        self.get_logger().info(f"Using fixed output path: {self.fixed_output_path}")
        self.get_logger().info(f"Requested tts_device: {self.tts_device}")
        self.get_logger().info(f"Runtime device hint: {self.runtime_device_hint}")
        self.get_logger().info(f"Resolved device: {self.requested_device}")
        self.get_logger().info(f"extra_site_packages: {self.extra_site_packages}")
        self.get_logger().info(
            f"isolate_site_packages: {self.isolate_site_packages} "
            f"(added={added}, removed={removed})"
        )

    def _resolve_output_path(self) -> str:
        out_path = Path(self.fixed_output_path).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        return str(out_path)

    def _build_runtime_env(self) -> None:
        lib_paths = []

        # CUDA libraries installed by pip inside coqui-venv (e.g. nvidia-cusparselt-cu12).
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

        # Keep system CUDA paths available as fallback.
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
                use_gpu = self.requested_device == "cuda"
                self.tts_engine = TTS(
                    model_name=self.model_name.strip(),
                    progress_bar=False,
                    gpu=use_gpu,
                )

            if self.warmup_enabled and self.warmup_text.strip():
                fd, warmup_path = tempfile.mkstemp(prefix="coqui_tts_warmup_", suffix=".wav")
                os.close(fd)
                start = time.perf_counter()
                self.tts_engine.tts_to_file(
                    text=self.warmup_text,
                    file_path=warmup_path,
                    speaker=self.speaker_idx or None,
                    language=self.language_idx or None,
                    speaker_wav=self.speaker_wav or None,
                    split_sentences=False,
                )
                warmup_elapsed = time.perf_counter() - start
                Path(warmup_path).unlink(missing_ok=True)
                self.get_logger().info(
                    f"Warmup finished in {warmup_elapsed:.2f}s on {self.requested_device}."
                )
        except Exception as exc:  # pragma: no cover
            self.tts_engine = None
            self.engine_error = str(exc)
            self.get_logger().error(f"Failed to load TTS engine: {self.engine_error}")

    def handle_synthesize_request(self, request, response):
        response.elapsed_seconds = 0.0
        response.device_used = self.requested_device

        text = request.text.strip()
        if not text:
            response.success = False
            response.wav_path = ""
            response.message = "Request text is empty."
            return response

        if self.tts_engine is None:
            response.success = False
            response.wav_path = ""
            response.message = f"TTS engine is not ready: {self.engine_error}"
            return response

        if request.out_path.strip():
            self.get_logger().warn(
                "Request out_path is ignored. Using fixed_output_path parameter instead."
            )

        out_path = self._resolve_output_path()
        if os.path.exists(out_path):
            os.remove(out_path)

        self.get_logger().info(
            f"Synthesizing speech to {out_path} using in-memory engine ({self.requested_device})."
        )

        start_time = time.perf_counter()
        try:
            self.tts_engine.tts_to_file(
                text=text,
                file_path=out_path,
                speaker=self.speaker_idx or None,
                language=self.language_idx or None,
                speaker_wav=self.speaker_wav or None,
                split_sentences=self.split_sentences,
            )
        except Exception as exc:
            response.success = False
            response.wav_path = ""
            response.elapsed_seconds = float(time.perf_counter() - start_time)
            response.message = f"TTS generation failed: {exc}"
            return response

        elapsed_seconds = float(time.perf_counter() - start_time)
        response.elapsed_seconds = elapsed_seconds

        if not os.path.isfile(out_path):
            response.success = False
            response.wav_path = ""
            response.message = "TTS finished but output file was not created."
            return response

        response.success = True
        response.wav_path = out_path
        response.message = (
            f"Speech generated successfully in {elapsed_seconds:.2f}s "
            f"using {response.device_used}."
        )
        return response


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CoquiTtsServiceNode()
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

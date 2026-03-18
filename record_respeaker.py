#!/usr/bin/env python3
"""
Record audio from a ReSpeaker microphone array using ALSA arecord.
"""

from __future__ import annotations

import argparse
import struct
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

try:
    import usb.core  # type: ignore[import-not-found]
    import usb.util  # type: ignore[import-not-found]
except Exception as _usb_import_error:  # pragma: no cover - env dependent
    usb = None
    USB_IMPORT_ERROR = _usb_import_error
else:
    USB_IMPORT_ERROR = None


# Use PipeWire/Pulse shared capture by default to avoid "device busy"
# conflicts with direct hardware access.
DEFAULT_ALSA_DEVICE = "default"
MIC_VID = 0x2886
MIC_PID = 0x0018
LED_PIDS = (0x0007, 0x0018)
USB_TIMEOUT_MS = 100000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record a WAV file from a microphone using arecord."
    )
    parser.add_argument(
        "-o",
        "--output",
        default="respeaker_recording.wav",
        help="Output WAV filename (default: %(default)s)",
    )
    parser.add_argument(
        "-d",
        "--duration",
        type=int,
        default=5,
        help="Recording duration in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "-r",
        "--rate",
        type=int,
        default=16000,
        help="Sample rate in Hz (default: %(default)s)",
    )
    parser.add_argument(
        "-c",
        "--channels",
        type=int,
        default=1,
        help="Number of channels (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_ALSA_DEVICE,
        help=f"ALSA input device (default: {DEFAULT_ALSA_DEVICE})",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Show capture devices from arecord and exit.",
    )
    parser.add_argument(
        "--localize",
        action="store_true",
        help="Print ReSpeaker DOA angle while recording.",
    )
    parser.add_argument(
        "--led",
        action="store_true",
        help="With --localize, attempt to light LED direction on the ring.",
    )
    parser.add_argument(
        "--localize-interval",
        type=float,
        default=0.25,
        help="DOA polling interval in seconds (default: %(default)s)",
    )
    return parser


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True)


class ReSpeakerTuning:
    def __init__(self, vid: int = MIC_VID, pid: int = MIC_PID):
        if USB_IMPORT_ERROR is not None:
            raise RuntimeError(f"pyusb unavailable: {USB_IMPORT_ERROR}")
        self.dev = usb.core.find(idVendor=vid, idProduct=pid)
        if not self.dev:
            raise RuntimeError("ReSpeaker tuning USB device not found (2886:0018).")

    def _read_int(self, value_offset: int, index_id: int) -> int:
        cmd = 0x80 | value_offset | 0x40
        data = self.dev.ctrl_transfer(
            usb.util.CTRL_IN | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
            0,
            cmd,
            index_id,
            8,
            USB_TIMEOUT_MS,
        )
        return struct.unpack("ii", data.tobytes())[0]

    @property
    def direction(self) -> int:
        # From official tuning.py: DOAANGLE -> (id=21, offset=0, int)
        return self._read_int(value_offset=0, index_id=21)

    @property
    def voice_activity(self) -> int:
        # From official tuning.py: VOICEACTIVITY -> (id=19, offset=32, int)
        return self._read_int(value_offset=32, index_id=19)

    def close(self) -> None:
        usb.util.dispose_resources(self.dev)


class PixelRing:
    PIXELS_N = 12
    CUSTOM = 6
    TIMEOUT_MS = 8000
    CTRL_LED_INDEX = 0x1C
    CTRL_WAKEUP_CMD = 0x02

    def __init__(self, vid: int = MIC_VID, pids: tuple[int, ...] = LED_PIDS):
        if USB_IMPORT_ERROR is not None:
            raise RuntimeError(f"pyusb unavailable: {USB_IMPORT_ERROR}")
        self.dev = None
        self.ep_out = None
        self.mode = None

        last_error = None
        for pid in pids:
            dev = usb.core.find(idVendor=vid, idProduct=pid)
            if not dev:
                continue
            try:
                config = dev.get_active_configuration()
                interface_number = None
                interface = None
                for itf in config:
                    if itf.bInterfaceClass == 0x03:
                        interface_number = itf.bInterfaceNumber
                        interface = itf
                        break
                if interface is None:
                    continue

                try:
                    if dev.is_kernel_driver_active(interface_number):
                        dev.detach_kernel_driver(interface_number)
                except Exception:
                    pass

                ep_out = None
                for ep in interface:
                    if not (ep.bEndpointAddress & 0x80):
                        ep_out = ep
                        break
                if ep_out is None:
                    continue

                self.dev = dev
                self.ep_out = ep_out
                self.mode = "hid"
                break
            except Exception as exc:
                last_error = exc

        # Fallback for USB 4 Mic Array v2 protocol: vendor control transfer
        # on 2886:0018 (no HID endpoint needed).
        if self.dev is None:
            dev = usb.core.find(idVendor=vid, idProduct=MIC_PID)
            if dev is not None:
                self.dev = dev
                self.mode = "vendor_ctrl"

        if self.dev is None or self.ep_out is None:
            if self.mode != "vendor_ctrl":
                if last_error is None:
                    raise RuntimeError(
                        "Pixel ring interface not found. Tried USB IDs: "
                        + ", ".join(f"{MIC_VID:04x}:{pid:04x}" for pid in pids)
                    )
                raise RuntimeError(
                    "Pixel ring unavailable. Tried USB IDs: "
                    + ", ".join(f"{MIC_VID:04x}:{pid:04x}" for pid in pids)
                    + f". Last error: {last_error}"
                )

        colors = [0] * 4 * self.PIXELS_N
        colors[0] = 0x04
        colors[1] = 0x40
        colors[2] = 0x04
        colors[4 + 1] = 0x08
        colors[4 * 11 + 1] = 0x08
        self.direction_template = colors

    def _write(self, address: int, payload: list[int]) -> None:
        if self.mode == "hid":
            data = bytearray(payload)
            packet = bytearray(
                [address & 0xFF, (address >> 8) & 0xFF, len(data) & 0xFF, (len(data) >> 8) & 0xFF]
            ) + data
            self.ep_out.write(packet)
            return

        if self.mode == "vendor_ctrl":
            # Uses respeaker/pixel_ring usb_pixel_ring_v2 protocol.
            self.dev.ctrl_transfer(
                usb.util.CTRL_OUT | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
                0,
                address,
                self.CTRL_LED_INDEX,
                payload,
                self.TIMEOUT_MS,
            )
            return

        raise RuntimeError("Pixel ring not initialized.")

    def set_direction(self, angle: int) -> int:
        if angle < 0 or angle > 360:
            return -1
        pos = int((angle + 15) % 360 / 30) % self.PIXELS_N
        if self.mode == "vendor_ctrl":
            # v2 firmware supports custom 12xRGBA buffer via command 0x06.
            # Use one bright pixel + two dim neighbors to indicate angle.
            data = [0, 0, 0, 0] * self.PIXELS_N
            left = (pos - 1) % self.PIXELS_N
            right = (pos + 1) % self.PIXELS_N
            for idx, g in ((left, 0x18), (pos, 0x60), (right, 0x18)):
                base = idx * 4
                data[base + 0] = 0x00  # R
                data[base + 1] = g     # G
                data[base + 2] = 0x00  # B
                data[base + 3] = 0x00
            self._write(0x06, data)
        else:
            colors = self.direction_template[-pos * 4 :] + self.direction_template[: -pos * 4]
            self._write(0, [self.CUSTOM, 0, 0, 0])
            self._write(3, colors)
        return pos

    def off(self) -> None:
        if self.mode == "vendor_ctrl":
            self._write(1, [0, 0, 0, 0])
        else:
            self._write(0, [1, 0, 0, 0])

    def close(self) -> None:
        usb.util.dispose_resources(self.dev)


def localization_worker(
    stop_event: threading.Event,
    interval_s: float,
    use_led: bool,
) -> None:
    def _hint(exc: Exception) -> str:
        text = str(exc)
        if "pyusb unavailable" in text:
            return " (install dependency: pip install pyusb)"
        if "Errno 13" in text or "insufficient permissions" in text.lower():
            return (
                " (try: run with sudo, or add udev permissions for USB "
                "2886:0018 and 2886:0007)"
            )
        return ""

    try:
        tuning = ReSpeakerTuning()
    except Exception as exc:
        print(f"Localization unavailable: {exc}{_hint(exc)}", file=sys.stderr)
        return

    ring = None
    if use_led:
        try:
            ring = PixelRing()
            print("LED ring control enabled.")
        except Exception as exc:
            print(f"LED ring unavailable: {exc}{_hint(exc)}", file=sys.stderr)

    last_angle = None
    try:
        while not stop_event.is_set():
            try:
                vad = tuning.voice_activity
                angle = int(tuning.direction)
            except Exception as exc:
                print(f"Localization read error: {exc}{_hint(exc)}", file=sys.stderr)
                break

            if vad and angle != last_angle:
                print(f"Voice direction: {angle} deg")
                last_angle = angle
                if ring is not None:
                    try:
                        ring.set_direction(angle)
                    except Exception as exc:
                        print(f"LED update failed: {exc}", file=sys.stderr)
                        ring = None

            time.sleep(interval_s)
    finally:
        if ring is not None:
            try:
                ring.off()
                ring.close()
            except Exception:
                pass
        tuning.close()


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

    if args.duration <= 0:
        print("Error: --duration must be a whole number > 0.", file=sys.stderr)
        return 2
    if args.rate <= 0:
        print("Error: --rate must be > 0.", file=sys.stderr)
        return 2
    if args.channels <= 0:
        print("Error: --channels must be > 0.", file=sys.stderr)
        return 2
    if args.localize_interval <= 0:
        print("Error: --localize-interval must be > 0.", file=sys.stderr)
        return 2

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "arecord",
        "-D",
        args.device,
        "-f",
        "S16_LE",
        "-c",
        str(args.channels),
        "-r",
        str(args.rate),
        "-d",
        str(args.duration),
        str(output_path),
    ]

    stop_event = threading.Event()
    worker = None
    if args.localize:
        worker = threading.Thread(
            target=localization_worker,
            args=(stop_event, args.localize_interval, args.led),
            daemon=True,
        )
        worker.start()

    print(f"Recording {args.duration}s from {args.device} -> {output_path}")
    result = subprocess.run(cmd, text=True, capture_output=True)
    stop_event.set()
    if worker is not None:
        worker.join(timeout=1.0)

    if result.returncode != 0:
        print("arecord failed:", file=sys.stderr)
        if result.stderr.strip():
            print(result.stderr.strip(), file=sys.stderr)
        if "Device or resource busy" in result.stderr:
            print(
                "Hint: use shared capture (--device default or --device pulse) "
                "or stop the process using the hardware node.",
                file=sys.stderr,
            )
        return result.returncode

    print("Recording complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

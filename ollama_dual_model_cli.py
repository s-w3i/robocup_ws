#!/usr/bin/env python3
"""Terminal CLI for local Ollama with fixed dual-model routing.

- Text-only turns use qwen3 text model (default qwen3:14b)
- Image turns use qwen3-vl (no output-length/options restrictions)
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import time
from typing import Any

import requests

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
TEXT_MODEL = os.environ.get("TEXT_MODEL", "qwen3:14b")
VISION_MODEL = os.environ.get("VISION_MODEL", "qwen3-vl:8b")
KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "30m")
TEXT_TEMPERATURE = float(os.environ.get("TEXT_TEMPERATURE", "0.2"))
COMMAND_TEMPERATURE = float(os.environ.get("COMMAND_TEMPERATURE", "0.0"))
DEFAULT_TEXT_THINK = os.environ.get("DEFAULT_TEXT_THINK", "true").strip().lower() in {"1", "true", "yes", "on"}
DEFAULT_COMMAND_THINK = os.environ.get("DEFAULT_COMMAND_THINK", "true").strip().lower() in {"1", "true", "yes", "on"}

COMMAND_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": ["string", "null"]},
        "action": {"type": ["string", "null"]},
        "person": {"type": ["string", "null"]},
        "object": {"type": ["string", "null"]},
        "source_location": {"type": ["string", "null"]},
        "destination_location": {"type": ["string", "null"]},
        "constraints": {"type": ["string", "null"]},
        "need_vision": {"type": ["boolean", "null"]},
        "confidence": {"type": ["number", "null"]},
    },
    "required": [
        "intent",
        "action",
        "person",
        "object",
        "source_location",
        "destination_location",
        "constraints",
        "need_vision",
        "confidence",
    ],
}


def _ollama_ready(timeout: float = 1.0) -> bool:
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=timeout)
        return r.ok
    except requests.RequestException:
        return False


def ensure_ollama_running() -> None:
    if _ollama_ready():
        return
    print("Starting ollama serve...", flush=True)
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    deadline = time.time() + 20
    while time.time() < deadline:
        if _ollama_ready():
            return
        time.sleep(0.5)
    raise RuntimeError(f"Ollama not reachable at {OLLAMA_BASE_URL}")


def list_models() -> list[str]:
    r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
    r.raise_for_status()
    return [m.get("name", "") for m in r.json().get("models", []) if m.get("name")]


def require_models() -> None:
    have = set(list_models())
    missing = [m for m in (TEXT_MODEL, VISION_MODEL) if m not in have]
    if missing:
        print("Missing required model(s):", ", ".join(missing), file=sys.stderr)
        print("Pull them with:", file=sys.stderr)
        for m in missing:
            print(f"  ollama pull {m}", file=sys.stderr)
        raise SystemExit(1)


def _b64_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _parse_json_relaxed(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 2:
            cleaned = "\n".join(lines[1:-1]).strip()
    return json.loads(cleaned)


def _to_bool(value: str) -> bool:
    v = value.strip().lower()
    if v in {"on", "true", "1", "yes"}:
        return True
    if v in {"off", "false", "0", "no"}:
        return False
    raise ValueError(f"Invalid think flag: {value}. Use on/off.")


def _parse_think_and_text(body: str, default_think: bool) -> tuple[bool, str]:
    if "|" not in body:
        return default_think, body.strip()
    head, tail = body.split("|", 1)
    head = head.strip()
    tail = tail.strip()
    if not tail:
        raise ValueError("Missing prompt text after '|'.")
    try:
        think = _to_bool(head)
        return think, tail
    except ValueError:
        return default_think, body.strip()


def chat_text(user_text: str, history: list[dict[str, Any]], think: bool) -> tuple[str, list[dict[str, Any]], float]:
    history.append({"role": "user", "content": user_text})
    payload = {
        "model": TEXT_MODEL,
        "messages": history,
        "stream": False,
        "think": think,
        "keep_alive": KEEP_ALIVE,
        "options": {
            "temperature": TEXT_TEMPERATURE,
            "num_ctx": 2048,
            "num_batch": 128,
        },
    }
    t0 = time.time()
    r = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=180)
    elapsed_s = time.time() - t0
    r.raise_for_status()
    answer = (r.json().get("message", {}).get("content") or "").strip()
    if not answer:
        answer = "(empty response)"
    history.append({"role": "assistant", "content": answer})
    return answer, history, elapsed_s


def parse_command_text(user_text: str, think: bool) -> tuple[dict[str, Any], float]:
    payload = {
        "model": TEXT_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are the command parser for a home service robot. "
                    "Return only JSON matching the schema."
                ),
            },
            {"role": "user", "content": user_text},
        ],
        "stream": False,
        "think": think,
        "keep_alive": KEEP_ALIVE,
        "format": COMMAND_SCHEMA,
        "options": {
            "temperature": COMMAND_TEMPERATURE,
            "num_ctx": 2048,
            "num_batch": 128,
        },
    }
    t0 = time.time()
    r = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=180)
    elapsed_s = time.time() - t0
    r.raise_for_status()
    content = (r.json().get("message", {}).get("content") or "").strip()
    return _parse_json_relaxed(content), elapsed_s


def chat_vision(prompt: str, image_path: str, history: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]], float]:
    if not os.path.isfile(image_path):
        raise FileNotFoundError(image_path)
    msg = {"role": "user", "content": prompt, "images": [_b64_image(image_path)]}
    history.append(msg)
    payload = {
        "model": VISION_MODEL,
        "messages": history,
        "stream": False,
        "keep_alive": KEEP_ALIVE,
    }
    # Intentionally no `options` to avoid restricting qwen3-vl output quality.
    t0 = time.time()
    r = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=300)
    elapsed_s = time.time() - t0
    r.raise_for_status()
    answer = (r.json().get("message", {}).get("content") or "").strip()
    if not answer:
        answer = "(empty response)"
    history.append({"role": "assistant", "content": answer})
    return answer, history, elapsed_s


def run_interactive() -> None:
    print(f"Text model   : {TEXT_MODEL}")
    print(f"Vision model : {VISION_MODEL}")
    print("Commands:")
    print("  /img <path> | <prompt>   use vision model")
    print("  /cmd [on|off] | <text>   parse command as strict JSON (per-request think)")
    print("  /text [on|off] | <text>  normal text chat (per-request think)")
    print("  /exit                     quit")
    print("Anything else is sent to text model with default think setting.")
    print(f"Default think: text={DEFAULT_TEXT_THINK}, cmd={DEFAULT_COMMAND_THINK}")

    text_history: list[dict[str, Any]] = []
    vision_history: list[dict[str, Any]] = []

    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.lower() in {"/exit", "exit", "quit"}:
            break

        try:
            if line.startswith("/img "):
                body = line[5:].strip()
                if "|" in body:
                    path, prompt = [x.strip() for x in body.split("|", 1)]
                else:
                    path = body
                    prompt = "Describe this image in detail."
                answer, vision_history, elapsed_s = chat_vision(prompt=prompt, image_path=path, history=vision_history)
                print(f"[{VISION_MODEL}] {answer}")
                print(f"[{VISION_MODEL}] response_time_s={elapsed_s:.3f}")
            elif line.startswith("/cmd "):
                body = line[5:].strip()
                think, cmd = _parse_think_and_text(body, DEFAULT_COMMAND_THINK)
                parsed, elapsed_s = parse_command_text(cmd, think=think)
                print(json.dumps(parsed, ensure_ascii=False, indent=2))
                print(f"[{TEXT_MODEL} cmd] think={think} response_time_s={elapsed_s:.3f}")
            elif line.startswith("/text "):
                body = line[6:].strip()
                think, text = _parse_think_and_text(body, DEFAULT_TEXT_THINK)
                answer, text_history, elapsed_s = chat_text(text, text_history, think=think)
                print(f"[{TEXT_MODEL}] {answer}")
                print(f"[{TEXT_MODEL}] think={think} response_time_s={elapsed_s:.3f}")
            else:
                answer, text_history, elapsed_s = chat_text(line, text_history, think=DEFAULT_TEXT_THINK)
                print(f"[{TEXT_MODEL}] {answer}")
                print(f"[{TEXT_MODEL}] think={DEFAULT_TEXT_THINK} response_time_s={elapsed_s:.3f}")
        except Exception as exc:
            print(f"Error: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dual-model terminal chat for Ollama")
    parser.add_argument("--text", help="One-shot text prompt")
    parser.add_argument("--cmd", help="One-shot strict command parse (JSON output)")
    parser.add_argument("--image", help="One-shot image path")
    parser.add_argument("--prompt", help="One-shot vision prompt (used with --image)")
    parser.add_argument("--text-think", default="off", choices=["on", "off"], help="One-shot text think flag")
    parser.add_argument("--cmd-think", default="off", choices=["on", "off"], help="One-shot cmd think flag")
    args = parser.parse_args()

    ensure_ollama_running()
    require_models()

    if args.image:
        prompt = args.prompt or "Describe this image in detail."
        answer, _, elapsed_s = chat_vision(prompt=prompt, image_path=args.image, history=[])
        print(answer)
        print(f"response_time_s={elapsed_s:.3f}")
        return

    if args.text:
        think = _to_bool(args.text_think)
        answer, _, elapsed_s = chat_text(args.text, history=[], think=think)
        print(answer)
        print(f"think={think} response_time_s={elapsed_s:.3f}")
        return

    if args.cmd:
        think = _to_bool(args.cmd_think)
        parsed, elapsed_s = parse_command_text(args.cmd, think=think)
        print(json.dumps(parsed, ensure_ascii=False, indent=2))
        print(f"think={think} response_time_s={elapsed_s:.3f}")
        return

    run_interactive()


if __name__ == "__main__":
    main()

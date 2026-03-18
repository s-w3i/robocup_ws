#!/usr/bin/env python3
"""Terminal tester for receptionist-style name/drink extraction.

Based on the RoboCup receptionist flow, but with terminal text input only.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from typing import Any

import requests

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
TEXT_MODEL = os.environ.get("TEXT_MODEL", "qwen3:14b")
KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "30m")
TEXT_TEMPERATURE = float(os.environ.get("TEXT_TEMPERATURE", "0.0"))
DEFAULT_TEXT_THINK = os.environ.get("DEFAULT_TEXT_THINK", "false").strip().lower() in {"1", "true", "yes", "on"}

SYSTEM_PROMPT = (
    "You are a home service robot assistant. Analyze the guest's input and:\n"
    "1. Extract the guest's name as stated (any name is acceptable).\n"
    "2. Extract the guest's favourite drink only if the input explicitly provides a beverage.\n"
    "3. If the mentioned item is food/object or not a beverage, set drink to null.\n"
    "4. Categorize task strictly from extracted entities: Both (name+drink), Name (name only), Drink (drink only), unknown (none).\n"
    "5. Return ONLY valid JSON with keys: task, reason, entities{name,drink}.\n"
    "Use null for missing values.\n"
    "Examples:\n"
    "- 'my name is Jason and I like tea' => name='Jason', drink='tea', task='Both'\n"
    "- 'Jason likes potato' => name='Jason', drink=null, task='Name'\n"
    "- 'I want orange juice' => name=null, drink='orange juice', task='Drink'"
)

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "task": {"type": ["string", "null"]},
        "reason": {"type": ["string", "null"]},
        "entities": {
            "type": "object",
            "properties": {
                "name": {"type": ["string", "null"]},
                "drink": {"type": ["string", "null"]},
            },
            "required": ["name", "drink"],
        },
    },
    "required": ["task", "reason", "entities"],
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


def model_exists(name: str) -> bool:
    r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
    r.raise_for_status()
    models = [m.get("name", "") for m in r.json().get("models", [])]
    return name in models


def parse_json_relaxed(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0))


def _to_bool(value: str) -> bool:
    v = value.strip().lower()
    if v in {"on", "true", "1", "yes"}:
        return True
    if v in {"off", "false", "0", "no"}:
        return False
    raise ValueError(f"Invalid think flag: {value}. Use on/off.")


def parse_user_text_and_think(user_text: str) -> tuple[str, bool]:
    # Per-request override format:
    #   /think on | my name is Alice
    #   /think off | my name is Alice
    if not user_text.lower().startswith("/think "):
        return user_text, DEFAULT_TEXT_THINK

    body = user_text[7:].strip()
    if "|" not in body:
        raise ValueError("Use '/think on|off | your text'.")
    mode, text = body.split("|", 1)
    text = text.strip()
    if not text:
        raise ValueError("Missing text after '|'.")
    think = _to_bool(mode.strip())
    return text, think


def classify_input(user_text: str, think: bool) -> tuple[dict[str, Any], float]:
    payload = {
        "model": TEXT_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        "stream": False,
        "think": think,
        "keep_alive": KEEP_ALIVE,
        "format": JSON_SCHEMA,
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
    content = (r.json().get("message", {}).get("content") or "").strip()
    return parse_json_relaxed(content), elapsed_s


def confirmation_sentence(name: str | None, drink: str | None) -> str:
    if not name and not drink:
        return "Could you please provide both your name and your favourite drink?"
    if name and not drink:
        return f"Hello {name}, may I have your favourite drink, please?"
    if drink and not name:
        return f"You mentioned {drink} as your favourite drink. Could you please tell me your name?"
    return f"Thank you, {name}! We have noted your favourite drink as {drink}."


def normalize(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() in {"null", "none", "unknown"}:
        return None
    return s


def infer_task(name: str | None, drink: str | None) -> str:
    if name and drink:
        return "Both"
    if name:
        return "Name"
    if drink:
        return "Drink"
    return "unknown"


def main() -> None:
    ensure_ollama_running()
    if not model_exists(TEXT_MODEL):
        print(f"Model not found: {TEXT_MODEL}", file=sys.stderr)
        print(f"Pull it first: ollama pull {TEXT_MODEL}", file=sys.stderr)
        raise SystemExit(1)

    print(f"Using model: {TEXT_MODEL}")
    print("Type guest utterances. Commands: /reset, /quit")
    print(f"Default think: {DEFAULT_TEXT_THINK}")
    print("Per request override: /think on|off | your text")

    guest_name: str | None = None
    guest_drink: str | None = None

    while True:
        try:
            user_text = input("\nguest> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_text:
            continue
        if user_text.lower() in {"/quit", "quit", "exit"}:
            break
        if user_text.lower() == "/reset":
            guest_name, guest_drink = None, None
            print("State reset.")
            continue

        try:
            parsed_text, think = parse_user_text_and_think(user_text)
            result, elapsed_s = classify_input(parsed_text, think=think)
        except Exception as exc:
            print(f"LLM error: {exc}")
            continue

        entities = result.get("entities", {})
        parsed_name = normalize(entities.get("name"))
        parsed_drink = normalize(entities.get("drink"))

        if parsed_name:
            guest_name = parsed_name
        if parsed_drink:
            guest_drink = parsed_drink

        task = normalize(result.get("task")) or infer_task(parsed_name, parsed_drink)
        reason = normalize(result.get("reason")) or ""
        confirm = confirmation_sentence(guest_name, guest_drink)

        print(f"task   : {task}")
        print(f"name   : {guest_name}")
        print(f"drink  : {guest_drink}")
        if reason:
            print(f"reason : {reason}")
        print(f"robot  : {confirm}")
        print(f"think  : {think}")
        print(f"time_s : {elapsed_s:.3f}")

        if guest_name and guest_drink:
            print("\nFinal guest info:")
            print(json.dumps({"name": guest_name, "drink": guest_drink}, ensure_ascii=False))
            print("Conversation complete. Use /reset for next guest or /quit.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Local Ollama vision chat UI.

Usage:
  python3 ollama_vision_ui.py

Then open the printed local URL in your browser.
"""

from __future__ import annotations

import atexit
import base64
import io
import importlib.util
import os
import subprocess
import sys
import time
from typing import Any

import requests

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3-vl:8b")
DEFAULT_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "2048"))
DEFAULT_NUM_PREDICT = int(os.environ.get("OLLAMA_NUM_PREDICT", "192"))
DEFAULT_NUM_BATCH = int(os.environ.get("OLLAMA_NUM_BATCH", "128"))
DEFAULT_TEMPERATURE = float(os.environ.get("OLLAMA_TEMPERATURE", "0.2"))
DEFAULT_TOP_P = float(os.environ.get("OLLAMA_TOP_P", "0.9"))
DEFAULT_TOP_K = int(os.environ.get("OLLAMA_TOP_K", "20"))
DEFAULT_IMAGE_MAX_EDGE = int(os.environ.get("OLLAMA_IMAGE_MAX_EDGE", "640"))
DEFAULT_KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "30m")
RETRY_NUM_PREDICT = int(os.environ.get("OLLAMA_RETRY_NUM_PREDICT", str(max(DEFAULT_NUM_PREDICT * 2, 320))))
ROBOT_SYSTEM_PROMPT = os.environ.get(
    "OLLAMA_SYSTEM_PROMPT",
    (
        "You are a home service robot assistant. "
        "Respond immediately with the final answer only. "
        "Keep replies short and actionable unless user asks for detail."
    ),
)
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
_started_proc: subprocess.Popen[Any] | None = None


def _ensure_gradio() -> None:
    if importlib.util.find_spec("gradio") is not None:
        return

    print("gradio not found. Installing it now...", flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gradio>=4.0.0"])


def _ollama_ready(timeout: float = 1.0) -> bool:
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=timeout)
        return r.ok
    except requests.RequestException:
        return False


def ensure_ollama_running() -> None:
    global _started_proc

    if _ollama_ready():
        return

    if _started_proc is None:
        print("Starting local Ollama server...", flush=True)
        _started_proc = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )

    deadline = time.time() + 20
    while time.time() < deadline:
        if _ollama_ready(timeout=1.0):
            return
        time.sleep(0.5)

    raise RuntimeError(
        "Ollama server is not reachable at "
        f"{OLLAMA_BASE_URL}. Start it manually with: ollama serve"
    )


def _cleanup() -> None:
    if _started_proc is not None and _started_proc.poll() is None:
        _started_proc.terminate()


atexit.register(_cleanup)


def list_models() -> list[str]:
    ensure_ollama_running()
    r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
    r.raise_for_status()
    models = [m.get("name", "") for m in r.json().get("models", []) if m.get("name")]
    return models


def _encode_image(path: str) -> str:
    # Downscale image before sending to reduce prompt-processing latency.
    if DEFAULT_IMAGE_MAX_EDGE > 0:
        try:
            from PIL import Image

            with Image.open(path) as img:
                img = img.convert("RGB")
                img.thumbnail((DEFAULT_IMAGE_MAX_EDGE, DEFAULT_IMAGE_MAX_EDGE))
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=85, optimize=True)
                return base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception:
            pass

    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def warm_model(model: str) -> str:
    if not model:
        return "No model selected."

    payload = {
        "model": model,
        "prompt": " ",
        "stream": False,
        "keep_alive": DEFAULT_KEEP_ALIVE,
        "options": {
            "num_ctx": DEFAULT_NUM_CTX,
            "num_predict": 1,
            "num_batch": DEFAULT_NUM_BATCH,
        },
    }
    try:
        r = requests.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=120)
        r.raise_for_status()
        return f"Model warmed and kept alive: `{model}` (keep_alive={DEFAULT_KEEP_ALIVE})"
    except Exception as exc:
        return f"Model warmup failed for `{model}`: {exc}"


def _chat_request(
    model: str,
    messages: list[dict[str, Any]],
    num_predict: int,
    temperature: float | None = None,
    timeout_s: int = 120,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "keep_alive": DEFAULT_KEEP_ALIVE,
        "options": {
            "num_ctx": DEFAULT_NUM_CTX,
            "num_predict": num_predict,
            "num_batch": DEFAULT_NUM_BATCH,
            "temperature": DEFAULT_TEMPERATURE if temperature is None else temperature,
            "top_p": DEFAULT_TOP_P,
            "top_k": DEFAULT_TOP_K,
        },
    }
    r = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=timeout_s)
    r.raise_for_status()
    return r.json()


def chat_with_ollama(
    message: str,
    image_path: str | None,
    model: str,
    ui_history: list[dict[str, str]],
    ollama_history: list[dict[str, Any]],
) -> tuple[str, None, list[dict[str, str]], list[dict[str, str]], list[dict[str, Any]]]:
    ui_history = ui_history or []
    ollama_history = ollama_history or []
    if not ollama_history or ollama_history[0].get("role") != "system":
        ollama_history.insert(0, {"role": "system", "content": ROBOT_SYSTEM_PROMPT})

    text = (message or "").strip()
    if not text and not image_path:
        return "", None, ui_history, ui_history, ollama_history

    if not text and image_path:
        text = "Describe this image."

    user_message: dict[str, Any] = {"role": "user", "content": text}
    ui_user_text = text

    if image_path:
        user_message["images"] = [_encode_image(image_path)]
        ui_user_text = f"{text}\n\n[Attached image: {os.path.basename(image_path)}]"

    ollama_history.append(user_message)

    try:
        first = _chat_request(model=model, messages=ollama_history, num_predict=DEFAULT_NUM_PREDICT)
        first_msg = first.get("message", {})
        answer = (first_msg.get("content") or "").strip()

        # Some qwen3-vl runs end with thinking-only output and empty final content.
        # Retry once with larger token budget + direct final-answer instruction.
        if not answer:
            retry_messages = list(ollama_history) + [
                {"role": "user", "content": "Return the final answer now in one short sentence only."}
            ]
            second = _chat_request(
                model=model,
                messages=retry_messages,
                num_predict=RETRY_NUM_PREDICT,
                temperature=0.1,
                timeout_s=180,
            )
            second_msg = second.get("message", {})
            answer = (second_msg.get("content") or "").strip()

        if not answer:
            answer = (
                "I could not get a final answer from the model this time. "
                "Please retry, or raise OLLAMA_NUM_PREDICT/OLLAMA_RETRY_NUM_PREDICT."
            )
    except Exception as exc:
        answer = f"Error calling Ollama: {exc}"

    ui_history.append({"role": "user", "content": ui_user_text})
    ui_history.append({"role": "assistant", "content": answer})
    ollama_history.append({"role": "assistant", "content": answer})

    return "", None, ui_history, ui_history, ollama_history


def refresh_models() -> tuple[Any, str]:
    import gradio as gr

    models = list_models()
    if not models:
        return gr.Dropdown(choices=[], value=None), "No local models found. Pull one first: ollama pull qwen3-vl:8b"

    selected = DEFAULT_MODEL if DEFAULT_MODEL in models else models[0]
    warm_status = warm_model(selected)
    return gr.Dropdown(choices=models, value=selected), f"Loaded {len(models)} model(s). {warm_status}"


def build_ui() -> Any:
    import gradio as gr

    models = list_models()
    if not models:
        raise RuntimeError("No local models found. Example: ollama pull qwen3-vl:8b")

    selected = DEFAULT_MODEL if DEFAULT_MODEL in models else models[0]

    initial_status = warm_model(selected)

    with gr.Blocks(title="Ollama Vision Chat") as demo:
        gr.Markdown("# Ollama Vision Chat\nChat with local Ollama and optionally attach an image.")

        status = gr.Markdown(initial_status)
        with gr.Row():
            model = gr.Dropdown(choices=models, value=selected, label="Model")
            refresh = gr.Button("Refresh Models")

        chatbot = gr.Chatbot(height=500, label="Conversation")

        with gr.Row():
            msg = gr.Textbox(label="Message", placeholder="Ask something about text and/or image", scale=3)
            image = gr.Image(type="filepath", label="Attach image", scale=2)

        with gr.Row():
            send = gr.Button("Send", variant="primary")
            clear = gr.Button("Clear")

        ui_state = gr.State([])
        ollama_state = gr.State([])

        send.click(
            chat_with_ollama,
            inputs=[msg, image, model, ui_state, ollama_state],
            outputs=[msg, image, chatbot, ui_state, ollama_state],
        )

        msg.submit(
            chat_with_ollama,
            inputs=[msg, image, model, ui_state, ollama_state],
            outputs=[msg, image, chatbot, ui_state, ollama_state],
        )

        clear.click(lambda: ([], [], []), outputs=[chatbot, ui_state, ollama_state])

        refresh.click(refresh_models, outputs=[model, status])
        model.change(warm_model, inputs=[model], outputs=[status])

    return demo


def main() -> None:
    _ensure_gradio()
    ensure_ollama_running()
    demo = build_ui()
    preferred_port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    try:
        demo.launch(server_name="127.0.0.1", server_port=preferred_port, inbrowser=True)
    except OSError as exc:
        if "Cannot find empty port" not in str(exc):
            raise
        demo.launch(server_name="127.0.0.1", server_port=preferred_port + 1, inbrowser=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""One-shot YOLOE service that returns the pointed object using VLM selection."""

from __future__ import annotations

import math
import base64
import json
import os
import pathlib
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import cv2
import numpy as np
import rclpy
import requests
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformBroadcaster, TransformException, TransformListener

from yoloe_detection_interfaces.srv import DetectObjectPrompt
from yoloe_detection_service.yoloe_detection_service_node import (
    ensure_torch_runtime_libs,
    preload_cupti_if_needed,
)


@dataclass
class PointedDetection:
    class_name: str
    confidence: float
    box_xyxy: tuple[int, int, int, int]
    center_uv: tuple[int, int]
    score: float
    object_depth_m: float | None


@dataclass
class FrameSnapshot:
    color_image: np.ndarray
    depth_image: np.ndarray
    depth_encoding: str
    depth_frame: str
    camera_info: CameraInfo
    color_header: Any
    stamp_ns: int


@dataclass
class FrameDecision:
    success: bool
    message: str
    selected: PointedDetection | None
    point_in_camera: np.ndarray | None
    detections_in_frame: int
    inference_ms: float
    annotated: np.ndarray
    color_header: Any


@dataclass
class CandidateVote:
    count: int
    score_sum: float
    best_score: float
    best_decision: FrameDecision

    @property
    def avg_score(self) -> float:
        return self.score_sum / max(1, self.count)


@dataclass
class DetectionRunResult:
    success: bool
    message: str
    selected: PointedDetection | None
    pose_camera_link: PoseStamped | None
    tf_child_frame: str
    detections_in_frame: int
    tf_published_count: int
    inference_ms: float
    saved_image_path: str


@dataclass
class TimedTransform:
    translation: np.ndarray
    expires_at_monotonic: float


def inject_known_site_packages() -> None:
    """Add known site-packages paths for mixed-venv deployments."""
    candidates = [
        pathlib.Path("/home/usern/coqui-venv/lib/python3.10/site-packages"),
        pathlib.Path("/home/usern/follow-venv/lib/python3.10/site-packages"),
    ]
    for candidate in candidates:
        if candidate.is_dir():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.append(candidate_str)


class YoloeVlmPointedDetectionServiceNode(Node):
    def __init__(self) -> None:
        super().__init__("yoloe_vlm_pointed_detection_service_node")

        self.declare_parameter("service_name", "/yoloe/detect_pointed_prompt_vlm")
        self.declare_parameter("model_path", "/home/usern/yoloe-26l-seg.pt")
        self.declare_parameter("device", "auto")
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("conf", 0.25)
        self.declare_parameter("iou", 0.45)
        self.declare_parameter("max_det", 200)
        self.declare_parameter("force_torch_nms", True)

        self.declare_parameter("color_topic", "/camera0/color/image_raw")
        self.declare_parameter("depth_topic", "/camera0/depth/image_rect_raw")
        self.declare_parameter("camera_info_topic", "/camera0/color/camera_info")
        self.declare_parameter("camera_link_frame", "camera0_link")
        self.declare_parameter("pose_topic", "/yoloe/vlm_pointed_object_pose")
        self.declare_parameter("result_image_topic", "/yoloe/vlm_pointing_result_image")
        self.declare_parameter("object_frame_prefix", "pointed_vlm")
        self.declare_parameter("tf_ttl_sec", 60.0)
        self.declare_parameter("tf_republish_hz", 10.0)

        self.declare_parameter("depth_window_size", 7)
        self.declare_parameter("min_depth_m", 0.05)
        self.declare_parameter("max_depth_m", 10.0)

        self.declare_parameter("vote_frames", 5)
        self.declare_parameter("vote_interval_ms", 90)
        self.declare_parameter("vote_min_ratio", 0.45)
        self.declare_parameter("vote_cell_px", 90)
        self.declare_parameter("vote_ambiguity_gap", 2.0)

        self.declare_parameter("publish_result_image", True)
        self.declare_parameter("show_ui", True)
        self.declare_parameter("ui_wait_ms", 1200)
        self.declare_parameter("save_dir", "/home/usern/robocup_ws/yoloe_out")
        self.declare_parameter("always_save_image", False)
        self.declare_parameter("publish_vlm_debug_image", True)
        self.declare_parameter("vlm_debug_image_topic", "/yoloe/vlm_pointing_query_image")

        self.declare_parameter("ollama_base_url", os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"))
        self.declare_parameter("vlm_model", os.environ.get("VISION_MODEL", "qwen3-vl:8b"))
        self.declare_parameter("vlm_timeout_s", 60.0)
        self.declare_parameter("vlm_num_ctx", 2048)
        self.declare_parameter("vlm_num_predict", 128)
        self.declare_parameter("vlm_retry_num_predict", 384)
        self.declare_parameter("vlm_max_retries", 1)
        self.declare_parameter("vlm_use_thinking_fallback", True)
        self.declare_parameter("vlm_max_candidates", 12)
        self.declare_parameter("vlm_keep_alive", os.environ.get("OLLAMA_KEEP_ALIVE", "30m"))
        self.declare_parameter("vlm_image_max_edge", 960)
        self.declare_parameter("ensure_ollama_running", True)

        self.service_name = str(self.get_parameter("service_name").value)
        self.model_path = str(self.get_parameter("model_path").value)
        self.device_request = str(self.get_parameter("device").value)
        self.imgsz = int(self.get_parameter("imgsz").value)
        self.conf = float(self.get_parameter("conf").value)
        self.iou = float(self.get_parameter("iou").value)
        self.max_det = int(self.get_parameter("max_det").value)
        self.force_torch_nms = bool(self.get_parameter("force_torch_nms").value)

        self.color_topic = str(self.get_parameter("color_topic").value)
        self.depth_topic = str(self.get_parameter("depth_topic").value)
        self.camera_info_topic = str(self.get_parameter("camera_info_topic").value)
        self.camera_link_frame = str(self.get_parameter("camera_link_frame").value)
        self.pose_topic = str(self.get_parameter("pose_topic").value)
        self.result_image_topic = str(self.get_parameter("result_image_topic").value)
        self.object_frame_prefix = str(self.get_parameter("object_frame_prefix").value)
        self.tf_ttl_sec = max(0.0, float(self.get_parameter("tf_ttl_sec").value))
        self.tf_republish_hz = max(0.5, float(self.get_parameter("tf_republish_hz").value))

        self.depth_window_size = max(1, int(self.get_parameter("depth_window_size").value))
        self.min_depth_m = float(self.get_parameter("min_depth_m").value)
        self.max_depth_m = float(self.get_parameter("max_depth_m").value)

        self.vote_frames = max(1, int(self.get_parameter("vote_frames").value))
        self.vote_interval_ms = max(0, int(self.get_parameter("vote_interval_ms").value))
        self.vote_min_ratio = float(self.get_parameter("vote_min_ratio").value)
        self.vote_cell_px = max(10, int(self.get_parameter("vote_cell_px").value))
        self.vote_ambiguity_gap = float(self.get_parameter("vote_ambiguity_gap").value)

        self.publish_result_image = bool(self.get_parameter("publish_result_image").value)
        self.show_ui = bool(self.get_parameter("show_ui").value)
        self.ui_wait_ms = max(1, int(self.get_parameter("ui_wait_ms").value))
        self.save_dir = pathlib.Path(str(self.get_parameter("save_dir").value)).expanduser().resolve()
        self.always_save_image = bool(self.get_parameter("always_save_image").value)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.publish_vlm_debug_image = bool(self.get_parameter("publish_vlm_debug_image").value)
        self.vlm_debug_image_topic = str(self.get_parameter("vlm_debug_image_topic").value)

        self.ollama_base_url = str(self.get_parameter("ollama_base_url").value).rstrip("/")
        self.vlm_model = str(self.get_parameter("vlm_model").value)
        self.vlm_timeout_s = max(5.0, float(self.get_parameter("vlm_timeout_s").value))
        self.vlm_num_ctx = max(512, int(self.get_parameter("vlm_num_ctx").value))
        self.vlm_num_predict = max(32, int(self.get_parameter("vlm_num_predict").value))
        self.vlm_retry_num_predict = max(64, int(self.get_parameter("vlm_retry_num_predict").value))
        self.vlm_max_retries = max(0, int(self.get_parameter("vlm_max_retries").value))
        self.vlm_use_thinking_fallback = bool(self.get_parameter("vlm_use_thinking_fallback").value)
        self.vlm_max_candidates = max(1, int(self.get_parameter("vlm_max_candidates").value))
        self.vlm_keep_alive = str(self.get_parameter("vlm_keep_alive").value)
        self.vlm_image_max_edge = max(256, int(self.get_parameter("vlm_image_max_edge").value))
        self.ensure_ollama_running = bool(self.get_parameter("ensure_ollama_running").value)

        self._bridge = CvBridge()
        self._lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._ui_failed = False

        self._latest_color_image: np.ndarray | None = None
        self._latest_depth_image: np.ndarray | None = None
        self._latest_depth_frame: str = ""
        self._latest_depth_encoding: str = ""
        self._latest_camera_info: CameraInfo | None = None
        self._latest_color_header = None
        self._latest_color_stamp_ns: int = -1

        self._model: Any = None
        self._prompt_key: tuple[str, ...] | None = None
        self._device: str = "cpu"
        self._active_prompts: list[str] = []
        self._last_vlm_reason: str = ""
        self._active_timed_tfs: dict[str, TimedTransform] = {}
        self._tf_publish_lock = threading.Lock()

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._tf_broadcaster = TransformBroadcaster(self)

        self._pose_pub = self.create_publisher(PoseStamped, self.pose_topic, 10)
        self._result_image_pub = None
        if self.publish_result_image:
            self._result_image_pub = self.create_publisher(
                Image,
                self.result_image_topic,
                qos_profile_sensor_data,
            )
        self._vlm_debug_image_pub = None
        if self.publish_vlm_debug_image:
            self._vlm_debug_image_pub = self.create_publisher(
                Image,
                self.vlm_debug_image_topic,
                qos_profile_sensor_data,
            )

        self.create_subscription(Image, self.color_topic, self._on_color_image, qos_profile_sensor_data)
        self.create_subscription(Image, self.depth_topic, self._on_depth_image, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, self.camera_info_topic, self._on_camera_info, qos_profile_sensor_data)
        self.create_service(DetectObjectPrompt, self.service_name, self._handle_detect_request)
        self._tf_publish_timer = self.create_timer(1.0 / self.tf_republish_hz, self._on_tf_publish_timer)

        self._load_model()
        if self.ensure_ollama_running:
            self._ensure_ollama_running()

        self.get_logger().info(f"VLM pointed-object service ready on {self.service_name}")
        self.get_logger().info(f"Model path: {self.model_path}")
        self.get_logger().info(f"Device: {self._device}")
        self.get_logger().info(f"Color topic: {self.color_topic}")
        self.get_logger().info(f"Depth topic: {self.depth_topic}")
        self.get_logger().info(f"Camera info topic: {self.camera_info_topic}")
        self.get_logger().info(
            f"TF publish policy: dynamic /tf for {self.tf_ttl_sec:.1f}s at {self.tf_republish_hz:.1f} Hz"
        )
        self.get_logger().info(
            f"VLM endpoint: {self.ollama_base_url}, model: {self.vlm_model}, timeout={self.vlm_timeout_s:.1f}s"
        )
        self.get_logger().info(
            f"VLM parse policy: num_predict={self.vlm_num_predict}, retry_num_predict={self.vlm_retry_num_predict}, "
            f"max_retries={self.vlm_max_retries}, thinking_fallback={self.vlm_use_thinking_fallback}, "
            f"max_candidates={self.vlm_max_candidates}"
        )
        self.get_logger().info(
            f"VLM debug image topic enabled={self.publish_vlm_debug_image}: {self.vlm_debug_image_topic}"
        )
        self.get_logger().info(
            f"Voting: frames={self.vote_frames}, interval_ms={self.vote_interval_ms}, "
            f"min_ratio={self.vote_min_ratio:.2f}"
        )

    def _load_model(self) -> None:
        inject_known_site_packages()
        ensure_torch_runtime_libs()
        preload_cupti_if_needed()

        import torch
        from ultralytics import YOLOE

        self._device = self._choose_device(self.device_request, torch)

        started = time.perf_counter()
        self._model = YOLOE(self.model_path)
        elapsed_s = time.perf_counter() - started
        self.get_logger().info(f"Loaded YOLOE model in {elapsed_s:.2f}s")
        self.get_logger().info(f"Torch CUDA available: {torch.cuda.is_available()}")

    @staticmethod
    def _choose_device(requested: str, torch_module: Any) -> str:
        if requested in ("cpu", "cuda"):
            return requested
        return "cuda" if torch_module.cuda.is_available() else "cpu"

    def _ollama_ready(self, timeout: float = 1.0) -> bool:
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=timeout)
            return response.ok
        except requests.RequestException:
            return False

    def _ensure_ollama_running(self) -> None:
        if self._ollama_ready():
            return

        self.get_logger().info("Starting local ollama serve...")
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        deadline = time.time() + 20.0
        while time.time() < deadline:
            if self._ollama_ready():
                return
            time.sleep(0.5)
        raise RuntimeError(f"Ollama is not reachable at {self.ollama_base_url}")

    def _on_color_image(self, msg: Image) -> None:
        try:
            image = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            self.get_logger().warn(f"Failed to convert color image: {exc}")
            return

        stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        with self._lock:
            self._latest_color_image = image
            self._latest_color_header = msg.header
            self._latest_color_stamp_ns = stamp_ns

    def _on_depth_image(self, msg: Image) -> None:
        try:
            depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except CvBridgeError as exc:
            self.get_logger().warn(f"Failed to convert depth image: {exc}")
            return

        with self._lock:
            self._latest_depth_image = depth
            self._latest_depth_frame = msg.header.frame_id
            self._latest_depth_encoding = msg.encoding

    def _on_camera_info(self, msg: CameraInfo) -> None:
        with self._lock:
            self._latest_camera_info = msg

    def _handle_detect_request(
        self, request: DetectObjectPrompt.Request, response: DetectObjectPrompt.Response
    ) -> DetectObjectPrompt.Response:
        response.success = False
        response.message = ""
        response.detected_classes = []
        response.confidences = []
        response.poses_camera_link = []
        response.tf_child_frames = []
        response.saved_image_path = ""
        response.detections_in_frame = 0
        response.tf_published_count = 0
        response.inference_ms = 0.0

        try:
            prompts = self._parse_prompts(request.prompt_text)
        except ValueError as exc:
            response.message = str(exc)
            return response

        run_result = self._run_detection(prompts, request.save_image)
        response.detections_in_frame = run_result.detections_in_frame
        response.tf_published_count = run_result.tf_published_count
        response.inference_ms = run_result.inference_ms
        response.saved_image_path = run_result.saved_image_path

        if not run_result.success or run_result.selected is None or run_result.pose_camera_link is None:
            response.message = run_result.message
            return response

        response.success = True
        response.message = run_result.message
        response.detected_classes = [run_result.selected.class_name]
        response.confidences = [run_result.selected.confidence]
        response.poses_camera_link = [run_result.pose_camera_link]
        response.tf_child_frames = [run_result.tf_child_frame] if run_result.tf_child_frame else []
        return response

    def _run_detection(self, prompts: list[str], save_image_request: bool) -> DetectionRunResult:
        prompt_key = tuple(prompts)
        self._active_prompts = list(prompts)

        with self._inference_lock:
            if prompt_key != self._prompt_key:
                started = time.perf_counter()
                self._model.set_classes(prompts)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                self._prompt_key = prompt_key
                self.get_logger().info(
                    f"Updated prompt classes ({', '.join(prompts)}) in {elapsed_ms:.1f} ms"
                )

        frame_decisions: list[FrameDecision] = []
        last_stamp_ns = -1

        for i in range(self.vote_frames):
            snapshot = self._wait_for_new_snapshot(last_stamp_ns)
            if snapshot is None:
                if not frame_decisions:
                    return DetectionRunResult(
                        False,
                        (
                            f"No synchronized color/depth/camera_info available. "
                            f"Check topics: {self.color_topic}, {self.depth_topic}, {self.camera_info_topic}"
                        ),
                        None,
                        None,
                        "",
                        0,
                        0,
                        0.0,
                        "",
                    )
                break

            last_stamp_ns = snapshot.stamp_ns
            decision = self._process_frame(snapshot)
            frame_decisions.append(decision)

            if i + 1 < self.vote_frames and self.vote_interval_ms > 0:
                time.sleep(self.vote_interval_ms / 1000.0)

        if not frame_decisions:
            return DetectionRunResult(
                False,
                "No frames processed for this request.",
                None,
                None,
                "",
                0,
                0,
                0.0,
                "",
            )

        success_decisions = [
            d for d in frame_decisions if d.success and d.selected is not None and d.point_in_camera is not None
        ]

        avg_inference_ms = float(
            sum(d.inference_ms for d in frame_decisions) / max(1, len(frame_decisions))
        )
        max_detections = int(max(d.detections_in_frame for d in frame_decisions))

        if not success_decisions:
            last = frame_decisions[-1]
            saved_path = self._publish_ui_outputs(
                annotated=last.annotated,
                color_header=last.color_header,
                prompts=prompts,
                label="no_valid_pointed_object",
                save_image_request=save_image_request,
            )
            return DetectionRunResult(
                False,
                last.message,
                None,
                None,
                "",
                max_detections,
                0,
                avg_inference_ms,
                saved_path,
            )

        votes: dict[tuple[str, int, int], CandidateVote] = {}
        for decision in success_decisions:
            selected = decision.selected
            assert selected is not None
            key = (
                selected.class_name,
                int(selected.center_uv[0] // self.vote_cell_px),
                int(selected.center_uv[1] // self.vote_cell_px),
            )
            vote = votes.get(key)
            if vote is None:
                votes[key] = CandidateVote(
                    count=1,
                    score_sum=float(selected.score),
                    best_score=float(selected.score),
                    best_decision=decision,
                )
                continue

            vote.count += 1
            vote.score_sum += float(selected.score)
            if float(selected.score) < vote.best_score:
                vote.best_score = float(selected.score)
                vote.best_decision = decision

        sorted_votes = sorted(votes.items(), key=lambda kv: (-kv[1].count, kv[1].avg_score))
        winner_key, winner_vote = sorted_votes[0]

        required_votes = max(1, int(math.ceil(self.vote_min_ratio * len(frame_decisions))))
        if winner_vote.count < required_votes:
            fail_decision = winner_vote.best_decision
            saved_path = self._publish_ui_outputs(
                annotated=fail_decision.annotated,
                color_header=fail_decision.color_header,
                prompts=prompts,
                label="low_vote_confidence",
                save_image_request=save_image_request,
            )
            return DetectionRunResult(
                False,
                (
                    f"Low vote confidence for pointed object: winner votes "
                    f"{winner_vote.count}/{len(frame_decisions)} < required {required_votes}."
                ),
                None,
                None,
                "",
                max_detections,
                0,
                avg_inference_ms,
                saved_path,
            )

        if len(sorted_votes) > 1:
            second_vote = sorted_votes[1][1]
            if (
                winner_vote.count == second_vote.count
                and abs(winner_vote.avg_score - second_vote.avg_score) <= self.vote_ambiguity_gap
            ):
                fail_decision = winner_vote.best_decision
                saved_path = self._publish_ui_outputs(
                    annotated=fail_decision.annotated,
                    color_header=fail_decision.color_header,
                    prompts=prompts,
                    label="ambiguous_vote",
                    save_image_request=save_image_request,
                )
                return DetectionRunResult(
                    False,
                    (
                        "Ambiguous pointing result: top candidates have similar vote count/score. "
                        "Move closer or point longer."
                    ),
                    None,
                    None,
                    "",
                    max_detections,
                    0,
                    avg_inference_ms,
                    saved_path,
                )

        final_decision = winner_vote.best_decision
        final_selected = final_decision.selected
        final_point = final_decision.point_in_camera
        assert final_selected is not None
        assert final_point is not None

        pose_msg = PoseStamped()
        pose_msg.header.frame_id = self.camera_link_frame
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.pose.position.x = float(final_point[0])
        pose_msg.pose.position.y = float(final_point[1])
        pose_msg.pose.position.z = float(final_point[2])
        pose_msg.pose.orientation.w = 1.0
        self._safe_publish(self._pose_pub, pose_msg)

        child_frame = f"{self.object_frame_prefix}_{self._slug(final_selected.class_name)}"
        self._publish_tf(child_frame, final_point)

        annotated = final_decision.annotated.copy()
        cv2.putText(
            annotated,
            f"Votes: {winner_vote.count}/{len(frame_decisions)}",
            (12, 86),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"VLM: {self._last_vlm_reason[:70]}",
            (12, 112),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (200, 255, 200),
            2,
            cv2.LINE_AA,
        )
        saved_path = self._publish_ui_outputs(
            annotated=annotated,
            color_header=final_decision.color_header,
            prompts=prompts,
            label=final_selected.class_name,
            save_image_request=save_image_request,
        )

        message = (
            f"Selected '{final_selected.class_name}' via VLM decision. "
            f"Votes {winner_vote.count}/{len(frame_decisions)}. "
            f"Centroid [{final_point[0]:.3f}, {final_point[1]:.3f}, {final_point[2]:.3f}] m "
            f"in {self.camera_link_frame}. VLM={self._last_vlm_reason}"
        )
        return DetectionRunResult(
            True,
            message,
            final_selected,
            pose_msg,
            child_frame,
            max_detections,
            1,
            avg_inference_ms,
            saved_path,
        )

    def _wait_for_new_snapshot(self, last_stamp_ns: int) -> FrameSnapshot | None:
        timeout_s = max(0.4, (self.vote_interval_ms / 1000.0) * 2.5)
        deadline = time.monotonic() + timeout_s

        while time.monotonic() < deadline:
            with self._lock:
                color = None if self._latest_color_image is None else self._latest_color_image.copy()
                depth = None if self._latest_depth_image is None else self._latest_depth_image.copy()
                depth_frame = self._latest_depth_frame
                depth_encoding = self._latest_depth_encoding
                cam_info = self._latest_camera_info
                header = self._latest_color_header
                stamp_ns = self._latest_color_stamp_ns

            if (
                color is not None
                and depth is not None
                and cam_info is not None
                and stamp_ns != last_stamp_ns
                and stamp_ns >= 0
            ):
                return FrameSnapshot(
                    color_image=color,
                    depth_image=depth,
                    depth_encoding=depth_encoding,
                    depth_frame=depth_frame,
                    camera_info=cam_info,
                    color_header=header,
                    stamp_ns=stamp_ns,
                )

            time.sleep(0.01)

        return None

    def _process_frame(self, snapshot: FrameSnapshot) -> FrameDecision:
        color_image = snapshot.color_image
        depth_image = snapshot.depth_image
        if depth_image.ndim > 2:
            depth_image = depth_image[:, :, 0]

        image_h, image_w = color_image.shape[:2]

        with self._inference_lock:
            if self.force_torch_nms:
                sys.modules.pop("torchvision", None)
            infer_started = time.perf_counter()
            results = self._model.predict(
                source=color_image,
                device=self._device,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                max_det=self.max_det,
                verbose=False,
                save=False,
            )
            inference_ms = float((time.perf_counter() - infer_started) * 1000.0)

        if not results:
            annotated = self._draw_no_detection_ui(color_image, "YOLOE returned no results")
            return FrameDecision(
                success=False,
                message="YOLOE returned no results.",
                selected=None,
                point_in_camera=None,
                detections_in_frame=0,
                inference_ms=inference_ms,
                annotated=annotated,
                color_header=snapshot.color_header,
            )

        result = results[0]
        boxes = result.boxes
        detections_in_frame = 0 if boxes is None else int(len(boxes))
        if boxes is None or len(boxes) == 0:
            annotated = self._draw_no_detection_ui(color_image, "No prompt object detected")
            return FrameDecision(
                success=False,
                message="No objects detected for the requested prompt.",
                selected=None,
                point_in_camera=None,
                detections_in_frame=detections_in_frame,
                inference_ms=inference_ms,
                annotated=annotated,
                color_header=snapshot.color_header,
            )

        selected = self._select_pointed_detection(
            result=result,
            image_w=image_w,
            image_h=image_h,
            color_image=color_image,
            depth_image=depth_image,
            depth_encoding=snapshot.depth_encoding,
        )
        if selected is None:
            annotated = self._draw_unmatched_ui(color_image, result)
            return FrameDecision(
                success=False,
                message="VLM could not confidently select a pointed object from current view.",
                selected=None,
                point_in_camera=None,
                detections_in_frame=detections_in_frame,
                inference_ms=inference_ms,
                annotated=annotated,
                color_header=snapshot.color_header,
            )

        depth_m = selected.object_depth_m
        if depth_m is None:
            depth_m = self._sample_depth_meters(
                depth_image,
                snapshot.depth_encoding,
                selected.center_uv[0],
                selected.center_uv[1],
            )
        if depth_m is None:
            annotated = self._draw_depth_failure_ui(color_image, result, selected)
            return FrameDecision(
                success=False,
                message="Selected pointed object has invalid depth at centroid.",
                selected=None,
                point_in_camera=None,
                detections_in_frame=detections_in_frame,
                inference_ms=inference_ms,
                annotated=annotated,
                color_header=snapshot.color_header,
            )

        fx = float(snapshot.camera_info.k[0])
        fy = float(snapshot.camera_info.k[4])
        cx = float(snapshot.camera_info.k[2])
        cy = float(snapshot.camera_info.k[5])
        if fx <= 0.0 or fy <= 0.0:
            annotated = self._draw_no_detection_ui(color_image, "Invalid camera intrinsics")
            return FrameDecision(
                success=False,
                message="Invalid camera intrinsics (fx/fy <= 0).",
                selected=None,
                point_in_camera=None,
                detections_in_frame=detections_in_frame,
                inference_ms=inference_ms,
                annotated=annotated,
                color_header=snapshot.color_header,
            )

        u, v = selected.center_uv
        point_in_depth = np.array(
            [
                ((float(u) - cx) / fx) * depth_m,
                ((float(v) - cy) / fy) * depth_m,
                depth_m,
            ],
            dtype=np.float64,
        )
        depth_frame = snapshot.depth_frame or snapshot.camera_info.header.frame_id
        point_in_camera = self._transform_point_to_camera_link(point_in_depth, depth_frame)
        if point_in_camera is None:
            annotated = self._draw_no_detection_ui(color_image, "Depth->camera TF unavailable")
            return FrameDecision(
                success=False,
                message=f"Failed TF transform from {depth_frame} to {self.camera_link_frame}.",
                selected=None,
                point_in_camera=None,
                detections_in_frame=detections_in_frame,
                inference_ms=inference_ms,
                annotated=annotated,
                color_header=snapshot.color_header,
            )

        annotated = self._draw_success_ui(
            color_image=color_image,
            result=result,
            selected=selected,
            point_in_camera=point_in_camera,
        )
        return FrameDecision(
            success=True,
            message="Pointed object resolved in this frame.",
            selected=selected,
            point_in_camera=point_in_camera,
            detections_in_frame=detections_in_frame,
            inference_ms=inference_ms,
            annotated=annotated,
            color_header=snapshot.color_header,
        )

    def _select_pointed_detection(
        self,
        result: Any,
        image_w: int,
        image_h: int,
        color_image: np.ndarray,
        depth_image: np.ndarray,
        depth_encoding: str,
    ) -> PointedDetection | None:
        boxes = result.boxes
        if boxes is None:
            return None

        candidates: list[PointedDetection] = []

        for i in range(len(boxes)):
            box = boxes[i]
            x1f, y1f, x2f, y2f = [float(value) for value in box.xyxy[0].tolist()]
            x1 = int(np.clip(round(x1f), 0, image_w - 1))
            y1 = int(np.clip(round(y1f), 0, image_h - 1))
            x2 = int(np.clip(round(x2f), 0, image_w - 1))
            y2 = int(np.clip(round(y2f), 0, image_h - 1))
            if x2 <= x1 or y2 <= y1:
                continue

            center_u = int(round((x1 + x2) * 0.5))
            center_v = int(round((y1 + y2) * 0.5))
            cls_id = int(box.cls.item())
            class_name = self._class_name(result.names, cls_id)
            confidence = float(box.conf.item())
            obj_depth = self._sample_depth_meters(depth_image, depth_encoding, center_u, center_v)
            candidates.append(
                PointedDetection(
                    class_name=class_name,
                    confidence=confidence,
                    box_xyxy=(x1, y1, x2, y2),
                    center_uv=(center_u, center_v),
                    score=-confidence,
                    object_depth_m=obj_depth,
                )
            )

        if not candidates:
            return None

        if len(candidates) > self.vlm_max_candidates:
            candidates = sorted(candidates, key=lambda c: c.confidence, reverse=True)[
                : self.vlm_max_candidates
            ]

        selected_idx = self._vlm_select_candidate(color_image, candidates)
        if selected_idx is not None and 0 <= selected_idx < len(candidates):
            return candidates[selected_idx]

        self._last_vlm_reason = "vlm_no_selection"
        return None

    def _vlm_select_candidate(
        self,
        color_image: np.ndarray,
        candidates: list[PointedDetection],
    ) -> int | None:
        if not candidates:
            return None

        if self.ensure_ollama_running and not self._ollama_ready():
            self._ensure_ollama_running()

        prompt_text = ", ".join(self._active_prompts) if self._active_prompts else "unknown"

        candidate_lines = [
            f"id={idx}, class={candidate.class_name}"
            for idx, candidate in enumerate(candidates)
        ]

        query_image = self._build_vlm_query_image(
            color_image=color_image,
            candidates=candidates,
            selected_id=None,
            status="pending",
        )
        self._publish_vlm_debug_image(query_image)
        image_b64 = self._encode_image_base64(query_image)
        prompt = (
            "You are selecting one object that a human is pointing to for a robot.\n"
            f"Requested prompt object classes: {prompt_text}\n"
            "Use the attached image where each candidate bounding box has label ID-<n>.\n"
            "Infer pointing direction only from the visual scene in the image.\n"
            "Choose exactly one candidate id that is being pointed to. If none is clearly pointed, use -1.\n"
            "Return JSON only:\n"
            '{"selected_id": int, "reason": string, "confidence": number}\n'
            "Candidate IDs:\n"
            + "\n".join(candidate_lines)
        )

        predict_attempts = [self.vlm_num_predict]
        if self.vlm_max_retries > 0 and self.vlm_retry_num_predict != self.vlm_num_predict:
            predict_attempts.extend([self.vlm_retry_num_predict] * self.vlm_max_retries)

        last_error = "unknown"
        for attempt_index, num_predict in enumerate(predict_attempts, start=1):
            payload = {
                "model": self.vlm_model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [image_b64],
                    }
                ],
                "stream": False,
                "keep_alive": self.vlm_keep_alive,
                "options": {
                    "temperature": 0.0,
                    "num_ctx": self.vlm_num_ctx,
                    "num_predict": int(num_predict),
                },
            }

            try:
                response = requests.post(
                    f"{self.ollama_base_url}/api/chat",
                    json=payload,
                    timeout=self.vlm_timeout_s,
                )
                response.raise_for_status()
                message = response.json().get("message", {})
                content = str(message.get("content", "") or "").strip()
                thinking = str(message.get("thinking", "") or "").strip()
                selected_id, confidence, reason, source = self._decode_vlm_selection(
                    content=content,
                    thinking=thinking,
                    candidate_count=len(candidates),
                )
                self._last_vlm_reason = (
                    f"vlm:{source}:id={selected_id},conf={confidence:.2f},reason={reason[:96]}"
                )
                query_selected = self._build_vlm_query_image(
                    color_image=color_image,
                    candidates=candidates,
                    selected_id=selected_id if selected_id >= 0 else None,
                    status=f"id={selected_id}, conf={confidence:.2f}, try={attempt_index}",
                )
                self._publish_vlm_debug_image(query_selected)
                if selected_id < 0:
                    return None
                return selected_id
            except Exception as exc:
                last_error = str(exc)
                self.get_logger().warn(
                    f"VLM selection attempt {attempt_index}/{len(predict_attempts)} failed: {exc}"
                )

        self._last_vlm_reason = f"vlm_error:{last_error[:160]}"
        query_error = self._build_vlm_query_image(
            color_image=color_image,
            candidates=candidates,
            selected_id=None,
            status="error",
        )
        self._publish_vlm_debug_image(query_error)
        return None

    def _build_vlm_query_image(
        self,
        color_image: np.ndarray,
        candidates: list[PointedDetection],
        selected_id: int | None = None,
        status: str = "",
    ) -> np.ndarray:
        annotated = color_image.copy()
        for idx, candidate in enumerate(candidates):
            x1, y1, x2, y2 = candidate.box_xyxy
            color = (255, 64, 64)
            thickness = 2
            if selected_id is not None and idx == selected_id:
                color = (0, 255, 0)
                thickness = 3
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            label = f"ID-{idx} {candidate.class_name} {candidate.confidence:.2f}"
            cv2.putText(
                annotated,
                label,
                (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                color,
                2,
                cv2.LINE_AA,
            )
            cv2.circle(annotated, candidate.center_uv, 5, color, -1)

        cv2.putText(
            annotated,
            f"Prompt: {', '.join(self._active_prompts)}",
            (12, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"VLM status: {status or 'ready'}",
            (12, 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        if selected_id is not None and selected_id >= 0:
            cv2.putText(
                annotated,
                f"VLM selected ID-{selected_id}",
                (12, 82),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.70,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        return annotated

    def _publish_vlm_debug_image(self, image_bgr: np.ndarray) -> None:
        if self._vlm_debug_image_pub is not None:
            try:
                msg = self._bridge.cv2_to_imgmsg(image_bgr, encoding="bgr8")
                self._safe_publish(self._vlm_debug_image_pub, msg)
            except CvBridgeError as exc:
                self.get_logger().warn(f"Failed to publish VLM debug image: {exc}")

        if self.show_ui and not self._ui_failed:
            try:
                cv2.imshow("VLM Pointing Query", image_bgr)
                cv2.waitKey(1)
            except cv2.error as exc:
                self._ui_failed = True
                self.get_logger().warn(f"VLM debug window disabled due to OpenCV error: {exc}")

    def _encode_image_base64(self, image_bgr: np.ndarray) -> str:
        image = image_bgr
        height, width = image.shape[:2]
        max_edge = max(height, width)
        if max_edge > self.vlm_image_max_edge:
            scale = float(self.vlm_image_max_edge) / float(max_edge)
            new_width = max(1, int(round(width * scale)))
            new_height = max(1, int(round(height * scale)))
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if not ok:
            raise RuntimeError("Failed to encode query image for VLM.")
        return base64.b64encode(encoded.tobytes()).decode("utf-8")

    @staticmethod
    def _parse_json_relaxed(text: str) -> dict[str, Any]:
        cleaned = text.strip()
        cleaned = re.sub(r"<think>.*?</think>", " ", cleaned, flags=re.DOTALL | re.IGNORECASE).strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if len(lines) >= 2:
                cleaned = "\n".join(lines[1:-1]).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            if not match:
                raise
            return json.loads(match.group(0))

    def _decode_vlm_selection(
        self,
        content: str,
        thinking: str,
        candidate_count: int,
    ) -> tuple[int, float, str, str]:
        parse_errors: list[str] = []
        sources: list[tuple[str, str]] = []
        if content:
            sources.append(("content", content))
        if self.vlm_use_thinking_fallback and thinking:
            sources.append(("thinking", thinking))

        if not sources:
            raise ValueError("empty VLM response (both content and thinking are empty)")

        for source, text in sources:
            try:
                parsed = self._parse_json_relaxed(text)
                selected_id = int(float(parsed.get("selected_id", -1)))
                if selected_id < -1 or selected_id >= candidate_count:
                    raise ValueError(
                        f"selected_id out of range: {selected_id}, candidates={candidate_count}"
                    )
                reason = str(parsed.get("reason", "")).strip()[:160]
                confidence = float(parsed.get("confidence", 0.0))
                confidence = float(np.clip(confidence, 0.0, 1.0))
                return selected_id, confidence, reason, source
            except Exception as exc:
                parse_errors.append(f"{source}:json:{exc}")

            heuristic_id = self._extract_selected_id_from_text(text)
            if heuristic_id is None:
                parse_errors.append(f"{source}:heuristic:no_selected_id")
                continue

            if heuristic_id < -1 or heuristic_id >= candidate_count:
                parse_errors.append(
                    f"{source}:heuristic:selected_id_out_of_range={heuristic_id}"
                )
                continue

            confidence = self._extract_confidence_from_text(
                text,
                default=0.5 if heuristic_id >= 0 else 0.0,
            )
            reason = self._extract_reason_from_text(text)
            return heuristic_id, confidence, reason, f"{source}_heuristic"

        raise ValueError("; ".join(parse_errors))

    @staticmethod
    def _extract_selected_id_from_text(text: str) -> int | None:
        patterns = [
            r'"selected_id"\s*:\s*(-?\d+)',
            r"\bselected[_\s-]*id\s*(?:is|=|:)\s*(-?\d+)",
            r"\b(?:pointed|target)\s*(?:object\s*)?id\s*(?:is|=|:)\s*(-?\d+)",
            r"\b(?:id)\s*(?:is|=|:)\s*(-?\d+)",
            r"\bthat's\s+id\s*(?:is|=|:)?\s*(-?\d+)",
        ]
        for pattern in patterns:
            matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
            if matches:
                return int(matches[-1].group(1))
        return None

    @staticmethod
    def _extract_confidence_from_text(text: str, default: float = 0.0) -> float:
        patterns = [
            r'"confidence"\s*:\s*([0-9]*\.?[0-9]+)',
            r"\bconfidence\s*(?:is|=|:)\s*([0-9]*\.?[0-9]+)",
        ]
        for pattern in patterns:
            matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
            if matches:
                value = float(matches[-1].group(1))
                return float(np.clip(value, 0.0, 1.0))
        return float(np.clip(default, 0.0, 1.0))

    @staticmethod
    def _extract_reason_from_text(text: str) -> str:
        compact = re.sub(r"\s+", " ", text).strip()
        if not compact:
            return "heuristic_parse"
        if len(compact) > 160:
            compact = compact[-160:]
        return compact

    def _publish_ui_outputs(
        self,
        annotated: np.ndarray,
        color_header: Any,
        prompts: list[str],
        label: str,
        save_image_request: bool,
    ) -> str:
        if self._result_image_pub is not None:
            try:
                msg = self._bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
                if color_header is not None:
                    msg.header = color_header
                self._safe_publish(self._result_image_pub, msg)
            except CvBridgeError as exc:
                self.get_logger().warn(f"Failed to publish result image: {exc}")

        if self.show_ui and not self._ui_failed:
            try:
                cv2.imshow("Pointed Object Detection", annotated)
                cv2.waitKey(self.ui_wait_ms)
            except cv2.error as exc:
                self._ui_failed = True
                self.get_logger().warn(f"UI display disabled due to OpenCV error: {exc}")

        if self.always_save_image or save_image_request:
            return self._save_annotated_image(annotated, prompts, label)
        return ""

    def _draw_no_detection_ui(
        self,
        color_image: np.ndarray,
        message: str,
    ) -> np.ndarray:
        annotated = color_image.copy()
        cv2.putText(
            annotated,
            message,
            (12, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return annotated

    def _draw_unmatched_ui(
        self,
        color_image: np.ndarray,
        result: Any,
    ) -> np.ndarray:
        annotated = color_image.copy()
        self._draw_all_boxes(annotated, result, default_color=(0, 0, 255))
        cv2.putText(
            annotated,
            "VLM could not select a pointed object",
            (12, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return annotated

    def _draw_depth_failure_ui(
        self,
        color_image: np.ndarray,
        result: Any,
        selected: PointedDetection,
    ) -> np.ndarray:
        annotated = color_image.copy()
        self._draw_all_boxes(annotated, result, default_color=(0, 0, 255))
        self._highlight_selected_box(annotated, selected, (0, 165, 255))
        cv2.putText(
            annotated,
            "Selected object depth invalid",
            (12, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            (0, 165, 255),
            2,
            cv2.LINE_AA,
        )
        return annotated

    def _draw_success_ui(
        self,
        color_image: np.ndarray,
        result: Any,
        selected: PointedDetection,
        point_in_camera: np.ndarray,
    ) -> np.ndarray:
        annotated = color_image.copy()
        self._draw_all_boxes(annotated, result, default_color=(60, 60, 220))
        self._highlight_selected_box(annotated, selected, (0, 255, 0))

        cv2.putText(
            annotated,
            f"Selected: {selected.class_name} {selected.confidence:.2f} (VLM)",
            (12, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.66,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            (
                f"Centroid {self.camera_link_frame}: "
                f"[{point_in_camera[0]:.2f}, {point_in_camera[1]:.2f}, {point_in_camera[2]:.2f}] m"
            ),
            (12, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return annotated

    @staticmethod
    def _draw_all_boxes(image: np.ndarray, result: Any, default_color: tuple[int, int, int]) -> None:
        boxes = result.boxes
        if boxes is None:
            return
        names = result.names
        for i in range(len(boxes)):
            box = boxes[i]
            x1f, y1f, x2f, y2f = [float(value) for value in box.xyxy[0].tolist()]
            x1, y1, x2, y2 = int(round(x1f)), int(round(y1f)), int(round(x2f)), int(round(y2f))
            cls_id = int(box.cls.item())
            class_name = YoloeVlmPointedDetectionServiceNode._class_name(names, cls_id)
            conf = float(box.conf.item())
            cv2.rectangle(image, (x1, y1), (x2, y2), default_color, 2)
            cv2.putText(
                image,
                f"{class_name} {conf:.2f}",
                (x1, max(16, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                default_color,
                2,
                cv2.LINE_AA,
            )

    @staticmethod
    def _highlight_selected_box(
        image: np.ndarray,
        selected: PointedDetection,
        color: tuple[int, int, int],
    ) -> None:
        x1, y1, x2, y2 = selected.box_xyxy
        u, v = selected.center_uv
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        cv2.circle(image, (u, v), 6, color, -1)

    def _publish_tf(self, child_frame: str, translation: np.ndarray) -> None:
        if self.tf_ttl_sec <= 0.0:
            self._send_tf_transform(child_frame, translation)
            return

        expires_at = time.monotonic() + self.tf_ttl_sec
        with self._tf_publish_lock:
            self._active_timed_tfs[child_frame] = TimedTransform(
                translation=np.array(translation, dtype=np.float64),
                expires_at_monotonic=expires_at,
            )
        self._send_tf_transform(child_frame, translation)

    def _on_tf_publish_timer(self) -> None:
        now = time.monotonic()
        to_publish: list[tuple[str, np.ndarray]] = []
        expired: list[str] = []

        with self._tf_publish_lock:
            for child_frame, timed_tf in list(self._active_timed_tfs.items()):
                if timed_tf.expires_at_monotonic <= now:
                    expired.append(child_frame)
                    self._active_timed_tfs.pop(child_frame, None)
                    continue
                to_publish.append((child_frame, timed_tf.translation.copy()))

        for child_frame, translation in to_publish:
            self._send_tf_transform(child_frame, translation)

        for child_frame in expired:
            self.get_logger().info(
                f"Expired TF frame '{child_frame}' after {self.tf_ttl_sec:.1f}s"
            )

    def _send_tf_transform(self, child_frame: str, translation: np.ndarray) -> None:
        tf_msg = TransformStamped()
        tf_msg.header.stamp = self.get_clock().now().to_msg()
        tf_msg.header.frame_id = self.camera_link_frame
        tf_msg.child_frame_id = child_frame
        tf_msg.transform.translation.x = float(translation[0])
        tf_msg.transform.translation.y = float(translation[1])
        tf_msg.transform.translation.z = float(translation[2])
        tf_msg.transform.rotation.w = 1.0
        self._tf_broadcaster.sendTransform(tf_msg)

    def _transform_point_to_camera_link(
        self, point_in_source: np.ndarray, source_frame: str
    ) -> np.ndarray | None:
        if source_frame == self.camera_link_frame or source_frame == "":
            return point_in_source

        try:
            transform = self._tf_buffer.lookup_transform(
                self.camera_link_frame,
                source_frame,
                Time(),
            )
        except TransformException as exc:
            self.get_logger().warn(
                f"TF lookup failed ({source_frame} -> {self.camera_link_frame}): {exc}"
            )
            return None

        rotation = transform.transform.rotation
        translation = transform.transform.translation
        rotated = self._rotate_vector_by_quaternion(
            point_in_source,
            rotation.x,
            rotation.y,
            rotation.z,
            rotation.w,
        )
        transformed = rotated + np.array(
            [translation.x, translation.y, translation.z], dtype=np.float64
        )
        return transformed

    @staticmethod
    def _rotate_vector_by_quaternion(
        vector: np.ndarray,
        qx: float,
        qy: float,
        qz: float,
        qw: float,
    ) -> np.ndarray:
        norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        if norm == 0.0:
            return vector

        x = qx / norm
        y = qy / norm
        z = qz / norm
        w = qw / norm

        xx = x * x
        yy = y * y
        zz = z * z
        xy = x * y
        xz = x * z
        yz = y * z
        wx = w * x
        wy = w * y
        wz = w * z

        rotation_matrix = np.array(
            [
                [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
                [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
                [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
            ],
            dtype=np.float64,
        )
        return rotation_matrix.dot(vector)

    def _sample_depth_meters(
        self,
        depth_image: np.ndarray,
        encoding: str,
        u: int,
        v: int,
    ) -> float | None:
        height, width = depth_image.shape[:2]
        if u < 0 or v < 0 or u >= width or v >= height:
            return None

        radius = max(0, self.depth_window_size // 2)
        values: list[float] = []
        for yy in range(max(0, v - radius), min(height, v + radius + 1)):
            for xx in range(max(0, u - radius), min(width, u + radius + 1)):
                depth_m = self._depth_value_to_meters(
                    depth_image[yy, xx], encoding, depth_image.dtype
                )
                if depth_m is None:
                    continue
                if depth_m < self.min_depth_m or depth_m > self.max_depth_m:
                    continue
                values.append(depth_m)

        if not values:
            return None
        return float(np.median(values))

    @staticmethod
    def _depth_value_to_meters(
        value: Any, encoding: str, dtype: np.dtype[Any]
    ) -> float | None:
        depth_value = float(value)
        if not math.isfinite(depth_value) or depth_value <= 0.0:
            return None

        enc = encoding.upper() if encoding else ""
        if "16U" in enc or "MONO16" in enc:
            return depth_value * 0.001
        if "32F" in enc:
            return depth_value

        if dtype in (np.uint16, np.int16):
            return depth_value * 0.001
        if dtype in (np.float32, np.float64):
            return depth_value
        return depth_value

    @staticmethod
    def _parse_prompts(prompt_text: str) -> list[str]:
        prompts = [entry.strip() for entry in prompt_text.split(",") if entry.strip()]
        if not prompts:
            raise ValueError("prompt_text is empty. Example: 'bottle' or 'cup,bottle'.")
        return prompts

    @staticmethod
    def _slug(text: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9_]+", "_", text).strip("_").lower()
        return slug or "object"

    @staticmethod
    def _class_name(names: Any, class_id: int) -> str:
        if isinstance(names, dict):
            return str(names.get(class_id, class_id))
        if isinstance(names, list) and 0 <= class_id < len(names):
            return str(names[class_id])
        return str(class_id)

    def _save_annotated_image(
        self, image: np.ndarray, prompts: list[str], detected_class: str
    ) -> str:
        prompt_text = "_".join(prompts)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{ts}_{self._slug(prompt_text)}_{self._slug(detected_class)}_pointed.jpg"
        path = self.save_dir / filename
        cv2.imwrite(str(path), image)
        self.get_logger().info(f"Saved pointed-object image: {path}")
        return str(path)

    @staticmethod
    def _safe_publish(publisher: Any, msg: Any) -> None:
        if not rclpy.ok():
            return
        try:
            publisher.publish(msg)
        except Exception:
            return

    def destroy_node(self) -> bool:
        if self.show_ui and not self._ui_failed:
            cv2.destroyAllWindows()
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = YoloeVlmPointedDetectionServiceNode()
    try:
        rclpy.spin(node)
    except ExternalShutdownException:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

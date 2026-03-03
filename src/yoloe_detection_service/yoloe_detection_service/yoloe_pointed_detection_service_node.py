#!/usr/bin/env python3
"""One-shot YOLOE service that returns the object pointed by a human hand/arm."""

from __future__ import annotations

import math
import pathlib
import re
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import rclpy
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
class PointingCue:
    source: str  # "hand" or "arm"
    side: str  # "left" or "right" or "unknown"
    tip_uv: tuple[int, int]
    ray_end_uv: tuple[int, int]
    direction_uv: np.ndarray
    max_angle_deg: float
    min_forward_px: float
    tip_depth_m: float | None


@dataclass
class PointedDetection:
    class_name: str
    confidence: float
    box_xyxy: tuple[int, int, int, int]
    center_uv: tuple[int, int]
    cue_source: str
    cue_side: str
    tip_uv: tuple[int, int]
    ray_end_uv: tuple[int, int]
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


class YoloePointedDetectionServiceNode(Node):
    def __init__(self) -> None:
        super().__init__("yoloe_pointed_detection_service_node")

        self.declare_parameter("service_name", "/yoloe/detect_pointed_prompt")
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
        self.declare_parameter("camera_link_frame", "camera0_color_optical_frame")
        self.declare_parameter("pose_topic", "/yoloe/pointed_object_pose")
        self.declare_parameter("result_image_topic", "/yoloe/pointing_result_image")
        self.declare_parameter("object_frame_prefix", "pointed")

        self.declare_parameter("depth_window_size", 7)
        self.declare_parameter("min_depth_m", 0.05)
        self.declare_parameter("max_depth_m", 10.0)

        self.declare_parameter("max_num_hands", 2)
        self.declare_parameter("hand_min_detection_confidence", 0.45)
        self.declare_parameter("hand_min_tracking_confidence", 0.45)
        self.declare_parameter("pointing_min_forward_px", 24.0)
        self.declare_parameter("pointing_max_angle_deg", 20.0)

        self.declare_parameter("arm_pointing_min_forward_px", 20.0)
        self.declare_parameter("arm_pointing_max_angle_deg", 34.0)
        self.declare_parameter("min_pose_visibility", 0.45)
        self.declare_parameter("min_arm_length_px", 18.0)

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

        self.depth_window_size = max(1, int(self.get_parameter("depth_window_size").value))
        self.min_depth_m = float(self.get_parameter("min_depth_m").value)
        self.max_depth_m = float(self.get_parameter("max_depth_m").value)

        self.max_num_hands = max(1, int(self.get_parameter("max_num_hands").value))
        self.hand_min_detection_confidence = float(
            self.get_parameter("hand_min_detection_confidence").value
        )
        self.hand_min_tracking_confidence = float(
            self.get_parameter("hand_min_tracking_confidence").value
        )
        self.pointing_min_forward_px = float(self.get_parameter("pointing_min_forward_px").value)
        self.pointing_max_angle_deg = float(self.get_parameter("pointing_max_angle_deg").value)

        self.arm_pointing_min_forward_px = float(
            self.get_parameter("arm_pointing_min_forward_px").value
        )
        self.arm_pointing_max_angle_deg = float(
            self.get_parameter("arm_pointing_max_angle_deg").value
        )
        self.min_pose_visibility = float(self.get_parameter("min_pose_visibility").value)
        self.min_arm_length_px = float(self.get_parameter("min_arm_length_px").value)

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

        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            model_complexity=0,
            max_num_hands=self.max_num_hands,
            min_detection_confidence=self.hand_min_detection_confidence,
            min_tracking_confidence=self.hand_min_tracking_confidence,
        )
        self._mp_pose = mp.solutions.pose
        self._pose = self._mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,
            enable_segmentation=False,
            min_detection_confidence=self.hand_min_detection_confidence,
            min_tracking_confidence=self.hand_min_tracking_confidence,
        )

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

        self.create_subscription(Image, self.color_topic, self._on_color_image, qos_profile_sensor_data)
        self.create_subscription(Image, self.depth_topic, self._on_depth_image, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, self.camera_info_topic, self._on_camera_info, qos_profile_sensor_data)
        self.create_service(DetectObjectPrompt, self.service_name, self._handle_detect_request)

        self._load_model()

        self.get_logger().info(f"Pointed-object service ready on {self.service_name}")
        self.get_logger().info(f"Model path: {self.model_path}")
        self.get_logger().info(f"Device: {self._device}")
        self.get_logger().info(f"Color topic: {self.color_topic}")
        self.get_logger().info(f"Depth topic: {self.depth_topic}")
        self.get_logger().info(f"Camera info topic: {self.camera_info_topic}")
        self.get_logger().info(
            f"Hybrid pointing cues enabled: hand_cone={self.pointing_max_angle_deg:.1f} deg, "
            f"arm_cone={self.arm_pointing_max_angle_deg:.1f} deg"
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
        saved_path = self._publish_ui_outputs(
            annotated=annotated,
            color_header=final_decision.color_header,
            prompts=prompts,
            label=final_selected.class_name,
            save_image_request=save_image_request,
        )

        message = (
            f"Selected '{final_selected.class_name}' via {final_selected.cue_side}/{final_selected.cue_source} "
            f"cue. Votes {winner_vote.count}/{len(frame_decisions)}. "
            f"Centroid [{final_point[0]:.3f}, {final_point[1]:.3f}, {final_point[2]:.3f}] m "
            f"in {self.camera_link_frame}."
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
        cues = self._extract_pointing_cues(
            image_bgr=color_image,
            width=image_w,
            height=image_h,
            depth_image=depth_image,
            depth_encoding=snapshot.depth_encoding,
        )
        if not cues:
            annotated = self._draw_no_pointing_ui(color_image)
            return FrameDecision(
                success=False,
                message="No clear pointing cue detected (hand/arm).",
                selected=None,
                point_in_camera=None,
                detections_in_frame=0,
                inference_ms=0.0,
                annotated=annotated,
                color_header=snapshot.color_header,
            )

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
            annotated = self._draw_no_detection_ui(color_image, cues, "YOLOE returned no results")
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
            annotated = self._draw_no_detection_ui(color_image, cues, "No prompt object detected")
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
            cues=cues,
            image_w=image_w,
            image_h=image_h,
            depth_image=depth_image,
            depth_encoding=snapshot.depth_encoding,
        )
        if selected is None:
            annotated = self._draw_unmatched_ui(color_image, result, cues)
            return FrameDecision(
                success=False,
                message="Detections exist, but none align with the pointing cue.",
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
            annotated = self._draw_depth_failure_ui(color_image, result, selected, cues)
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
            annotated = self._draw_no_detection_ui(color_image, cues, "Invalid camera intrinsics")
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
            annotated = self._draw_no_detection_ui(color_image, cues, "Depth->camera TF unavailable")
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
            cues=cues,
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

    def _extract_pointing_cues(
        self,
        image_bgr: np.ndarray,
        width: int,
        height: int,
        depth_image: np.ndarray,
        depth_encoding: str,
    ) -> list[PointingCue]:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        hand_result = self._hands.process(rgb)
        pose_result = self._pose.process(rgb)

        cues: list[PointingCue] = []

        if hand_result.multi_hand_landmarks is not None:
            for idx, hand_landmarks in enumerate(hand_result.multi_hand_landmarks):
                side = "unknown"
                if hand_result.multi_handedness and idx < len(hand_result.multi_handedness):
                    classes = hand_result.multi_handedness[idx].classification
                    if classes:
                        side = classes[0].label.lower()

                lm = hand_landmarks.landmark
                if not self._is_pointing_gesture(lm):
                    continue

                tip = self._landmark_to_pixel(lm[8], width, height)
                pip = self._landmark_to_pixel(lm[6], width, height)
                direction = np.array([tip[0] - pip[0], tip[1] - pip[1]], dtype=np.float32)
                norm = float(np.linalg.norm(direction))
                if norm < 1e-6:
                    continue
                direction /= norm

                ray_end = self._ray_to_image_boundary(tip, direction, width, height)
                tip_depth = self._sample_depth_meters(depth_image, depth_encoding, tip[0], tip[1])
                cues.append(
                    PointingCue(
                        source="hand",
                        side=side,
                        tip_uv=tip,
                        ray_end_uv=ray_end,
                        direction_uv=direction,
                        max_angle_deg=self.pointing_max_angle_deg,
                        min_forward_px=self.pointing_min_forward_px,
                        tip_depth_m=tip_depth,
                    )
                )

        pose_landmarks = pose_result.pose_landmarks
        if pose_landmarks is not None:
            lm = pose_landmarks.landmark
            side_map = {
                "left": (11, 13, 15),
                "right": (12, 14, 16),
            }
            for side, (shoulder_idx, elbow_idx, wrist_idx) in side_map.items():
                shoulder = lm[shoulder_idx]
                elbow = lm[elbow_idx]
                wrist = lm[wrist_idx]

                if float(wrist.visibility) < self.min_pose_visibility:
                    continue
                if float(elbow.visibility) < self.min_pose_visibility:
                    continue

                shoulder_uv = self._landmark_to_pixel(shoulder, width, height)
                elbow_uv = self._landmark_to_pixel(elbow, width, height)
                wrist_uv = self._landmark_to_pixel(wrist, width, height)

                direction = np.array(
                    [wrist_uv[0] - elbow_uv[0], wrist_uv[1] - elbow_uv[1]],
                    dtype=np.float32,
                )
                norm = float(np.linalg.norm(direction))

                if norm < self.min_arm_length_px and float(shoulder.visibility) >= self.min_pose_visibility:
                    direction = np.array(
                        [wrist_uv[0] - shoulder_uv[0], wrist_uv[1] - shoulder_uv[1]],
                        dtype=np.float32,
                    )
                    norm = float(np.linalg.norm(direction))

                if norm < self.min_arm_length_px:
                    continue

                direction /= norm
                ray_end = self._ray_to_image_boundary(wrist_uv, direction, width, height)
                tip_depth = self._sample_depth_meters(
                    depth_image,
                    depth_encoding,
                    wrist_uv[0],
                    wrist_uv[1],
                )
                cues.append(
                    PointingCue(
                        source="arm",
                        side=side,
                        tip_uv=wrist_uv,
                        ray_end_uv=ray_end,
                        direction_uv=direction,
                        max_angle_deg=self.arm_pointing_max_angle_deg,
                        min_forward_px=self.arm_pointing_min_forward_px,
                        tip_depth_m=tip_depth,
                    )
                )

        return cues

    @staticmethod
    def _is_pointing_gesture(landmarks: Any) -> bool:
        wrist = np.array([float(landmarks[0].x), float(landmarks[0].y)], dtype=np.float32)

        def extension_ratio(tip_idx: int, pip_idx: int) -> float:
            tip = np.array([float(landmarks[tip_idx].x), float(landmarks[tip_idx].y)], dtype=np.float32)
            pip = np.array([float(landmarks[pip_idx].x), float(landmarks[pip_idx].y)], dtype=np.float32)
            base = max(1e-6, float(np.linalg.norm(pip - wrist)))
            return float(np.linalg.norm(tip - wrist)) / base

        index_ratio = extension_ratio(8, 6)
        middle_ratio = extension_ratio(12, 10)
        ring_ratio = extension_ratio(16, 14)
        pinky_ratio = extension_ratio(20, 18)

        return index_ratio > 1.15 and middle_ratio < 1.08 and ring_ratio < 1.08 and pinky_ratio < 1.08

    def _select_pointed_detection(
        self,
        result: Any,
        cues: list[PointingCue],
        image_w: int,
        image_h: int,
        depth_image: np.ndarray,
        depth_encoding: str,
    ) -> PointedDetection | None:
        boxes = result.boxes
        best: PointedDetection | None = None
        best_score = float("inf")

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

            for cue in cues:
                vec = np.array(
                    [center_u - cue.tip_uv[0], center_v - cue.tip_uv[1]],
                    dtype=np.float32,
                )
                vec_norm = float(np.linalg.norm(vec))
                if vec_norm < 1e-6:
                    continue

                proj = float(np.dot(vec, cue.direction_uv))
                if proj < cue.min_forward_px:
                    continue

                cosine = float(np.clip(proj / vec_norm, -1.0, 1.0))
                angle_deg = float(math.degrees(math.acos(cosine)))
                if angle_deg > cue.max_angle_deg:
                    continue

                perp_dist = abs(
                    float(cue.direction_uv[0]) * float(vec[1])
                    - float(cue.direction_uv[1]) * float(vec[0])
                )

                score = angle_deg + (perp_dist / max(20.0, proj)) * 24.0 - confidence * 2.0
                if cue.source == "arm":
                    score += 2.2
                else:
                    score -= 0.6

                if obj_depth is None:
                    score += 1.5
                if cue.tip_depth_m is not None and obj_depth is not None:
                    if obj_depth + 0.08 < cue.tip_depth_m:
                        score += 8.0
                    elif obj_depth > cue.tip_depth_m:
                        score -= min(1.8, (obj_depth - cue.tip_depth_m) * 0.5)

                if score < best_score:
                    best_score = score
                    best = PointedDetection(
                        class_name=class_name,
                        confidence=confidence,
                        box_xyxy=(x1, y1, x2, y2),
                        center_uv=(center_u, center_v),
                        cue_source=cue.source,
                        cue_side=cue.side,
                        tip_uv=cue.tip_uv,
                        ray_end_uv=cue.ray_end_uv,
                        score=score,
                        object_depth_m=obj_depth,
                    )

        return best

    @staticmethod
    def _landmark_to_pixel(landmark: Any, width: int, height: int) -> tuple[int, int]:
        x = int(np.clip(round(float(landmark.x) * width), 0, width - 1))
        y = int(np.clip(round(float(landmark.y) * height), 0, height - 1))
        return x, y

    @staticmethod
    def _ray_to_image_boundary(
        tip_uv: tuple[int, int],
        direction_uv: np.ndarray,
        width: int,
        height: int,
    ) -> tuple[int, int]:
        tip = np.array([float(tip_uv[0]), float(tip_uv[1])], dtype=np.float32)
        direction = direction_uv.astype(np.float32)
        candidates: list[float] = []

        if abs(float(direction[0])) > 1e-6:
            for bound_x in (0.0, float(width - 1)):
                t = (bound_x - float(tip[0])) / float(direction[0])
                if t > 0.0:
                    y = float(tip[1]) + t * float(direction[1])
                    if 0.0 <= y <= float(height - 1):
                        candidates.append(t)

        if abs(float(direction[1])) > 1e-6:
            for bound_y in (0.0, float(height - 1)):
                t = (bound_y - float(tip[1])) / float(direction[1])
                if t > 0.0:
                    x = float(tip[0]) + t * float(direction[0])
                    if 0.0 <= x <= float(width - 1):
                        candidates.append(t)

        if not candidates:
            far = tip + direction * max(width, height)
            return (
                int(np.clip(round(float(far[0])), 0, width - 1)),
                int(np.clip(round(float(far[1])), 0, height - 1)),
            )

        t_min = min(candidates)
        point = tip + direction * t_min
        return (
            int(np.clip(round(float(point[0])), 0, width - 1)),
            int(np.clip(round(float(point[1])), 0, height - 1)),
        )

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

    def _draw_no_pointing_ui(self, color_image: np.ndarray) -> np.ndarray:
        annotated = color_image.copy()
        cv2.putText(
            annotated,
            "No clear pointing cue (hand/arm)",
            (12, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return annotated

    def _draw_no_detection_ui(
        self,
        color_image: np.ndarray,
        cues: list[PointingCue],
        message: str,
    ) -> np.ndarray:
        annotated = color_image.copy()
        self._draw_pointing_cues(annotated, cues)
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
        cues: list[PointingCue],
    ) -> np.ndarray:
        annotated = color_image.copy()
        self._draw_all_boxes(annotated, result, default_color=(0, 0, 255))
        self._draw_pointing_cues(annotated, cues)
        cv2.putText(
            annotated,
            "Detected objects not aligned with pointing cone",
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
        cues: list[PointingCue],
    ) -> np.ndarray:
        annotated = color_image.copy()
        self._draw_all_boxes(annotated, result, default_color=(0, 0, 255))
        self._draw_pointing_cues(annotated, cues)
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
        cues: list[PointingCue],
        point_in_camera: np.ndarray,
    ) -> np.ndarray:
        annotated = color_image.copy()
        self._draw_all_boxes(annotated, result, default_color=(60, 60, 220))
        self._draw_pointing_cues(annotated, cues)
        self._highlight_selected_box(annotated, selected, (0, 255, 0))

        cv2.putText(
            annotated,
            (
                f"Selected: {selected.class_name} {selected.confidence:.2f} "
                f"({selected.cue_side}/{selected.cue_source})"
            ),
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
    def _draw_pointing_cues(image: np.ndarray, cues: list[PointingCue]) -> None:
        for cue in cues:
            color = (0, 255, 255) if cue.source == "hand" else (255, 255, 0)
            cv2.arrowedLine(
                image,
                cue.tip_uv,
                cue.ray_end_uv,
                color,
                2,
                tipLength=0.06,
            )
            cv2.circle(image, cue.tip_uv, 5, color, -1)
            cv2.putText(
                image,
                f"{cue.side}-{cue.source}",
                (cue.tip_uv[0] + 8, max(16, cue.tip_uv[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )

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
            class_name = YoloePointedDetectionServiceNode._class_name(names, cls_id)
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
        cv2.arrowedLine(
            image,
            selected.tip_uv,
            selected.ray_end_uv,
            color,
            2,
            tipLength=0.06,
        )

    def _publish_tf(self, child_frame: str, translation: np.ndarray) -> None:
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
        self._hands.close()
        self._pose.close()
        if self.show_ui and not self._ui_failed:
            cv2.destroyAllWindows()
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = YoloePointedDetectionServiceNode()
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

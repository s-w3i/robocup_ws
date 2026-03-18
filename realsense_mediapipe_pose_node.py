#!/usr/bin/env python3
"""ROS 2 node for MediaPipe human pose detection on a RealSense color topic."""

from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String


class RealsenseMediapipePoseNode(Node):
    SUPPORTED_ENCODINGS: dict[str, int] = {
        "bgr8": 3,
        "rgb8": 3,
        "bgra8": 4,
        "rgba8": 4,
        "mono8": 1,
    }

    def __init__(self) -> None:
        super().__init__("realsense_mediapipe_pose_node")

        self.declare_parameter("image_topic", "/camera0/color/image_raw")
        self.declare_parameter("annotated_topic", "/camera0/color/pose_annotated")
        self.declare_parameter("gesture_topic", "/camera0/hand_gesture")
        self.declare_parameter("pointing_topic", "/camera0/pointing_pixel")
        self.declare_parameter("publish_annotated", True)
        self.declare_parameter("show_image", True)
        self.declare_parameter("enable_pose", True)
        self.declare_parameter("enable_hands", True)
        self.declare_parameter("static_image_mode", False)
        self.declare_parameter("model_complexity", 1)
        self.declare_parameter("min_detection_confidence", 0.5)
        self.declare_parameter("min_tracking_confidence", 0.5)
        self.declare_parameter("max_num_hands", 2)
        self.declare_parameter("hand_min_detection_confidence", 0.5)
        self.declare_parameter("hand_min_tracking_confidence", 0.5)
        self.declare_parameter("input_width", 640)
        self.declare_parameter("process_every_n", 1)
        self.declare_parameter("draw_landmarks", True)
        self.declare_parameter("draw_hand_landmarks", True)

        self._image_topic = str(self.get_parameter("image_topic").value)
        self._annotated_topic = str(self.get_parameter("annotated_topic").value)
        self._gesture_topic = str(self.get_parameter("gesture_topic").value)
        self._pointing_topic = str(self.get_parameter("pointing_topic").value)
        self._publish_annotated = bool(self.get_parameter("publish_annotated").value)
        self._show_image = bool(self.get_parameter("show_image").value)
        self._enable_pose = bool(self.get_parameter("enable_pose").value)
        self._enable_hands = bool(self.get_parameter("enable_hands").value)
        self._input_width = int(self.get_parameter("input_width").value)
        self._process_every_n = max(1, int(self.get_parameter("process_every_n").value))
        self._draw_pose_landmarks = bool(self.get_parameter("draw_landmarks").value)
        self._draw_hand_landmarks = bool(self.get_parameter("draw_hand_landmarks").value)
        self._static_image_mode = bool(self.get_parameter("static_image_mode").value)
        self._model_complexity = int(self.get_parameter("model_complexity").value)
        self._min_detection_confidence = float(self.get_parameter("min_detection_confidence").value)
        self._min_tracking_confidence = float(self.get_parameter("min_tracking_confidence").value)
        self._max_num_hands = int(self.get_parameter("max_num_hands").value)
        self._hand_min_detection_confidence = float(
            self.get_parameter("hand_min_detection_confidence").value
        )
        self._hand_min_tracking_confidence = float(
            self.get_parameter("hand_min_tracking_confidence").value
        )

        self._window_name = "RealSense MediaPipe Pose"
        self._warned_encodings: set[str] = set()
        self._frame_count = 0
        self._last_pose_landmarks = None
        self._last_hand_landmarks = []
        self._last_handedness = []

        self._mp_pose = mp.solutions.pose
        self._mp_hands = mp.solutions.hands
        self._drawing_utils = mp.solutions.drawing_utils
        self._drawing_styles = mp.solutions.drawing_styles
        self._pose = None
        if self._enable_pose:
            self._pose = self._mp_pose.Pose(
                static_image_mode=self._static_image_mode,
                model_complexity=self._model_complexity,
                enable_segmentation=False,
                min_detection_confidence=self._min_detection_confidence,
                min_tracking_confidence=self._min_tracking_confidence,
            )
        self._hands = None
        if self._enable_hands:
            self._hands = self._mp_hands.Hands(
                static_image_mode=self._static_image_mode,
                max_num_hands=self._max_num_hands,
                model_complexity=0,
                min_detection_confidence=self._hand_min_detection_confidence,
                min_tracking_confidence=self._hand_min_tracking_confidence,
            )

        self._annotated_pub = None
        if self._publish_annotated:
            self._annotated_pub = self.create_publisher(
                Image,
                self._annotated_topic,
                qos_profile_sensor_data,
            )

        self._gesture_pub = self.create_publisher(
            String,
            self._gesture_topic,
            qos_profile_sensor_data,
        )
        self._pointing_pub = self.create_publisher(
            PointStamped,
            self._pointing_topic,
            qos_profile_sensor_data,
        )

        self._image_sub = self.create_subscription(
            Image,
            self._image_topic,
            self._image_callback,
            qos_profile_sensor_data,
        )

        self.get_logger().info(
            f"Listening on '{self._image_topic}', publishing annotated frames: "
            f"{self._publish_annotated} ({self._annotated_topic}), "
            f"input_width={self._input_width}, process_every_n={self._process_every_n}"
        )
        self.get_logger().info(
            f"Pose={self._enable_pose}, Hands={self._enable_hands}, "
            f"gesture_topic={self._gesture_topic}, pointing_topic={self._pointing_topic}"
        )
        if self._show_image:
            self.get_logger().info("Press 'q' or ESC in the preview window to stop.")

    def _image_callback(self, msg: Image) -> None:
        self._frame_count += 1
        frame_bgr = self._ros_image_to_bgr(msg)
        if frame_bgr is None:
            return

        run_inference = (
            self._last_pose_landmarks is None
            or self._frame_count % self._process_every_n == 0
        )
        if run_inference:
            inference_bgr = frame_bgr
            if self._input_width > 0 and frame_bgr.shape[1] > self._input_width:
                new_width = self._input_width
                new_height = max(1, int(frame_bgr.shape[0] * new_width / frame_bgr.shape[1]))
                inference_bgr = cv2.resize(
                    frame_bgr,
                    (new_width, new_height),
                    interpolation=cv2.INTER_AREA,
                )

            frame_rgb = cv2.cvtColor(inference_bgr, cv2.COLOR_BGR2RGB)
            if self._pose is not None:
                pose_results = self._pose.process(frame_rgb)
                self._last_pose_landmarks = pose_results.pose_landmarks
            if self._hands is not None:
                hands_results = self._hands.process(frame_rgb)
                self._last_hand_landmarks = hands_results.multi_hand_landmarks or []
                self._last_handedness = hands_results.multi_handedness or []

        annotated = frame_bgr.copy()
        if self._draw_pose_landmarks and self._last_pose_landmarks:
            self._drawing_utils.draw_landmarks(
                annotated,
                self._last_pose_landmarks,
                self._mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self._drawing_styles.get_default_pose_landmarks_style(),
            )

        hand_gestures: list[str] = []
        pointing_pixel: tuple[int, int] | None = None

        for idx, hand_landmarks in enumerate(self._last_hand_landmarks):
            handedness = self._get_handedness_label(idx)
            gesture = self._classify_hand_gesture(hand_landmarks)
            hand_gestures.append(f"{handedness}:{gesture}")

            if self._draw_hand_landmarks:
                self._drawing_utils.draw_landmarks(
                    annotated,
                    hand_landmarks,
                    self._mp_hands.HAND_CONNECTIONS,
                    self._drawing_styles.get_default_hand_landmarks_style(),
                    self._drawing_styles.get_default_hand_connections_style(),
                )

            if gesture == "point" and pointing_pixel is None:
                pointing_pixel = self._estimate_pointing_pixel(
                    hand_landmarks,
                    annotated.shape[1],
                    annotated.shape[0],
                )
                if pointing_pixel is not None:
                    tip_pixel = self._landmark_to_pixel(
                        hand_landmarks.landmark[8],
                        annotated.shape[1],
                        annotated.shape[0],
                    )
                    cv2.arrowedLine(
                        annotated,
                        tip_pixel,
                        pointing_pixel,
                        (0, 255, 255),
                        2,
                        tipLength=0.15,
                    )
                    cv2.circle(annotated, pointing_pixel, 6, (0, 255, 255), -1)

        gesture_msg = String()
        gesture_msg.data = ",".join(hand_gestures) if hand_gestures else "none"
        self._safe_publish(self._gesture_pub, gesture_msg)
        cv2.putText(
            annotated,
            f"Gesture: {gesture_msg.data}",
            (12, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        if pointing_pixel is not None:
            point_msg = PointStamped()
            point_msg.header = msg.header
            point_msg.point.x = float(pointing_pixel[0])
            point_msg.point.y = float(pointing_pixel[1])
            point_msg.point.z = 0.0
            self._safe_publish(self._pointing_pub, point_msg)
            cv2.putText(
                annotated,
                f"Point: ({pointing_pixel[0]}, {pointing_pixel[1]})",
                (12, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        if self._annotated_pub is not None:
            self._safe_publish(self._annotated_pub, self._bgr_to_ros_image(annotated, msg))

        if self._show_image:
            cv2.imshow(self._window_name, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                self.get_logger().info("Exit key pressed; shutting down.")
                rclpy.shutdown()

    def _ros_image_to_bgr(self, msg: Image) -> np.ndarray | None:
        encoding = msg.encoding.lower()
        channels = self.SUPPORTED_ENCODINGS.get(encoding)
        if channels is None:
            if encoding not in self._warned_encodings:
                self.get_logger().warning(
                    f"Unsupported encoding '{msg.encoding}'. "
                    f"Supported: {sorted(self.SUPPORTED_ENCODINGS.keys())}"
                )
                self._warned_encodings.add(encoding)
            return None

        expected_size = msg.step * msg.height
        if expected_size == 0 or len(msg.data) < expected_size:
            self.get_logger().warning("Invalid image payload size; skipping frame.")
            return None

        row_pixels = msg.width * channels
        if msg.step < row_pixels:
            self.get_logger().warning(
                f"Invalid image step ({msg.step}) for width ({msg.width}) and encoding ({msg.encoding})."
            )
            return None

        raw = np.frombuffer(msg.data, dtype=np.uint8, count=expected_size)
        rows = raw.reshape((msg.height, msg.step))
        image = rows[:, :row_pixels]

        if channels == 1:
            image = image.reshape((msg.height, msg.width))
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        image = image.reshape((msg.height, msg.width, channels))

        if encoding == "bgr8":
            return image
        if encoding == "rgb8":
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if encoding == "bgra8":
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        if encoding == "rgba8":
            return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        return None

    @staticmethod
    def _bgr_to_ros_image(image_bgr: np.ndarray, source: Image) -> Image:
        out = Image()
        out.header = source.header
        out.height = int(image_bgr.shape[0])
        out.width = int(image_bgr.shape[1])
        out.encoding = "bgr8"
        out.is_bigendian = 0
        out.step = out.width * 3
        out.data = image_bgr.tobytes()
        return out

    def _get_handedness_label(self, idx: int) -> str:
        if idx < len(self._last_handedness):
            classes = self._last_handedness[idx].classification
            if classes:
                return classes[0].label.lower()
        return f"hand{idx}"

    @staticmethod
    def _is_extended(tip: object, pip: object) -> bool:
        return float(tip.y) < (float(pip.y) - 0.02)

    def _classify_hand_gesture(self, hand_landmarks: object) -> str:
        lm = hand_landmarks.landmark
        thumb_extended = abs(float(lm[4].x) - float(lm[2].x)) > 0.04
        index_extended = self._is_extended(lm[8], lm[6])
        middle_extended = self._is_extended(lm[12], lm[10])
        ring_extended = self._is_extended(lm[16], lm[14])
        pinky_extended = self._is_extended(lm[20], lm[18])

        if index_extended and not middle_extended and not ring_extended and not pinky_extended:
            return "point"
        if index_extended and middle_extended and ring_extended and pinky_extended:
            return "open_palm"
        if not index_extended and not middle_extended and not ring_extended and not pinky_extended and not thumb_extended:
            return "fist"
        return "other"

    @staticmethod
    def _landmark_to_pixel(landmark: object, width: int, height: int) -> tuple[int, int]:
        x = int(np.clip(round(float(landmark.x) * width), 0, width - 1))
        y = int(np.clip(round(float(landmark.y) * height), 0, height - 1))
        return x, y

    def _estimate_pointing_pixel(
        self,
        hand_landmarks: object,
        width: int,
        height: int,
    ) -> tuple[int, int] | None:
        lm = hand_landmarks.landmark
        tip = np.array(self._landmark_to_pixel(lm[8], width, height), dtype=np.float32)
        pip = np.array(self._landmark_to_pixel(lm[6], width, height), dtype=np.float32)
        direction = tip - pip
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            return None
        direction /= norm

        candidates: list[float] = []
        if abs(float(direction[0])) > 1e-6:
            for bound_x in (0.0, float(width - 1)):
                t = (bound_x - float(tip[0])) / float(direction[0])
                if t > 0:
                    y = float(tip[1]) + t * float(direction[1])
                    if 0.0 <= y <= float(height - 1):
                        candidates.append(t)
        if abs(float(direction[1])) > 1e-6:
            for bound_y in (0.0, float(height - 1)):
                t = (bound_y - float(tip[1])) / float(direction[1])
                if t > 0:
                    x = float(tip[0]) + t * float(direction[0])
                    if 0.0 <= x <= float(width - 1):
                        candidates.append(t)

        if not candidates:
            far_point = tip + direction * max(width, height)
            x = int(np.clip(round(float(far_point[0])), 0, width - 1))
            y = int(np.clip(round(float(far_point[1])), 0, height - 1))
            return x, y

        t_min = min(candidates)
        point = tip + direction * t_min
        x = int(np.clip(round(float(point[0])), 0, width - 1))
        y = int(np.clip(round(float(point[1])), 0, height - 1))
        return x, y

    @staticmethod
    def _safe_publish(publisher: object, msg: object) -> None:
        if not rclpy.ok():
            return
        try:
            publisher.publish(msg)
        except Exception:
            # Can happen during shutdown when ROS context becomes invalid.
            return

    def destroy_node(self) -> bool:
        if self._pose is not None:
            self._pose.close()
        if self._hands is not None:
            self._hands.close()
        if self._show_image:
            cv2.destroyAllWindows()
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = RealsenseMediapipePoseNode()
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

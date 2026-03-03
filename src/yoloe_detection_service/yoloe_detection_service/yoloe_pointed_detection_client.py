#!/usr/bin/env python3
"""Client for one-shot pointed-object detection service."""

from __future__ import annotations

import argparse

import rclpy
from rclpy.node import Node

from yoloe_detection_interfaces.srv import DetectObjectPrompt


class YoloePointedDetectionClient(Node):
    def __init__(self, service_name: str) -> None:
        super().__init__("yoloe_pointed_detection_client")
        self._client = self.create_client(DetectObjectPrompt, service_name)
        while not self._client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f"Waiting for service {service_name} ...")

    def call(self, prompt_text: str, save_image: bool) -> DetectObjectPrompt.Response:
        request = DetectObjectPrompt.Request()
        request.prompt_text = prompt_text
        request.save_image = save_image

        future = self._client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Call pointed-object detection service")
    parser.add_argument("prompt", help="Prompt text, e.g. 'bottle' or 'cup,bottle'")
    parser.add_argument(
        "--service",
        default="/yoloe/detect_pointed_prompt",
        help="Service name",
    )
    parser.add_argument(
        "--no-save-image",
        action="store_true",
        help="Disable image saving for this request (unless server always_save_image=true)",
    )
    return parser.parse_args()


def main(args: list[str] | None = None) -> None:
    cli_args = parse_args()

    rclpy.init(args=args)
    node = YoloePointedDetectionClient(cli_args.service)
    try:
        response = node.call(cli_args.prompt, save_image=not cli_args.no_save_image)
        print("success:", response.success)
        print("message:", response.message)
        print("detections_in_frame:", response.detections_in_frame)
        print("tf_published_count:", response.tf_published_count)
        if response.poses_camera_link:
            pose = response.poses_camera_link[0].pose.position
            cls = response.detected_classes[0] if response.detected_classes else "unknown"
            conf = response.confidences[0] if response.confidences else 0.0
            print(
                f"selected: {cls} conf={conf:.3f} "
                f"centroid=[{pose.x:.3f}, {pose.y:.3f}, {pose.z:.3f}] "
                f"frame={response.poses_camera_link[0].header.frame_id}"
            )
        print("saved_image_path:", response.saved_image_path)
        print("inference_ms:", response.inference_ms)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

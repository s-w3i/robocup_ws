#!/usr/bin/env python3
"""Simple client for the YOLOE prompt detection service."""

from __future__ import annotations

import argparse

import rclpy
from rclpy.node import Node

from yoloe_detection_interfaces.srv import DetectObjectPrompt


class YoloeDetectionClient(Node):
    def __init__(self, service_name: str) -> None:
        super().__init__("yoloe_detection_client")
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
    parser = argparse.ArgumentParser(description="Call YOLOE prompt detection service")
    parser.add_argument("prompt", help="Prompt text, e.g. 'bottle' or 'cup,bottle'")
    parser.add_argument("--service", default="/yoloe/detect_prompt", help="Service name")
    parser.add_argument(
        "--no-save-image",
        action="store_true",
        help="Disable image saving for this request (unless server always_save_image=true)",
    )
    return parser.parse_args()


def main(args: list[str] | None = None) -> None:
    cli_args = parse_args()

    rclpy.init(args=args)
    node = YoloeDetectionClient(cli_args.service)
    try:
        response = node.call(cli_args.prompt, save_image=not cli_args.no_save_image)
        print("success:", response.success)
        print("message:", response.message)
        print("detections_in_frame:", response.detections_in_frame)
        print("tf_published_count:", response.tf_published_count)
        for idx, (name, conf, pose_msg, frame) in enumerate(
            zip(
                response.detected_classes,
                response.confidences,
                response.poses_camera_link,
                response.tf_child_frames,
            ),
            start=1,
        ):
            print(
                f"[{idx}] class={name} conf={conf:.3f} tf={frame} "
                f"pos=[{pose_msg.pose.position.x:.3f}, {pose_msg.pose.position.y:.3f}, {pose_msg.pose.position.z:.3f}]"
            )
        print("saved_image_path:", response.saved_image_path)
        print("inference_ms:", response.inference_ms)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

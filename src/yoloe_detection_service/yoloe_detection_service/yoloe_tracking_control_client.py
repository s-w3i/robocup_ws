#!/usr/bin/env python3
"""Client to start/stop tracking service."""

from __future__ import annotations

import argparse

import rclpy
from rclpy.node import Node

from yoloe_detection_interfaces.srv import SetTracking


class YoloeTrackingControlClient(Node):
    def __init__(self, service_name: str) -> None:
        super().__init__("yoloe_tracking_control_client")
        self._client = self.create_client(SetTracking, service_name)
        while not self._client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f"Waiting for service {service_name} ...")

    def call(
        self,
        enable: bool,
        save_image: bool,
        rate_hz: float,
    ) -> SetTracking.Response:
        request = SetTracking.Request()
        request.enable = enable
        request.save_image = save_image
        request.rate_hz = float(rate_hz)

        future = self._client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        return future.result()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Control tracking service")
    parser.add_argument("mode", choices=["start", "stop"], help="Start or stop tracking mode")
    parser.add_argument(
        "prompt",
        nargs="?",
        default="",
        help="Deprecated and ignored. Tracking is human-only.",
    )
    parser.add_argument(
        "--service",
        default="/yoloe/set_tracking",
        help="Tracking control service name",
    )
    parser.add_argument("--rate-hz", type=float, default=2.0, help="Tracking rate")
    parser.add_argument(
        "--save-image",
        action="store_true",
        help="Save image on each tracking iteration",
    )
    return parser.parse_args()


def main(args: list[str] | None = None) -> None:
    cli_args = parse_args()

    enable = cli_args.mode == "start"

    rclpy.init(args=args)
    node = YoloeTrackingControlClient(cli_args.service)
    try:
        if cli_args.prompt:
            node.get_logger().warn(
                f"Ignoring deprecated prompt argument '{cli_args.prompt}'. "
                "Tracking is fixed to class 'person'."
            )
        response = node.call(
            enable=enable,
            save_image=cli_args.save_image,
            rate_hz=cli_args.rate_hz,
        )
        print("success:", response.success)
        print("message:", response.message)
        print("running:", response.running)
        print("tracking_class:", response.tracking_class)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Mock lost-found location image service for Telegram/VLM testing."""

from __future__ import annotations

from pathlib import Path

import rclpy
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node

from vlm_interfaces.srv import LocationImage


class MockLocationImageServiceNode(Node):
    def __init__(self) -> None:
        super().__init__("mock_location_image_service_node")

        self.declare_parameter("service_name", "/lost_found/location_image")
        self.declare_parameter("image_path", "")
        self.declare_parameter("camera_name", "camera0")
        self.declare_parameter("success", True)

        self.service_name = str(self.get_parameter("service_name").value).strip() or "/lost_found/location_image"
        self.image_path = str(self.get_parameter("image_path").value).strip()
        self.camera_name = str(self.get_parameter("camera_name").value).strip() or "camera0"
        self.success = bool(self.get_parameter("success").value)

        self.create_service(LocationImage, self.service_name, self._handle_request)
        self.get_logger().info(
            f"Mock location image service ready on {self.service_name} | "
            f"camera={self.camera_name} | image_path={self.image_path or '<camera fallback>'}"
        )

    def _handle_request(
        self,
        request: LocationImage.Request,
        response: LocationImage.Response,
    ) -> LocationImage.Response:
        response.success = self.success
        response.camera_name = self.camera_name
        response.image_path = self.image_path
        if self.success:
            location = str(request.location_name).strip() or "unknown"
            if self.image_path and not Path(self.image_path).expanduser().is_file():
                response.success = False
                response.message = f"Configured mock image path does not exist: {self.image_path}"
            else:
                response.message = f"Mock image ready for {location}."
        else:
            response.message = "Mock location image service configured to fail."
        return response


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = MockLocationImageServiceNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()

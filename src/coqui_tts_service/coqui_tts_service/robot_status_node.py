#!/usr/bin/env python3

from __future__ import annotations

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import Bool, String

from coqui_tts_interfaces.srv import RobotStatus


VALID_STATUSES = ("sleep", "listening", "idle", "operating")
TOGGLE_COMMANDS = ("", "toggle", "next", "cycle")


class RobotStatusNode(Node):
    def __init__(self) -> None:
        super().__init__("robot_status_node")

        self.declare_parameter("status_topic", "/robot_status")
        self.declare_parameter("awake_topic", "/awake")
        self.declare_parameter("status_service", "/robot_status")
        self.declare_parameter("initial_status", "sleep")

        self.status_topic = str(self.get_parameter("status_topic").value)
        self.awake_topic = str(self.get_parameter("awake_topic").value)
        self.status_service = str(self.get_parameter("status_service").value)
        initial_status = str(self.get_parameter("initial_status").value).strip().lower()
        if initial_status not in VALID_STATUSES:
            self.get_logger().warn(
                f"Invalid initial_status '{initial_status}', falling back to 'sleep'."
            )
            initial_status = "sleep"
        self._status = initial_status

        qos = QoSProfile(depth=1)
        qos.reliability = QoSReliabilityPolicy.RELIABLE
        qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL
        self._status_pub = self.create_publisher(String, self.status_topic, qos)
        self._awake_pub = self.create_publisher(Bool, self.awake_topic, qos)
        self.create_service(RobotStatus, self.status_service, self._handle_robot_status)

        self._publish_status()
        if self._status == "sleep":
            self._publish_awake(False)
        self.get_logger().info(
            f"Robot status service ready on {self.status_service} | topic={self.status_topic} | awake={self.awake_topic}"
        )
        self.get_logger().info(
            "Valid statuses: sleep, listening, idle, operating "
            "(send empty string or 'toggle'/'next'/'cycle' to advance)."
        )
        self.get_logger().info(f"Current robot status: {self._status}")

    def _next_status(self) -> str:
        idx = VALID_STATUSES.index(self._status)
        return VALID_STATUSES[(idx + 1) % len(VALID_STATUSES)]

    def _publish_status(self) -> None:
        msg = String()
        msg.data = self._status
        self._status_pub.publish(msg)

    def _publish_awake(self, value: bool) -> None:
        msg = Bool()
        msg.data = bool(value)
        self._awake_pub.publish(msg)

    def _handle_robot_status(
        self, request: RobotStatus.Request, response: RobotStatus.Response
    ) -> RobotStatus.Response:
        requested = str(request.status).strip().lower()

        if requested in TOGGLE_COMMANDS:
            target = self._next_status()
        elif requested in VALID_STATUSES:
            target = requested
        else:
            response.success = False
            response.status = self._status
            response.message = (
                f"Invalid status '{requested}'. Use one of: {', '.join(VALID_STATUSES)} "
                "or empty/'toggle' to cycle."
            )
            return response

        changed = target != self._status
        self._status = target
        self._publish_status()
        if self._status == "sleep":
            self._publish_awake(False)

        response.success = True
        response.status = self._status
        if changed:
            response.message = f"Robot status updated to '{self._status}'."
            self.get_logger().info(response.message)
        else:
            response.message = f"Robot status already '{self._status}'."
        return response


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RobotStatusNode()
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

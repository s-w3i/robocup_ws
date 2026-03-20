#!/usr/bin/env python3
"""Live monitor for behaviour tree snapshots published by BT executor nodes."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String


@dataclass
class TreeSnapshot:
    tree_id: str
    node_name: str
    event: str
    tick_count: int
    timestamp_sec: float
    tree_text: str
    dot_text: str
    summary: dict


class BtMonitorNode(Node):
    def __init__(self) -> None:
        super().__init__("bt_monitor_node")

        self.declare_parameter("snapshot_topic", "/bt_tree/snapshots")
        self.declare_parameter("refresh_hz", 2.0)
        self.declare_parameter("stale_timeout_sec", 10.0)

        self.snapshot_topic = (
            str(self.get_parameter("snapshot_topic").value).strip() or "/bt_tree/snapshots"
        )
        self.refresh_hz = max(0.2, float(self.get_parameter("refresh_hz").value))
        self.stale_timeout_sec = max(1.0, float(self.get_parameter("stale_timeout_sec").value))

        self._snapshots: dict[str, TreeSnapshot] = {}
        self._stdout_is_tty = sys.stdout.isatty()

        self.create_subscription(String, self.snapshot_topic, self._on_snapshot, 50)
        self.create_timer(1.0 / self.refresh_hz, self._render)

        self.get_logger().info(
            f"BT monitor listening on {self.snapshot_topic} at {self.refresh_hz:.1f} Hz"
        )

    def _on_snapshot(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except Exception as exc:
            self.get_logger().warn(f"Failed to parse BT snapshot: {exc}")
            return

        tree_id = str(payload.get("tree_id", "") or "unknown_tree")
        snapshot = TreeSnapshot(
            tree_id=tree_id,
            node_name=str(payload.get("node_name", "") or ""),
            event=str(payload.get("event", "") or ""),
            tick_count=int(payload.get("tick_count", 0) or 0),
            timestamp_sec=float(payload.get("timestamp_sec", time.time()) or time.time()),
            tree_text=str(payload.get("tree_text", "") or ""),
            dot_text=str(payload.get("dot_text", "") or ""),
            summary=dict(payload.get("summary", {}) or {}),
        )
        self._snapshots[tree_id] = snapshot

    def _render(self) -> None:
        now = time.time()
        active = [
            snapshot
            for snapshot in self._snapshots.values()
            if now - snapshot.timestamp_sec <= self.stale_timeout_sec
        ]
        active.sort(key=lambda snapshot: snapshot.tree_id)

        lines = [
            f"BT Monitor | active_trees={len(active)} | topic={self.snapshot_topic} | time={time.strftime('%H:%M:%S')}",
            "",
        ]
        if not active:
            lines.append("No recent BT snapshots.")
        else:
            for snapshot in active:
                summary = snapshot.summary
                lines.extend(
                    [
                        f"Tree: {snapshot.tree_id}",
                        f"Node: {snapshot.node_name} | event={snapshot.event} | tick={snapshot.tick_count}",
                        (
                            "Summary: "
                            f"completed={summary.get('completed', False)} "
                            f"attempt={summary.get('attempt_count', 0)} "
                            f"name={summary.get('name', '')!r} "
                            f"drink={summary.get('drink', '')!r} "
                            f"error={summary.get('last_error', '')!r}"
                        ),
                        f"Robot: {str(summary.get('robot_text', '') or '')}",
                        f"User: {str(summary.get('last_user_input', '') or '')}",
                        "Tree State:",
                        snapshot.tree_text.rstrip(),
                        "-" * 72,
                    ]
                )

        rendered = "\n".join(lines) + "\n"
        if self._stdout_is_tty:
            sys.stdout.write("\033[2J\033[H")
            sys.stdout.write(rendered)
            sys.stdout.flush()
        else:
            self.get_logger().info(rendered)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = BtMonitorNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()

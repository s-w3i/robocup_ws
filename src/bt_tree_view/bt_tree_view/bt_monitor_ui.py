#!/usr/bin/env python3
"""PyQt5 UI for live behaviour tree snapshots."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass

import pydot
from PyQt5 import QtCore, QtGui, QtWidgets

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


class BtSnapshotClient(Node):
    def __init__(self) -> None:
        super().__init__("bt_monitor_ui")

        self.declare_parameter("snapshot_topic", "/bt_tree/snapshots")
        self.declare_parameter("stale_timeout_sec", 15.0)

        self.snapshot_topic = (
            str(self.get_parameter("snapshot_topic").value).strip() or "/bt_tree/snapshots"
        )
        self.stale_timeout_sec = max(1.0, float(self.get_parameter("stale_timeout_sec").value))
        self.snapshots: dict[str, TreeSnapshot] = {}

        self.create_subscription(String, self.snapshot_topic, self._on_snapshot, 50)
        self.get_logger().info(f"BT monitor UI listening on {self.snapshot_topic}")

    def _on_snapshot(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except Exception as exc:
            self.get_logger().warn(f"Failed to parse BT snapshot: {exc}")
            return

        tree_id = str(payload.get("tree_id", "") or "unknown_tree")
        self.snapshots[tree_id] = TreeSnapshot(
            tree_id=tree_id,
            node_name=str(payload.get("node_name", "") or ""),
            event=str(payload.get("event", "") or ""),
            tick_count=int(payload.get("tick_count", 0) or 0),
            timestamp_sec=float(payload.get("timestamp_sec", time.time()) or time.time()),
            tree_text=str(payload.get("tree_text", "") or ""),
            dot_text=str(payload.get("dot_text", "") or ""),
            summary=dict(payload.get("summary", {}) or {}),
        )

    def active_snapshots(self) -> list[TreeSnapshot]:
        now = time.time()
        snapshots = [
            snapshot
            for snapshot in self.snapshots.values()
            if now - snapshot.timestamp_sec <= self.stale_timeout_sec
        ]
        snapshots.sort(key=lambda snapshot: snapshot.tree_id)
        return snapshots


class BtMonitorWindow(QtWidgets.QMainWindow):
    def __init__(self, ros_node: BtSnapshotClient) -> None:
        super().__init__()
        self.ros_node = ros_node
        self._selected_tree_id: str | None = None
        self._last_graph_tree_id: str | None = None
        self._last_graph_dot_text: str = ""

        self.setWindowTitle("BT Tree View")
        self.resize(1400, 860)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root_layout = QtWidgets.QHBoxLayout(central)

        self.tree_list = QtWidgets.QListWidget()
        self.tree_list.setMinimumWidth(320)
        self.tree_list.currentItemChanged.connect(self._on_tree_selected)
        root_layout.addWidget(self.tree_list, 1)

        right_panel = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        root_layout.addWidget(right_panel, 3)

        top_widget = QtWidgets.QWidget()
        top_layout = QtWidgets.QVBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)

        self.header_label = QtWidgets.QLabel("No active trees.")
        self.header_label.setWordWrap(True)
        self.header_label.setStyleSheet("font-weight: 600; font-size: 16px;")
        top_layout.addWidget(self.header_label)

        self.summary_box = QtWidgets.QPlainTextEdit()
        self.summary_box.setReadOnly(True)
        self.summary_box.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.summary_box.setFont(self._mono_font())
        top_layout.addWidget(self.summary_box, 1)

        right_panel.addWidget(top_widget)

        graph_tabs = QtWidgets.QTabWidget()
        right_panel.addWidget(graph_tabs)

        graph_widget = QtWidgets.QWidget()
        graph_layout = QtWidgets.QVBoxLayout(graph_widget)
        graph_layout.setContentsMargins(0, 0, 0, 0)

        graph_toolbar = QtWidgets.QHBoxLayout()
        self.zoom_out_button = QtWidgets.QPushButton("-")
        self.zoom_out_button.setFixedWidth(36)
        self.zoom_out_button.clicked.connect(lambda: self._zoom_graph(0.85))
        graph_toolbar.addWidget(self.zoom_out_button)

        self.zoom_in_button = QtWidgets.QPushButton("+")
        self.zoom_in_button.setFixedWidth(36)
        self.zoom_in_button.clicked.connect(lambda: self._zoom_graph(1.15))
        graph_toolbar.addWidget(self.zoom_in_button)

        self.reset_zoom_button = QtWidgets.QPushButton("Reset")
        self.reset_zoom_button.clicked.connect(self._reset_graph_zoom)
        graph_toolbar.addWidget(self.reset_zoom_button)

        self.zoom_label = QtWidgets.QLabel("100%")
        graph_toolbar.addWidget(self.zoom_label)
        graph_toolbar.addStretch(1)
        graph_layout.addLayout(graph_toolbar)

        self.graph_scene = QtWidgets.QGraphicsScene(self)
        self.graph_view = QtWidgets.QGraphicsView(self.graph_scene)
        self.graph_view.setRenderHints(
            QtGui.QPainter.Antialiasing
            | QtGui.QPainter.SmoothPixmapTransform
            | QtGui.QPainter.TextAntialiasing
        )
        self.graph_view.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        self.graph_view.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.graph_view.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.graph_view.setBackgroundBrush(QtGui.QColor("#f4f6f8"))
        self.graph_pixmap_item = QtWidgets.QGraphicsPixmapItem()
        self.graph_scene.addItem(self.graph_pixmap_item)
        self._graph_scale = 1.0
        graph_layout.addWidget(self.graph_view, 1)

        graph_tabs.addTab(graph_widget, "Graph")

        self.tree_text_box = QtWidgets.QPlainTextEdit()
        self.tree_text_box.setReadOnly(True)
        self.tree_text_box.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self.tree_text_box.setFont(self._mono_font())
        graph_tabs.addTab(self.tree_text_box, "Text")
        right_panel.setSizes([280, 520])

        self.statusBar().showMessage(f"Listening on {self.ros_node.snapshot_topic}")

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._refresh)
        self.timer.start(250)

    @staticmethod
    def _mono_font() -> QtGui.QFont:
        font = QtGui.QFont("Monospace")
        font.setStyleHint(QtGui.QFont.TypeWriter)
        font.setPointSize(10)
        return font

    def _refresh(self) -> None:
        snapshots = self.ros_node.active_snapshots()
        current_ids = [snapshot.tree_id for snapshot in snapshots]

        if self._selected_tree_id not in current_ids:
            self._selected_tree_id = current_ids[0] if current_ids else None

        self.tree_list.blockSignals(True)
        self.tree_list.clear()
        for snapshot in snapshots:
            summary = snapshot.summary
            label = (
                f"{snapshot.tree_id}\n"
                f"  event={snapshot.event} tick={snapshot.tick_count} "
                f"name={summary.get('name', '') or '-'} "
                f"drink={summary.get('drink', '') or '-'} "
                f"flavour={summary.get('food_flavour_preference', '') or '-'} "
                f"allergy={summary.get('food_allergy', '') or '-'}"
            )
            item = QtWidgets.QListWidgetItem(label)
            item.setData(QtCore.Qt.UserRole, snapshot.tree_id)
            self.tree_list.addItem(item)
            if snapshot.tree_id == self._selected_tree_id:
                self.tree_list.setCurrentItem(item)
        self.tree_list.blockSignals(False)

        selected = None
        for snapshot in snapshots:
            if snapshot.tree_id == self._selected_tree_id:
                selected = snapshot
                break
        if selected is None and snapshots:
            selected = snapshots[0]
            self._selected_tree_id = selected.tree_id

        self._render_snapshot(selected)

    def _on_tree_selected(self, current: QtWidgets.QListWidgetItem, previous: QtWidgets.QListWidgetItem) -> None:
        if current is None:
            return
        self._selected_tree_id = str(current.data(QtCore.Qt.UserRole))
        snapshots = {snapshot.tree_id: snapshot for snapshot in self.ros_node.active_snapshots()}
        self._render_snapshot(snapshots.get(self._selected_tree_id))

    def _render_snapshot(self, snapshot: TreeSnapshot | None) -> None:
        if snapshot is None:
            self.header_label.setText("No active trees.")
            self.summary_box.setPlainText("")
            self.tree_text_box.setPlainText("")
            self.graph_scene.clear()
            self.graph_pixmap_item = QtWidgets.QGraphicsPixmapItem()
            self.graph_scene.addItem(self.graph_pixmap_item)
            self._last_graph_tree_id = None
            self._last_graph_dot_text = ""
            return

        age_s = max(0.0, time.time() - snapshot.timestamp_sec)
        self.header_label.setText(
            f"{snapshot.tree_id} | node={snapshot.node_name} | event={snapshot.event} | "
            f"tick={snapshot.tick_count} | age={age_s:.1f}s"
        )
        summary = snapshot.summary
        summary_text = "\n".join(
            [
                f"completed: {summary.get('completed', False)}",
                f"attempt_count: {summary.get('attempt_count', 0)}",
                f"task: {summary.get('task', '')}",
                f"name: {summary.get('name', '')}",
                f"drink: {summary.get('drink', '')}",
                f"food_flavour_preference: {summary.get('food_flavour_preference', '')}",
                f"food_allergy: {summary.get('food_allergy', '')}",
                f"llm_complete: {summary.get('llm_complete', False)}",
                f"llm_continue_conversation: {summary.get('llm_continue_conversation', True)}",
                f"last_user_input: {summary.get('last_user_input', '')}",
                f"robot_text: {summary.get('robot_text', '')}",
                f"last_error: {summary.get('last_error', '')}",
                f"reason: {summary.get('reason', '')}",
            ]
        )
        self.summary_box.setPlainText(summary_text)
        self.tree_text_box.setPlainText(snapshot.tree_text.rstrip())
        tree_changed = snapshot.tree_id != self._last_graph_tree_id
        if tree_changed:
            self._reset_graph_zoom()
        self._render_graph(snapshot.tree_id, snapshot.dot_text, tree_changed)

    def _render_graph(self, tree_id: str, dot_text: str, tree_changed: bool) -> None:
        if tree_id == self._last_graph_tree_id and dot_text == self._last_graph_dot_text:
            return
        if not dot_text.strip():
            self.graph_pixmap_item.setPixmap(QtGui.QPixmap())
            self.graph_scene.setSceneRect(QtCore.QRectF())
            self._last_graph_tree_id = tree_id
            self._last_graph_dot_text = dot_text
            return
        try:
            graphs = pydot.graph_from_dot_data(dot_text)
            if not graphs:
                raise RuntimeError("No graph parsed from dot_text")
            png_bytes = graphs[0].create_png()
            pixmap = QtGui.QPixmap()
            if not pixmap.loadFromData(png_bytes):
                raise RuntimeError("Failed to decode PNG graph bytes")
            self.graph_pixmap_item.setPixmap(pixmap)
            self.graph_scene.setSceneRect(QtCore.QRectF(pixmap.rect()))
        except Exception as exc:
            error_pixmap = QtGui.QPixmap(900, 120)
            error_pixmap.fill(QtGui.QColor("#fff6f6"))
            painter = QtGui.QPainter(error_pixmap)
            painter.setPen(QtGui.QPen(QtGui.QColor("#992222")))
            painter.setFont(QtGui.QFont("Sans Serif", 12))
            painter.drawText(20, 40, "Graph render failed")
            painter.setFont(QtGui.QFont("Sans Serif", 10))
            painter.drawText(20, 75, str(exc))
            painter.end()
            self.graph_pixmap_item.setPixmap(error_pixmap)
            self.graph_scene.setSceneRect(QtCore.QRectF(error_pixmap.rect()))
        self._last_graph_tree_id = tree_id
        self._last_graph_dot_text = dot_text

    def _zoom_graph(self, factor: float) -> None:
        new_scale = max(0.2, min(5.0, self._graph_scale * factor))
        factor = new_scale / self._graph_scale
        self._graph_scale = new_scale
        self.graph_view.scale(factor, factor)
        self.zoom_label.setText(f"{int(self._graph_scale * 100)}%")

    def _reset_graph_zoom(self) -> None:
        self.graph_view.resetTransform()
        self._graph_scale = 1.0
        self.zoom_label.setText("100%")


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    ros_node = BtSnapshotClient()

    app = QtWidgets.QApplication(sys.argv)
    window = BtMonitorWindow(ros_node)
    window.show()

    ros_timer = QtCore.QTimer()

    def _spin_ros_once() -> None:
        if not rclpy.ok():
            ros_timer.stop()
            return
        try:
            rclpy.spin_once(ros_node, timeout_sec=0.0)
        except Exception:
            ros_timer.stop()

    ros_timer.timeout.connect(_spin_ros_once)
    ros_timer.start(50)

    try:
        app.exec_()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        ros_timer.stop()
        window.close()
        ros_node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()

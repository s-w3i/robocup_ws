#!/usr/bin/env python3
# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Joy, LaserScan
from simple_follower_ros2.msg import FollowerPosition as PositionMsg
from std_msgs.msg import Bool, String as StringMsg
from visualization_msgs.msg import Marker
from yoloe_detection_interfaces.srv import DetectObjectPrompt


class LaserTracker(Node):

	def __init__(self):
		super().__init__('lasertracker')
		qos = QoSProfile(depth=10)
		self.enabled = self.declare_parameter('enabled', False).value
		self.target_radius = self.declare_parameter('target_radius', 0.35).value
		self.acquire_max_distance = self.declare_parameter('acquire_max_distance', 3.0).value
		self.lost_timeout = self.declare_parameter('lost_timeout', 1.5).value
		self.min_cluster_points = self.declare_parameter('min_cluster_points', 3).value
		self.max_target_jump = self.declare_parameter('max_target_jump', 0.25).value
		self.target_smoothing_alpha = self.declare_parameter('target_smoothing_alpha', 0.25).value
		self.cluster_break_distance = self.declare_parameter('cluster_break_distance', 0.12).value
		self.min_target_width = self.declare_parameter('min_target_width', 0.06).value
		self.max_target_width = self.declare_parameter('max_target_width', 0.65).value
		self.max_cluster_depth = self.declare_parameter('max_cluster_depth', 0.35).value
		self.max_linearity_ratio = self.declare_parameter('max_linearity_ratio', 12.0).value
		self.vision_validation_enabled = self.declare_parameter('vision_validation_enabled', False).value
		self.vision_service_name = self.declare_parameter('vision_service_name', '/yoloe/detect_prompt').value
		self.vision_prompt = self.declare_parameter('vision_prompt', 'person').value
		self.vision_camera_name = self.declare_parameter('vision_camera_name', 'camera0').value
		self.vision_period = self.declare_parameter('vision_period', 1.0).value
		self.vision_request_timeout = self.declare_parameter('vision_request_timeout', 5.0).value
		self.vision_valid_timeout = self.declare_parameter('vision_valid_timeout', 3.0).value
		self.vision_min_confidence = self.declare_parameter('vision_min_confidence', 0.25).value
		self.require_vision_for_acquire = self.declare_parameter('require_vision_for_acquire', True).value
		self.target_x = None
		self.target_y = None
		self.last_target_time = None
		self.last_vision_person_time = None
		self.vision_future = None
		self.vision_request_time = None
		self.vision_warned_unavailable = False
		self.last_info_times = {}
		self.positionPublisher = self.create_publisher(PositionMsg, 'object_tracker/current_position', qos)
		self.infoPublisher = self.create_publisher(StringMsg, 'object_tracker/info', qos)
		self.markerPublisher = self.create_publisher(Marker, 'object_tracker/marker', qos)
		self.enableSubscriber = self.create_subscription(
		    Bool,
		    '/laser_follow_enabled',
		    self.enableCallback,
		    qos)
		self.scanSubscriber = self.create_subscription(
		    LaserScan,
		    '/scan',
		    self.registerScan,
		    qos)
		self.visionClient = None
		self.visionTimer = None
		if self.vision_validation_enabled:
			self.visionClient = self.create_client(DetectObjectPrompt, self.vision_service_name)
			timer_period = max(0.2, float(self.vision_period))
			self.visionTimer = self.create_timer(timer_period, self.requestVisionValidation)
			self.get_logger().info(
				f'vision validation enabled: service={self.vision_service_name}, '
				f'camera={self.vision_camera_name}, prompt="{self.vision_prompt}"'
			)

	def enableCallback(self, msg):
		previous = self.enabled
		self.enabled = msg.data
		if previous != self.enabled:
			state = 'enabled' if self.enabled else 'disabled'
			self.get_logger().info(f'laser tracking {state}')
		self.resetTarget()
		if not self.enabled:
			self.clearMarker()

	def resetTarget(self):
		self.target_x = None
		self.target_y = None
		self.last_target_time = None

	def requestVisionValidation(self):
		if not self.enabled or not self.vision_validation_enabled:
			return
		if self.visionClient is None:
			return
		if self.vision_future is not None and not self.vision_future.done():
			if self.vision_request_time is not None:
				elapsed = (self.get_clock().now() - self.vision_request_time).nanoseconds / 1e9
				if elapsed > float(self.vision_request_timeout):
					self.publishInfo('vision:detection request timed out', warn=True)
					self.vision_future = None
					self.vision_request_time = None
				else:
					return
			else:
				return
		if not self.visionClient.service_is_ready():
			if not self.vision_warned_unavailable:
				self.publishInfo(
					f'vision:waiting for YOLO service {self.vision_service_name}',
					warn=True,
				)
				self.vision_warned_unavailable = True
			return

		request = DetectObjectPrompt.Request()
		request.prompt_text = str(self.vision_prompt)
		request.save_image = False
		request.camera_name = str(self.vision_camera_name)
		self.vision_future = self.visionClient.call_async(request)
		self.vision_request_time = self.get_clock().now()
		self.vision_future.add_done_callback(self.handleVisionResponse)

	def handleVisionResponse(self, future):
		try:
			response = future.result()
		except Exception as exc:
			self.publishInfo(f'vision:detection request failed: {exc}', warn=True)
			return
		finally:
			self.vision_future = None
			self.vision_request_time = None

		if not response.success:
			self.publishInfo(f'vision:no valid person detection: {response.message}', warn=False)
			return

		person_seen = False
		for class_name, confidence in zip(response.detected_classes, response.confidences):
			if confidence < self.vision_min_confidence:
				continue
			if 'person' in class_name.lower() or 'human' in class_name.lower():
				person_seen = True
				break

		if person_seen:
			self.last_vision_person_time = self.get_clock().now()
			self.vision_warned_unavailable = False
			self.publishInfo('vision:person confirmed by camera0', warn=False)
		else:
			self.publishInfo('vision:no person class in YOLO result', warn=False)

	def visionRecentlySawPerson(self):
		if not self.vision_validation_enabled or not self.require_vision_for_acquire:
			return True
		if self.last_vision_person_time is None:
			return False
		elapsed = (self.get_clock().now() - self.last_vision_person_time).nanoseconds / 1e9
		return elapsed <= float(self.vision_valid_timeout)

	def clearMarker(self, frame_id='base_scan'):
		marker = Marker()
		marker.header.frame_id = frame_id
		marker.header.stamp = self.get_clock().now().to_msg()
		marker.ns = 'laser_tracker'
		marker.id = 0
		marker.action = Marker.DELETE
		self.markerPublisher.publish(marker)

	def publishMarker(self, scan_data, angle, distance):
		marker = Marker()
		marker.header.frame_id = scan_data.header.frame_id or 'base_scan'
		marker.header.stamp = self.get_clock().now().to_msg()
		marker.ns = 'laser_tracker'
		marker.id = 0
		marker.type = Marker.SPHERE
		marker.action = Marker.ADD
		marker.pose.position.x = float(distance * np.cos(angle))
		marker.pose.position.y = float(distance * np.sin(angle))
		marker.pose.position.z = 0.0
		marker.pose.orientation.w = 1.0
		marker.scale.x = 0.20
		marker.scale.y = 0.20
		marker.scale.z = 0.20
		marker.color.a = 1.0
		marker.color.r = 1.0
		marker.color.g = 0.2
		marker.color.b = 0.1
		self.markerPublisher.publish(marker)

	def publishPosition(self, scan_data, target_x, target_y):
		msgdata = PositionMsg()
		angle = float(np.arctan2(target_y, target_x))
		distance = float(np.hypot(target_x, target_y))
		msgdata.angle_x = angle
		msgdata.angle_y = 0.0
		msgdata.distance = distance
		self.positionPublisher.publish(msgdata)
		self.publishMarker(scan_data, angle, distance)

	def publishInfo(self, text, warn=True, throttle_sec=1.0):
		now = self.get_clock().now()
		last_time = self.last_info_times.get(text)
		if last_time is not None:
			elapsed = (now - last_time).nanoseconds / 1e9
			if elapsed < throttle_sec:
				return
		self.last_info_times[text] = now
		msg = StringMsg()
		msg.data = text
		if warn:
			self.get_logger().warn(text)
		else:
			self.get_logger().info(text)
		self.infoPublisher.publish(msg)

	def scanToPoints(self, scan_data):
		points = []
		for index, scan_range in enumerate(scan_data.ranges):
			if np.isinf(scan_range) or np.isnan(scan_range):
				continue
			if scan_range <= scan_data.range_min or scan_range >= scan_data.range_max:
				continue
			if scan_range > self.acquire_max_distance:
				continue

			angle = scan_data.angle_min + index * scan_data.angle_increment
			points.append((scan_range * np.cos(angle), scan_range * np.sin(angle), scan_range, index))
		return points

	def buildClusters(self, points):
		if not points:
			return []

		clusters = []
		current_cluster = [points[0]]
		for point in points[1:]:
			prev_x, prev_y, _, prev_index = current_cluster[-1]
			point_x, point_y, _, point_index = point
			point_gap = np.hypot(point_x - prev_x, point_y - prev_y)
			if point_index == prev_index + 1 and point_gap <= self.cluster_break_distance:
				current_cluster.append(point)
			else:
				clusters.append(current_cluster)
				current_cluster = [point]
		clusters.append(current_cluster)
		return clusters

	def clusterShape(self, cluster):
		xy = np.array([(point[0], point[1]) for point in cluster])
		centroid = np.mean(xy, axis=0)
		centered = xy - centroid
		if len(cluster) >= 2:
			_, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
			length = float(2.0 * singular_values[0] / np.sqrt(len(cluster)))
			depth = float(2.0 * singular_values[-1] / np.sqrt(len(cluster))) if len(singular_values) > 1 else 0.0
		else:
			length = 0.0
			depth = 0.0
		width = float(np.max(xy[:, 1]) - np.min(xy[:, 1]))
		range_depth = float(np.max(xy[:, 0]) - np.min(xy[:, 0]))
		extent = float(max(length, width, range_depth))
		linearity_ratio = length / max(depth, 1e-3)
		distance = float(np.hypot(centroid[0], centroid[1]))
		return {
			'centroid_x': float(centroid[0]),
			'centroid_y': float(centroid[1]),
			'distance': distance,
			'extent': extent,
			'depth': max(depth, range_depth),
			'linearity_ratio': linearity_ratio,
			'point_count': len(cluster),
		}

	def isHumanLikeCluster(self, cluster):
		if len(cluster) < self.min_cluster_points:
			return False, None

		shape = self.clusterShape(cluster)
		if shape['extent'] < self.min_target_width:
			return False, shape
		if shape['extent'] > self.max_target_width:
			return False, shape
		if shape['depth'] > self.max_cluster_depth:
			return False, shape
		if shape['linearity_ratio'] > self.max_linearity_ratio and shape['extent'] > 0.25:
			return False, shape
		return True, shape

	def findHumanLikeClusters(self, points):
		valid_clusters = []
		for cluster in self.buildClusters(points):
			is_valid, shape = self.isHumanLikeCluster(cluster)
			if is_valid:
				valid_clusters.append(shape)
		return valid_clusters

	def acquireTarget(self, scan_data, points):
		if not self.visionRecentlySawPerson():
			self.publishInfo('laser:waiting for camera0 person confirmation', warn=False)
			return False

		valid_clusters = self.findHumanLikeClusters(points)
		if not valid_clusters:
			return False

		target = min(valid_clusters, key=lambda cluster: cluster['distance'])
		self.target_x = target['centroid_x']
		self.target_y = target['centroid_y']
		self.last_target_time = self.get_clock().now()
		self.publishPosition(scan_data, self.target_x, self.target_y)
		self.publishInfo('laser:target acquired', warn=False)
		return True

	def updateLockedTarget(self, scan_data, points):
		valid_clusters = self.findHumanLikeClusters(points)
		nearby_clusters = []
		for cluster in valid_clusters:
			distance_to_target = np.hypot(
				cluster['centroid_x'] - self.target_x,
				cluster['centroid_y'] - self.target_y,
			)
			if distance_to_target <= self.target_radius:
				nearby_clusters.append((distance_to_target, cluster))

		if not nearby_clusters:
			return False

		_, target_cluster = min(nearby_clusters, key=lambda item: item[0])
		centroid_x = target_cluster['centroid_x']
		centroid_y = target_cluster['centroid_y']
		target_jump = np.hypot(centroid_x - self.target_x, centroid_y - self.target_y)
		if target_jump > self.max_target_jump:
			self.publishInfo('laser:target jump rejected')
			return False

		alpha = float(np.clip(self.target_smoothing_alpha, 0.0, 1.0))
		self.target_x = (1.0 - alpha) * self.target_x + alpha * centroid_x
		self.target_y = (1.0 - alpha) * self.target_y + alpha * centroid_y
		self.last_target_time = self.get_clock().now()
		self.publishPosition(scan_data, self.target_x, self.target_y)
		return True

	def targetLostTooLong(self):
		if self.last_target_time is None:
			return True
		elapsed = (self.get_clock().now() - self.last_target_time).nanoseconds / 1e9
		return elapsed > self.lost_timeout

	def registerScan(self, scan_data):
		if not self.enabled:
			return

		points = self.scanToPoints(scan_data)
		if self.target_x is None or self.target_y is None:
			if not self.acquireTarget(scan_data, points):
				self.publishInfo('laser:no target to acquire')
				self.clearMarker(scan_data.header.frame_id or 'base_scan')
			return

		if self.updateLockedTarget(scan_data, points):
			return

		if self.targetLostTooLong():
			self.publishInfo('laser:target lost, reacquiring')
			self.resetTarget()
			if not self.acquireTarget(scan_data, points):
				self.clearMarker(scan_data.header.frame_id or 'base_scan')
		else:
			self.publishInfo('laser:target temporarily lost')
def main(args=None):
    print('starting')
    rclpy.init(args=args)
    lasertracker = LaserTracker()
    print('seem to do something')
    try:
        rclpy.spin(lasertracker)
    finally:
        lasertracker.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

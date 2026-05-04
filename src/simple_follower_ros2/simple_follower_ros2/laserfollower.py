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
import _thread
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Point, Twist
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from simple_follower_ros2.msg import FollowerPosition as PositionMsg
from std_msgs.msg import Bool, Int8, String as StringMsg
from std_srvs.srv import SetBool
from visualization_msgs.msg import Marker, MarkerArray

class LaserFollower(Node):

	def __init__(self):
		super().__init__('laserfollower')
		self.controllerLossTimer = threading.Timer(1, self.controllerLoss) #if we lose connection
		self.controllerLossTimer.start()
		self.enabled = self.declare_parameter('enabled', False).value
		# as soon as we stop receiving Joy messages from the ps3 controller we stop all movement:
		#self.switchMode= self.declare_parameter('~switchMode').value # if this is set to False the O button has to be kept pressed in order for it to move
		self.switchMode= True
		self.max_speed = self.declare_parameter('max_speed', 0.4).value
		self.target_distance = self.declare_parameter('target_distance', 0.8).value
		self.avoidance_enabled = self.declare_parameter('avoidance_enabled', True).value
		self.apf_influence_dist = self.declare_parameter('apf_influence_dist', 0.9).value
		self.apf_slowdown_dist = self.declare_parameter('apf_slowdown_dist', 0.6).value
		self.apf_emergency_dist = self.declare_parameter('apf_emergency_dist', 0.3).value
		self.apf_repulse_gain = self.declare_parameter('apf_repulse_gain', 0.08).value
		self.apf_turn_gain = self.declare_parameter('apf_turn_gain', 1.3).value
		self.apf_front_angle_deg = self.declare_parameter('apf_front_angle_deg', 140.0).value
		self.p_values = self.declare_parameter('P', [1.6, 0.5]).value
		self.i_values = self.declare_parameter('I', [0.0, 0.0]).value
		self.d_values = self.declare_parameter('D', [0.03, 0.005]).value
		#self.controllButtonIndex = self.declare_parameter('~controllButtonIndex').value
		self.controllButtonIndex = -4
		self.buttonCallbackBusy=False
		self.active=False
		self.i=0
		self.latest_scan = None
		self.last_apf_frame_id = 'laser'
		qos = QoSProfile(depth=10)
		self.laserfwflagPublisher = self.create_publisher(Int8,'/laser_follow_flag',qos)
		self.cmdVelPublisher = self.create_publisher(Twist, 'cmd_vel', qos)
		self.apfMarkerPublisher = self.create_publisher(MarkerArray, 'laser_follow_apf_markers', qos)
		self.enablePublisher = self.create_publisher(
		    Bool,
		    '/laser_follow_enabled',
		    qos)
		self.positionSubscriber = self.create_subscription(
		    PositionMsg,
		    '/object_tracker/current_position',
		    self.positionUpdateCallback,
		    qos)
		self.scanSubscriber = self.create_subscription(
		    LaserScan,
		    '/scan',
		    self.scanCallback,
		    qos_profile_sensor_data)
		self.trackerInfoSubscriber = self.create_subscription(
		    StringMsg,
		    '/object_tracker/info',
		    self.trackerInfoCallback,
		    qos)
		self.enableService = self.create_service(
		    SetBool,
		    '/set_laser_follow_enabled',
		    self.handleEnableService)
		self.PID_controller = simplePID([0, self.target_distance], self.p_values, self.i_values, self.d_values)
		self.set_enabled(self.enabled)
		# PID parameters first is angular, dist

	def scanCallback(self, scan_msg):
		self.latest_scan = scan_msg
		if scan_msg.header.frame_id:
			self.last_apf_frame_id = scan_msg.header.frame_id

	def set_enabled(self, enabled):
		previous = self.enabled
		self.enabled = enabled
		enable_msg = Bool()
		enable_msg.data = self.enabled
		self.enablePublisher.publish(enable_msg)
		if previous != self.enabled:
			state = 'enabled' if self.enabled else 'disabled'
			self.get_logger().info(f'laser following {state}')
		if not self.enabled:
			self.stopMoving()
			self.publish_flag(0)
			self.publishApfMarkers([], 0.0, False)
		return previous != self.enabled

	def handleEnableService(self, request, response):
		self.set_enabled(request.data)
		response.success = True
		state = 'enabled' if self.enabled else 'disabled'
		response.message = f'laser tracking/following {state}'
		return response

	def trackerInfoCallback(self, info):
		# The tracker publishes status for RViz/debugging; avoid echoing it at scan rate.
		return

	def publish_flag(self, value):
		laser_follow_flag=Int8()
		laser_follow_flag.data=value
		self.laserfwflagPublisher.publish(laser_follow_flag)

	def positionUpdateCallback(self, position):
		if not self.enabled:
			return
		angle_x= position.angle_x
		distance = position.distance

		# call the PID controller to update it and get new speeds
		[uncliped_ang_speed, uncliped_lin_speed] = self.PID_controller.update([angle_x, distance])
		# clip these speeds to be less then the maximal speed specified above
		angularSpeed = np.clip(-uncliped_ang_speed, -self.max_speed, self.max_speed)
		linearSpeed  = np.clip(-uncliped_lin_speed, -self.max_speed, self.max_speed)
		forwardSpeed = max(0.0, linearSpeed)
		repulse_turn, slowdown_factor, emergency_stop = self.computeAvoidanceCommand()
		# create the Twist message to send to the cmd_vel topic
		velocity = Twist()	
		velocity.linear.y = 0.0
		velocity.linear.z = 0.0
		velocity.angular.x = 0.0
		velocity.angular.y = 0.0
		if emergency_stop:
			velocity.linear.x = 0.0
			velocity.angular.z = repulse_turn
		elif distance > self.target_distance:
			velocity.linear.x = forwardSpeed * slowdown_factor
			velocity.angular.z = np.clip(angularSpeed + repulse_turn, -0.8, 0.8)
		else:
			velocity.linear.x = 0.0
			velocity.angular.z = np.clip(angularSpeed + repulse_turn, -0.8, 0.8)
		#self.get_logger().info('linearSpeed: {}, angularSpeed: {}'.format(linearSpeed, angularSpeed))
		self.cmdVelPublisher.publish(velocity)
		self.publish_flag(1)

	def buildMarker(self, marker_id, marker_type, frame_id):
		marker = Marker()
		marker.header.frame_id = frame_id
		marker.header.stamp = self.get_clock().now().to_msg()
		marker.ns = 'laser_follow_apf'
		marker.id = marker_id
		marker.type = marker_type
		marker.action = Marker.ADD
		marker.pose.orientation.w = 1.0
		return marker

	def publishApfMarkers(self, obstacle_points, repulse_turn, emergency_stop):
		frame_id = self.last_apf_frame_id
		markers = []

		influence_marker = self.buildMarker(0, Marker.LINE_STRIP, frame_id)
		influence_marker.scale.x = 0.03
		influence_marker.color.a = 0.9
		influence_marker.color.r = 1.0
		influence_marker.color.g = 0.8
		influence_marker.color.b = 0.0

		slowdown_marker = self.buildMarker(1, Marker.LINE_STRIP, frame_id)
		slowdown_marker.scale.x = 0.03
		slowdown_marker.color.a = 0.9
		slowdown_marker.color.r = 1.0
		slowdown_marker.color.g = 0.5
		slowdown_marker.color.b = 0.0

		emergency_marker = self.buildMarker(2, Marker.LINE_STRIP, frame_id)
		emergency_marker.scale.x = 0.04
		emergency_marker.color.a = 0.95
		emergency_marker.color.r = 1.0
		emergency_marker.color.g = 0.0
		emergency_marker.color.b = 0.0

		front_half_angle = np.deg2rad(self.apf_front_angle_deg / 2.0)
		for sample_angle in np.linspace(-front_half_angle, front_half_angle, 31):
			for marker, radius in (
				(influence_marker, self.apf_influence_dist),
				(slowdown_marker, self.apf_slowdown_dist),
				(emergency_marker, self.apf_emergency_dist),
			):
				point = Point()
				point.x = float(radius * np.cos(sample_angle))
				point.y = float(radius * np.sin(sample_angle))
				point.z = 0.02
				marker.points.append(point)
		for marker in (influence_marker, slowdown_marker, emergency_marker):
			start_point = Point()
			start_point.x = 0.0
			start_point.y = 0.0
			start_point.z = 0.02
			marker.points.append(start_point)
			markers.append(marker)

		obstacles_marker = self.buildMarker(3, Marker.POINTS, frame_id)
		obstacles_marker.scale.x = 0.08
		obstacles_marker.scale.y = 0.08
		obstacles_marker.color.a = 1.0
		obstacles_marker.color.r = 1.0
		obstacles_marker.color.g = 0.2
		obstacles_marker.color.b = 0.1
		for point_x, point_y in obstacle_points:
			point = Point()
			point.x = float(point_x)
			point.y = float(point_y)
			point.z = 0.03
			obstacles_marker.points.append(point)
		markers.append(obstacles_marker)

		repulse_marker = self.buildMarker(4, Marker.ARROW, frame_id)
		repulse_marker.scale.x = 0.05
		repulse_marker.scale.y = 0.10
		repulse_marker.scale.z = 0.12
		repulse_marker.color.a = 1.0
		repulse_marker.color.r = 0.1
		repulse_marker.color.g = 0.6
		repulse_marker.color.b = 1.0
		start = Point()
		start.x = 0.0
		start.y = 0.0
		start.z = 0.05
		repulse_marker.points.append(start)
		end = Point()
		arrow_length = max(0.05, min(0.5, abs(repulse_turn) * 0.6))
		end.x = float(0.25)
		end.y = float(np.sign(repulse_turn) * arrow_length)
		end.z = 0.05
		if abs(repulse_turn) < 1e-4:
			end.y = 0.0
		repulse_marker.points.append(end)
		markers.append(repulse_marker)

		status_marker = self.buildMarker(5, Marker.TEXT_VIEW_FACING, frame_id)
		status_marker.scale.z = 0.18
		status_marker.color.a = 1.0
		status_marker.color.r = 1.0 if emergency_stop else 0.2
		status_marker.color.g = 0.2 if emergency_stop else 1.0
		status_marker.color.b = 0.2
		status_marker.pose.position.x = 0.0
		status_marker.pose.position.y = 0.0
		status_marker.pose.position.z = 0.35
		status_marker.text = f'APF turn={repulse_turn:.2f}'
		markers.append(status_marker)

		marker_array = MarkerArray()
		marker_array.markers = markers
		self.apfMarkerPublisher.publish(marker_array)

	def computeAvoidanceCommand(self):
		if not self.avoidance_enabled or self.latest_scan is None:
			self.publishApfMarkers([], 0.0, False)
			return 0.0, 1.0, False

		repulse_turn = 0.0
		min_front_distance = float('inf')
		front_half_angle = np.deg2rad(self.apf_front_angle_deg / 2.0)
		obstacle_points = []

		for index, scan_range in enumerate(self.latest_scan.ranges):
			if np.isinf(scan_range) or np.isnan(scan_range):
				continue
			if scan_range <= 0.0:
				continue

			angle = self.latest_scan.angle_min + index * self.latest_scan.angle_increment
			if abs(angle) > front_half_angle:
				continue

			point_x = scan_range * np.cos(angle)
			point_y = scan_range * np.sin(angle)
			if point_x < -0.05:
				continue

			distance = np.hypot(point_x, point_y)
			if distance < min_front_distance:
				min_front_distance = distance

			if distance >= self.apf_influence_dist:
				continue

			obstacle_points.append((point_x, point_y))

			force = self.apf_repulse_gain * (
				(1.0 / distance) - (1.0 / self.apf_influence_dist)
			) / (distance * distance)
			repulse_turn += -force * (point_y / max(distance, 1e-6))

		repulse_turn = np.clip(repulse_turn * self.apf_turn_gain, -0.8, 0.8)
		emergency_stop = min_front_distance < self.apf_emergency_dist
		self.publishApfMarkers(obstacle_points, repulse_turn, emergency_stop)

		if emergency_stop:
			return repulse_turn, 0.0, True

		slowdown_factor = 1.0
		if min_front_distance < self.apf_slowdown_dist:
			range_span = max(self.apf_slowdown_dist - self.apf_emergency_dist, 1e-3)
			slowdown_factor = (min_front_distance - self.apf_emergency_dist) / range_span
			slowdown_factor = float(np.clip(slowdown_factor, 0.0, 1.0))

		return repulse_turn, slowdown_factor, False
	def buttonCallback(self, joy_data):
		# this method gets called whenever we receive a message from the joy stick

		# there is a timer that always gets reset if we have a new joy stick message
		# if it runs out we know that we have lost connection and the controllerLoss function
		# will be called
		# if we are in switch mode, one button press will make the follower active / inactive 
		# but 'one' button press will be visible in roughly 10 joy messages (since they get published to fast) 
		# so we need to drop the remaining 9
		self.controllerLossTimer.cancel()
		self.controllerLossTimer = threading.Timer(0.5, self.controllerLoss)
		self.controllerLossTimer.start()

		if self.buttonCallbackBusy:
			# we are busy with dealing with the last message
			return 
		else:
			# we are not busy. i.e. there is a real 'new' button press
			# we deal with it in a seperate thread to be able to drop the other joy messages arriving in the mean
			# time
			thread.start_new_thread(self.threadedButtonCallback,  (joy_data, ))
			print("000000000000000")
	def threadedButtonCallback(self, joy_data):
		self.buttonCallbackBusy = True

		if(joy_data.buttons[self.controllButtonIndex]==self.switchMode and self.active):
			# we are active
			# switchMode = false: we will always be inactive whenever the button is not pressed (buttons[index]==false)
			# switchMode = true: we will only become inactive if we press the button. (if we keep pressing it, 
			# we would alternate between active and not in 0.5 second intervalls)
			self.get_logger().info('stoping')
			self.stopMoving()
			self.active = False
			time.sleep(0.5)
		elif(joy_data.buttons[self.controllButtonIndex]==True and not(self.active)):
			# if we are not active and just pressed the button (or are constantly pressing it) we become active
			self.get_logger().info('activating')
			self.active = True #enable response
			time.sleep(0.5)

		self.buttonCallbackBusy = False
	def stopMoving(self):
		velocity = Twist()
		velocity.linear.x = 0.0
		velocity.linear.y = 0.0
		velocity.linear.z = 0.0

		velocity.angular.x = 0.0
		velocity.angular.y = 0.0
		velocity.angular.z = 0.0
		self.cmdVelPublisher.publish(velocity)
	def controllerLoss(self):
		# we lost connection so we will stop moving and become inactive
		self.stopMoving()
		self.publish_flag(0)
		self.active = False
		self.get_logger().info('lost connection')
class simplePID:
	'''very simple discrete PID controller'''
	def __init__(self, target, P, I, D):
		# check if parameter shapes are compatabile. 
		if(not(np.size(P)==np.size(I)==np.size(D)) or ((np.size(target)==1) and np.size(P)!=1) or (np.size(target )!=1 and (np.size(P) != np.size(target) and (np.size(P) != 1)))):
			raise TypeError('input parameters shape is not compatable')
		self.Kp		=np.array(P)
		self.Ki		=np.array(I)
		self.Kd		=np.array(D)
		self.setPoint   =np.array(target)
		
		self.last_error=0
		self.integrator = 0
		self.integrator_max = float('inf')
		self.timeOfLastCall = None 
	def update(self, current_value):

		current_value=np.array(current_value)
		if(np.size(current_value) != np.size(self.setPoint)):
			raise TypeError('current_value and target do not have the same shape')
		if(self.timeOfLastCall is None):
			# the PID was called for the first time. we don't know the deltaT yet
			# no controll signal is applied
			self.timeOfLastCall = time.perf_counter()
			return np.zeros(np.size(current_value))

		
		error = self.setPoint - current_value

		if error[0]<0.1 and error[0]>-0.1:
			error[0]=0
		if error[1]<0.1 and error[1]>-0.1:
			error[1]=0
		
		#when target is little, amplify velocity by amplify error.
		if (error[1]>0 and self.setPoint[1]<1.2):
			error[1]=error[1]*(1.2/self.setPoint[1])*0.7

		P =  error
		
		currentTime = time.perf_counter()
		deltaT      = (currentTime-self.timeOfLastCall)

		# integral of the error is current error * time since last update
		self.integrator = self.integrator + (error*deltaT)
		I = self.integrator
		
		# derivative is difference in error / time since last update
		D = (error-self.last_error)/deltaT
		
		self.last_error = error
		self.timeOfLastCall = currentTime
		
		# return controll signal
		return self.Kp*P + self.Ki*I + self.Kd*D
def main(args=None):
    print('starting')
    rclpy.init(args=args)
    laserfollower = LaserFollower()
    try:
        rclpy.spin(laserfollower)
    finally:
        laserfollower.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

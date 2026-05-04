#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from robot_arm_action.action import MoveArm
from rclpy.action import ActionClient

class MoveArmClient(Node):
    def __init__(self):
        super().__init__('move_arm_client')
        self._action_client = ActionClient(self, MoveArm, 'move_arm')

    def send_goal(self):
        self.get_logger().info('Waiting for action server...')
        self._action_client.wait_for_server()

        goal_msg = MoveArm.Goal()

        # Joint limits from URDF:
        # joint1 [-2.618, 2.618], joint2 [0, 3.14], joint3 [-2.967, 0],
        # joint4 [-1.745, 1.745], joint5 [-1.22, 1.22], joint6 [0, 3.14159]
        goal_msg.joint_positions = [0.0, 1.2, -1.2, 0.0, 0.4, 1.0]

        self.get_logger().info('Sending goal...')
        send_goal_future = self._action_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return

        self.get_logger().info('Goal accepted :)')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Action result: {result.success}')
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    client = MoveArmClient()
    client.send_goal()
    rclpy.spin(client)

if __name__ == '__main__':
    main()

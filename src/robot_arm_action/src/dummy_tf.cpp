#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "std_msgs/msg/string.hpp"
#include "tf2_ros/static_transform_broadcaster.h"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("dummy_trigger_once");

  // Static TF broadcaster (persists forever)
  auto tf_broadcaster = std::make_shared<tf2_ros::StaticTransformBroadcaster>(node);

  // Publisher to trigger your server
  auto trigger_pub = node->create_publisher<std_msgs::msg::String>(
    "/detected_object_tf", 10);

  // -------------------------
  // 1. Publish TF (pick_target)
  // -------------------------
  geometry_msgs::msg::TransformStamped tf_msg;
  tf_msg.header.stamp = node->get_clock()->now();
  tf_msg.header.frame_id = "base_link";
  tf_msg.child_frame_id = "pick_target";   // 🔥 MUST match server

  tf_msg.transform.translation.x = 0.5;
  tf_msg.transform.translation.y = 0.0;
  tf_msg.transform.translation.z = 0.1;

  tf_msg.transform.rotation.x = 0.0;
  tf_msg.transform.rotation.y = 0.0;
  tf_msg.transform.rotation.z = 0.0;
  tf_msg.transform.rotation.w = 1.0;

  tf_broadcaster->sendTransform(tf_msg);

  RCLCPP_INFO(node->get_logger(), "TF 'pick_target' published.");

  // -------------------------
  // 2. Small delay (important!)
  // -------------------------
  rclcpp::sleep_for(std::chrono::milliseconds(500));

  // -------------------------
  // 3. Publish trigger ONCE
  // -------------------------
  std_msgs::msg::String msg;
  msg.data = "trigger";  // content doesn't matter

  trigger_pub->publish(msg);

  RCLCPP_INFO(node->get_logger(), "Trigger sent. Robot should pick now.");

  // Keep node alive so TF is available
  rclcpp::spin(node);

  rclcpp::shutdown();
  return 0;
}
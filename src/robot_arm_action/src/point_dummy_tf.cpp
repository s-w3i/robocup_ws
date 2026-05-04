#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "tf2_ros/static_transform_broadcaster.h"

#include "rclcpp_action/rclcpp_action.hpp"
#include "robot_arm_action/action/point.hpp"

class DummyPointTrigger : public rclcpp::Node
{
public:
  using Point = robot_arm_action::action::Point;
  using GoalHandlePoint = rclcpp_action::ClientGoalHandle<Point>;

  DummyPointTrigger()
  : Node("dummy_point_trigger")
  {
    // -------------------------
    // TF broadcaster
    // -------------------------
    tf_broadcaster_ =
      std::make_shared<tf2_ros::StaticTransformBroadcaster>(this);

    // -------------------------
    // Action client
    // -------------------------
    point_client_ =
      rclcpp_action::create_client<Point>(this, "point_object");

    RCLCPP_INFO(this->get_logger(), "Dummy Point Trigger Ready");

    // auto run once after startup
    timer_ = this->create_wall_timer(
      std::chrono::seconds(1),
      std::bind(&DummyPointTrigger::run_once, this)
    );
  }

private:
  std::shared_ptr<tf2_ros::StaticTransformBroadcaster> tf_broadcaster_;
  rclcpp_action::Client<Point>::SharedPtr point_client_;
  rclcpp::TimerBase::SharedPtr timer_;

  bool executed_ = false;

  void run_once()
  {
    if (executed_) return;
    executed_ = true;

    // -------------------------
    // 1. Publish TF (emptychair1)
    // -------------------------
    geometry_msgs::msg::TransformStamped tf_msg;
    tf_msg.header.stamp = this->get_clock()->now();
    tf_msg.header.frame_id = "base_link";
    tf_msg.child_frame_id = "emptychair1";

    tf_msg.transform.translation.x = 0.6;
    tf_msg.transform.translation.y = 0.3;
    tf_msg.transform.translation.z = 0.3;

    tf_msg.transform.rotation.x = 0.0;
    tf_msg.transform.rotation.y = 0.0;
    tf_msg.transform.rotation.z = 0.0;
    tf_msg.transform.rotation.w = 1.0;

    tf_broadcaster_->sendTransform(tf_msg);

    RCLCPP_INFO(this->get_logger(), "TF 'emptychair1' published");

    // -------------------------
    // 2. Wait for action server
    // -------------------------
    if (!point_client_->wait_for_action_server(std::chrono::seconds(2))) {
      RCLCPP_ERROR(this->get_logger(), "Point action server not available");
      return;
    }

    // -------------------------
    // 3. Send Point Action Goal
    // -------------------------
    auto goal_msg = Point::Goal();
    goal_msg.target_frame = "emptychair1";

    rclcpp_action::Client<Point>::SendGoalOptions options;

    options.result_callback =
      [this](const GoalHandlePoint::WrappedResult & result)
      {
        if (result.code == rclcpp_action::ResultCode::SUCCEEDED)
        {
          RCLCPP_INFO(this->get_logger(),
            "POINT SUCCESS: %s",
            result.result->message.c_str());
        }
        else
        {
          RCLCPP_ERROR(this->get_logger(), "POINT FAILED");
        }
      };

    options.feedback_callback =
      [this](GoalHandlePoint::SharedPtr,
             const std::shared_ptr<const Point::Feedback> feedback)
      {
        RCLCPP_INFO(this->get_logger(),
          "Feedback: %s",
          feedback->status.c_str());
      };

    point_client_->async_send_goal(goal_msg, options);

    RCLCPP_INFO(this->get_logger(), "Point goal sent");
  }
};

// -------------------------
// MAIN
// -------------------------
int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<DummyPointTrigger>();

  rclcpp::spin(node);

  rclcpp::shutdown();
  return 0;
}
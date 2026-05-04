#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "robot_arm_action/action/move_arm.hpp"

using MoveArm = robot_arm_action::action::MoveArm;
using GoalHandleMoveArm = rclcpp_action::ClientGoalHandle<MoveArm>;

class MoveArmClient : public rclcpp::Node
{
public:
  MoveArmClient() : Node("move_arm_client")
  {
    client_ptr_ = rclcpp_action::create_client<MoveArm>(this, "move_arm");
  }

  void send_goal(const std::string & target_frame)
  {
    if (!client_ptr_->wait_for_action_server(std::chrono::seconds(5))) {
      RCLCPP_ERROR(this->get_logger(), "Action server not available!");
      return;
    }

    auto goal_msg = MoveArm::Goal();
    goal_msg.target_frame = target_frame;

    auto send_goal_options = rclcpp_action::Client<MoveArm>::SendGoalOptions();

    send_goal_options.goal_response_callback =
      [this](const GoalHandleMoveArm::SharedPtr & goal_handle) {
        if (!goal_handle) {
          RCLCPP_ERROR(this->get_logger(), "Goal was rejected by server");
        } else {
          RCLCPP_INFO(this->get_logger(), "Goal accepted by server, waiting for result");
        }
      };

    send_goal_options.result_callback =
      [this](const GoalHandleMoveArm::WrappedResult & result) {
        switch (result.code) {
          case rclcpp_action::ResultCode::SUCCEEDED:
            RCLCPP_INFO(this->get_logger(), "Goal succeeded! status=%s", result.result->status.c_str());
            break;
          case rclcpp_action::ResultCode::ABORTED:
            RCLCPP_ERROR(
              this->get_logger(),
              "Goal aborted. status=%s message=%s",
              result.result->status.c_str(),
              result.result->message.c_str());
            break;
          default:
            RCLCPP_ERROR(this->get_logger(), "Goal failed with result code=%d", static_cast<int>(result.code));
            break;
        }
      };

    client_ptr_->async_send_goal(goal_msg, send_goal_options);
  }

private:
  rclcpp_action::Client<MoveArm>::SharedPtr client_ptr_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<MoveArmClient>();

  std::thread ros_thread([node]() { rclcpp::spin(node); });

  std::string input;
  while (rclcpp::ok()) {
    std::cout << "\nEnter target TF frame (or q): " << std::flush;
    if (!std::getline(std::cin, input) || input == "q" || input == "quit" || input == "exit") {
      break;
    }
    if (!input.empty()) {
      node->send_goal(input);
    }
  }

  rclcpp::shutdown();
  ros_thread.join();
  return 0;
}

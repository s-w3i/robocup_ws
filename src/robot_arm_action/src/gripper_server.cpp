#include <memory>
#include <string>
#include "rclcpp/rclcpp.hpp"
#include "robot_arm_control/srv/gripper_command.hpp"
#include "moveit/move_group_interface/move_group_interface.h"

using GripperCommand = robot_arm_control::srv::GripperCommand;

class GripperServer : public rclcpp::Node
{
public:
    GripperServer()
    : Node("gripper_server")
    {
        service_ = this->create_service<GripperCommand>(
            "gripper_control",
            std::bind(&GripperServer::handle_gripper, this, std::placeholders::_1, std::placeholders::_2));
    }

private:
    rclcpp::Service<GripperCommand>::SharedPtr service_;

    void handle_gripper(
        const std::shared_ptr<GripperCommand::Request> req,
        std::shared_ptr<GripperCommand::Response> res)
    {
        moveit::planning_interface::MoveGroupInterface gripper_group(this, "gripper");

        if (req->command == "open") {
            gripper_group.setNamedTarget("open");
        } else if (req->command == "close") {
            gripper_group.setNamedTarget("close");
        } else {
            res->success = false;
            res->message = "Unknown command";
            return;
        }

        if (gripper_group.move() == moveit::core::MoveItErrorCode::SUCCESS) {
            res->success = true;
            res->message = "Gripper moved";
        } else {
            res->success = false;
            res->message = "Move failed";
        }
    }
};

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<GripperServer>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
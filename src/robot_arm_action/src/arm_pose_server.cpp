#include <memory>
#include <string>
#include "rclcpp/rclcpp.hpp"
#include "robot_arm_control/srv/arm_pose.hpp"
#include "moveit/move_group_interface/move_group_interface.h"

using ArmPose = robot_arm_control::srv::ArmPose;

class ArmPoseServer : public rclcpp::Node
{
public:
    ArmPoseServer()
    : Node("arm_pose_server")
    {
        service_ = this->create_service<ArmPose>(
            "arm_pose",
            std::bind(&ArmPoseServer::handle_pose, this, std::placeholders::_1, std::placeholders::_2));
    }

private:
    rclcpp::Service<ArmPose>::SharedPtr service_;

    void handle_pose(
        const std::shared_ptr<ArmPose::Request> req,
        std::shared_ptr<ArmPose::Response> res)
    {
        moveit::planning_interface::MoveGroupInterface move_group(this, "arm");

        if (req->pose_name == "zero") {
            move_group.setNamedTarget("zero");
        } else if (req->pose_name == "detect") {
            move_group.setNamedTarget("detect");
        } else {
            res->success = false;
            res->message = "Unknown pose";
            return;
        }

        if (move_group.move() == moveit::core::MoveItErrorCode::SUCCESS) {
            res->success = true;
            res->message = "Pose reached";
        } else {
            res->success = false;
            res->message = "Move failed";
        }
    }
};

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ArmPoseServer>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
// #include <memory>
// #include <string>
// #include <thread>
// #include <mutex>

// #include "geometry_msgs/msg/transform_stamped.hpp"
// #include "geometry_msgs/msg/pose.hpp" // Added for Pose message
// #include "moveit/move_group_interface/move_group_interface.h"
// #include "rclcpp/rclcpp.hpp"
// #include "rclcpp_action/rclcpp_action.hpp"
// #include "robot_arm_action/action/move_arm.hpp"
// #include "tf2/exceptions.h"
// #include "tf2_ros/buffer.h"
// #include "tf2_ros/transform_listener.h"
// #include "tf2/LinearMath/Quaternion.h"
// #include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
// #include "moveit_msgs/msg/robot_trajectory.hpp"

// using MoveArm = robot_arm_action::action::MoveArm;
// using GoalHandleMoveArm = rclcpp_action::ServerGoalHandle<MoveArm>;

// class MoveArmServer : public rclcpp::Node
// {
// public:
//   explicit MoveArmServer(const rclcpp::NodeOptions & options)
//   : Node("move_arm_server", options)
//   {
//     tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
//     tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

//     action_server_ = rclcpp_action::create_server<MoveArm>(
//       this,
//       "move_arm",
//       std::bind(&MoveArmServer::handle_goal, this, std::placeholders::_1, std::placeholders::_2),
//       std::bind(&MoveArmServer::handle_cancel, this, std::placeholders::_1),
//       std::bind(&MoveArmServer::handle_accepted, this, std::placeholders::_1));

//     RCLCPP_INFO(this->get_logger(), "Move Arm Server is Online. Waiting for TF frame goals...");
//   }

// private:
//   rclcpp_action::Server<MoveArm>::SharedPtr action_server_;
//   std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
//   std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

//   std::mutex moveit_mutex_;

//   rclcpp_action::GoalResponse handle_goal(
//     const rclcpp_action::GoalUUID &,
//     std::shared_ptr<const MoveArm::Goal> goal)
//   {
//     if (goal->target_frame.empty()) {
//       RCLCPP_WARN(this->get_logger(), "Rejecting empty target_frame goal");
//       return rclcpp_action::GoalResponse::REJECT;
//     }
//     return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
//   }

//   rclcpp_action::CancelResponse handle_cancel(const std::shared_ptr<GoalHandleMoveArm>)
//   {
//     return rclcpp_action::CancelResponse::ACCEPT;
//   }

//   void handle_accepted(const std::shared_ptr<GoalHandleMoveArm> goal_handle)
//   {
//     std::thread{std::bind(&MoveArmServer::execute, this, std::placeholders::_1), goal_handle}.detach();
//   }

//   void execute(const std::shared_ptr<GoalHandleMoveArm> goal_handle)
//   {
//     std::lock_guard<std::mutex> lock(moveit_mutex_); 

//     auto result = std::make_shared<MoveArm::Result>();
//     const std::string target_frame = goal_handle->get_goal()->target_frame;

//     moveit::planning_interface::MoveGroupInterface move_group(shared_from_this(), "arm");
//     // Tolerance
//     move_group.setPlanningTime(5.0);
//     move_group.setNumPlanningAttempts(10);
//     move_group.setGoalPositionTolerance(0.01);
//     move_group.setGoalOrientationTolerance(0.02);

//     // move_group.setMaxVelocityScalingFactor(0.01);
//     // move_group.setMaxAccelerationScalingFactor(0.01);

//     geometry_msgs::msg::TransformStamped tf_goal;
//     try {
//       tf_goal = tf_buffer_->lookupTransform(
//         "base_link", target_frame, tf2::TimePointZero, std::chrono::seconds(1));
//     } catch (const tf2::TransformException & ex) {
//       RCLCPP_ERROR(this->get_logger(), "TF lookup failed for '%s': %s", target_frame.c_str(), ex.what());
//       result->success = false;
//       result->status = "TF lookup failed";
//       result->message = ex.what();
//       goal_handle->abort(result);
//       return;
//     }

//     const auto & t = tf_goal.transform.translation;
//     // move_group.setPositionTarget(t.x, t.y, t.z + 0.5, "gripper_tcp");
//     // //move_group.setOrientationTarget(0.0, (M_PI_2 / 2), 0.0, 0.0, "gripper_tcp");
//     geometry_msgs::msg::Pose target_pose;

//     // Position
//     target_pose.position.x = t.x ;
//     target_pose.position.y = t.y;
//     target_pose.position.z = t.z + 0.3;

//     // Orientation
//     tf2::Quaternion q_tf, q_offset, q_final;

//     tf2::fromMsg(tf_goal.transform.rotation, q_tf);

//     q_offset.setRPY(1.57, 1.3, 0);  // adjust as needed
//     q_offset.normalize();

//     q_final = q_tf * q_offset;
//     q_final.normalize();

//     target_pose.orientation = tf2::toMsg(q_final);

//     // Send to MoveIt
//     move_group.setPoseTarget(target_pose, "gripper_tcp");


//     RCLCPP_INFO(
//       this->get_logger(),
//       "Planning to TF '%s' at [X:%.3f, Y:%.3f, Z:%.3f] in base_link",
//       target_frame.c_str(), t.x, t.y, t.z);



//     moveit::planning_interface::MoveGroupInterface::Plan plan;
//     const bool success = (move_group.plan(plan) == moveit::core::MoveItErrorCode::SUCCESS);
//     if (success) {
//       if (goal_handle->is_canceling()) {
//         result->success = false;
//         result->status = "Goal canceled";
//         result->message = "Canceled before execution";
//         goal_handle->canceled(result);
//         return;
//       }

//       const auto exec_ret = move_group.execute(plan);
//       if (goal_handle->is_canceling()) {
//         result->success = false;
//         result->status = "Goal canceled";
//         result->message = "Canceled during execution";
//         goal_handle->canceled(result);
//         return;
//       }
//       if (exec_ret != moveit::core::MoveItErrorCode::SUCCESS) {
//         result->success = false;
//         result->status = "Execution failed";
//         result->message = "MoveIt execution did not succeed";
//         goal_handle->abort(result);
//         return;
//       }

//       result->success = true;
//       result->status = "Goal Reached";
//       goal_handle->succeed(result);
//       return;
//     }

//     result->success = false;
//     result->status = "Planning failed";
//     goal_handle->abort(result);
//   }
// };

// int main(int argc, char ** argv)
// {
//   rclcpp::init(argc, argv);

//   rclcpp::NodeOptions node_options;
//   node_options.automatically_declare_parameters_from_overrides(true);

//   auto node = std::make_shared<MoveArmServer>(node_options);
//   rclcpp::spin(node);
//   rclcpp::shutdown();
//   return 0;
// }
#include <memory>
#include <string>
#include <thread>
#include <mutex>

#include "geometry_msgs/msg/transform_stamped.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "moveit/move_group_interface/move_group_interface.h"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "robot_arm_action/action/move_arm.hpp"
#include "robot_arm_action/srv/gripper_command.hpp"
#include "tf2/exceptions.h"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

using MoveArm = robot_arm_action::action::MoveArm;
using GoalHandleMoveArm = rclcpp_action::ServerGoalHandle<MoveArm>;
using GripperCommand = robot_arm_action::srv::GripperCommand;

class MoveArmServer : public rclcpp::Node
{
public:
  explicit MoveArmServer(const rclcpp::NodeOptions & options)
  : Node("move_arm_server", options)
  {
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // 🔹 Action Server
    action_server_ = rclcpp_action::create_server<MoveArm>(
      this,
      "move_arm",
      std::bind(&MoveArmServer::handle_goal, this, std::placeholders::_1, std::placeholders::_2),
      std::bind(&MoveArmServer::handle_cancel, this, std::placeholders::_1),
      std::bind(&MoveArmServer::handle_accepted, this, std::placeholders::_1));

    // 🔹 Gripper Service
    gripper_service_ = this->create_service<GripperCommand>(
      "/open_gripper",
      std::bind(&MoveArmServer::handleGripperCommand, this,
                std::placeholders::_1, std::placeholders::_2));

    RCLCPP_INFO(this->get_logger(), "Move Arm Server Online. Waiting for targets...");
  }

  // ======================
  // 🔹 INIT MOVE GROUPS
  // ======================
  void initMoveGroups()
  {
    arm_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
      shared_from_this(), "arm");
    gripper_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
      shared_from_this(), "gripper");

    arm_group_->setPlanningTime(5.0);
    arm_group_->setNumPlanningAttempts(10);

    // 🔹 Move to detect pose + open gripper on startup
    std::thread([this]() { startupPose(); }).detach();
  }

private:
  // Servers
  rclcpp_action::Server<MoveArm>::SharedPtr action_server_;
  rclcpp::Service<GripperCommand>::SharedPtr gripper_service_;

  // TF
  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  // MoveIt
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> arm_group_;
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> gripper_group_;

  std::mutex moveit_mutex_;

  // ======================
  // 🔹 Startup Pose
  // ======================
  void startupPose()
  {
    std::lock_guard<std::mutex> lock(moveit_mutex_);
    if (!arm_group_ || !gripper_group_) return;

    RCLCPP_INFO(this->get_logger(), "Moving to detect pose and opening gripper...");
    
    // Arm
    arm_group_->setNamedTarget("detect");
    moveit::planning_interface::MoveGroupInterface::Plan plan;
    if (arm_group_->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS)
      arm_group_->execute(plan);

    // Gripper open
    gripper_group_->setNamedTarget("open");
    moveit::planning_interface::MoveGroupInterface::Plan grip_plan;
    if (gripper_group_->plan(grip_plan) == moveit::core::MoveItErrorCode::SUCCESS)
      gripper_group_->execute(grip_plan);
  }

  // ======================
  // 🔹 Action Callbacks
  // ======================
  rclcpp_action::GoalResponse handle_goal(
    const rclcpp_action::GoalUUID &, std::shared_ptr<const MoveArm::Goal> goal)
  {
    if (goal->target_frame.empty())
    {
      RCLCPP_WARN(this->get_logger(), "Rejecting empty target_frame");
      return rclcpp_action::GoalResponse::REJECT;
    }
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
  }

  rclcpp_action::CancelResponse handle_cancel(const std::shared_ptr<GoalHandleMoveArm>)
  {
    return rclcpp_action::CancelResponse::ACCEPT;
  }

  void handle_accepted(const std::shared_ptr<GoalHandleMoveArm> goal_handle)
  {
    std::thread([this, goal_handle]() { execute(goal_handle); }).detach();
  }

  // ======================
  // 🔹 Execute Pick Task
  // ======================
  void execute(const std::shared_ptr<GoalHandleMoveArm> goal_handle)
  {
    std::lock_guard<std::mutex> lock(moveit_mutex_);
    auto result = std::make_shared<MoveArm::Result>();
    const std::string target_frame = goal_handle->get_goal()->target_frame;

    if (!arm_group_ || !gripper_group_)
    {
      result->success = false;
      result->status = "MoveGroup not initialized";
      goal_handle->abort(result);
      return;
    }

    // 1️⃣ TF lookup
    geometry_msgs::msg::TransformStamped tf_goal;
    try {
      tf_goal = tf_buffer_->lookupTransform("base_link", target_frame,
                                            tf2::TimePointZero, std::chrono::seconds(1));
    } catch (const tf2::TransformException & ex) {
      RCLCPP_ERROR(this->get_logger(), "TF failed: %s", ex.what());
      resetDetectPose();
      result->success = false;
      goal_handle->abort(result);
      return;
    }

    // 2️⃣ Move arm to object
    geometry_msgs::msg::Pose target_pose;
    const auto & t = tf_goal.transform.translation;
    target_pose.position.x = t.x;
    target_pose.position.y = t.y;
    target_pose.position.z = t.z + 0.2;

    tf2::Quaternion q_tf, q_offset, q_final;
    tf2::fromMsg(tf_goal.transform.rotation, q_tf);
    q_offset.setRPY(1.57, 1.3, 0);  // adjust to face down
    q_final = q_tf * q_offset;
    q_final.normalize();
    target_pose.orientation = tf2::toMsg(q_final);

    arm_group_->setPoseTarget(target_pose, "gripper_tcp");

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    if (arm_group_->plan(plan) != moveit::core::MoveItErrorCode::SUCCESS)
    {
      resetDetectPose();
      result->success = false;
      goal_handle->abort(result);
      return;
    }

    if (arm_group_->execute(plan) != moveit::core::MoveItErrorCode::SUCCESS)
    {
      resetDetectPose();
      result->success = false;
      goal_handle->abort(result);
      return;
    }

    // 3️⃣ Close gripper
    gripper_group_->setNamedTarget("close");
    moveit::planning_interface::MoveGroupInterface::Plan grip_plan;
    if (gripper_group_->plan(grip_plan) == moveit::core::MoveItErrorCode::SUCCESS)
      gripper_group_->execute(grip_plan);

    // 4️⃣ Return to detect pose
    arm_group_->setNamedTarget("detect");
    moveit::planning_interface::MoveGroupInterface::Plan return_plan;
    if (arm_group_->plan(return_plan) == moveit::core::MoveItErrorCode::SUCCESS)
      arm_group_->execute(return_plan);

    result->success = true;
    result->status = "Pick complete";
    goal_handle->succeed(result);
  }

  // ======================
  // 🔹 Reset to detect + open gripper
  // ======================
  void resetDetectPose()
  {
    arm_group_->setNamedTarget("detect");
    moveit::planning_interface::MoveGroupInterface::Plan plan;
    if (arm_group_->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS)
      arm_group_->execute(plan);

    gripper_group_->setNamedTarget("open");
    moveit::planning_interface::MoveGroupInterface::Plan grip_plan;
    if (gripper_group_->plan(grip_plan) == moveit::core::MoveItErrorCode::SUCCESS)
      gripper_group_->execute(grip_plan);
  }

  // ======================
  // 🔹 Gripper Service Callback
  // ======================
  void handleGripperCommand(
    const std::shared_ptr<GripperCommand::Request> request,
    std::shared_ptr<GripperCommand::Response> response)
  {
    std::lock_guard<std::mutex> lock(moveit_mutex_);

    if (!gripper_group_)
    {
      response->success = false;
      response->message = "Gripper not initialized";
      return;
    }

    const std::string cmd = request->command;
    gripper_group_->setNamedTarget(cmd);

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    if (gripper_group_->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS &&
        gripper_group_->execute(plan) == moveit::core::MoveItErrorCode::SUCCESS)
    {
      response->success = true;
      response->message = "Gripper " + cmd + " executed";
    }
    else
    {
      response->success = false;
      response->message = "Failed to execute gripper " + cmd;
    }
  }
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions node_options;
  node_options.automatically_declare_parameters_from_overrides(true);

  auto node = std::make_shared<MoveArmServer>(node_options);
  node->initMoveGroups();

  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
#include <memory>
#include <string>
#include <thread>
#include <mutex>

#include "geometry_msgs/msg/transform_stamped.hpp"
#include "geometry_msgs/msg/pose.hpp"

#include "moveit/move_group_interface/move_group_interface.h"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"

#include "robot_arm_action/action/pick.hpp"
#include "robot_arm_action/srv/gripper_command.hpp"

#include "tf2/exceptions.h"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

// ----------------------
using Pick = robot_arm_action::action::Pick;
using GripperCommand = robot_arm_action::srv::GripperCommand;
using GoalHandlePick = rclcpp_action::ServerGoalHandle<Pick>;
// ----------------------

class AutoPickServer : public rclcpp::Node
{
public:
  AutoPickServer(const rclcpp::NodeOptions & options)
  : Node("auto_pick_server", options)
  {
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    gripper_service_ = this->create_service<GripperCommand>(
      "/open_gripper",
      std::bind(&AutoPickServer::handle_gripper, this,
                std::placeholders::_1, std::placeholders::_2));

    pick_server_ = rclcpp_action::create_server<Pick>(
      this,
      "pick_object",
      std::bind(&AutoPickServer::handle_pick_goal, this, std::placeholders::_1, std::placeholders::_2),
      std::bind(&AutoPickServer::handle_pick_cancel, this, std::placeholders::_1),
      std::bind(&AutoPickServer::handle_pick_accepted, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(), "AutoPickServer Ready");
  }

  void initMoveGroups()
  {
    arm_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
      shared_from_this(), "arm");

    gripper_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
      shared_from_this(), "gripper");

    std::thread([this]() { startupDetectPose(); }).detach();
  }

private:
  std::mutex moveit_mutex_;
  bool picking_in_progress_ = false;

  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> arm_group_;
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> gripper_group_;

  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  rclcpp_action::Server<Pick>::SharedPtr pick_server_;
  rclcpp::Service<GripperCommand>::SharedPtr gripper_service_;

  // -----------------------
  // Retry helper
  // -----------------------
  bool planAndExecuteWithRetry(
    moveit::planning_interface::MoveGroupInterface & group,
    moveit::planning_interface::MoveGroupInterface::Plan & plan,
    const std::string & name)
  {
    for (int i = 0; i < 3; i++)
    {
      RCLCPP_WARN(this->get_logger(), "[%s] attempt %d/3", name.c_str(), i + 1);

      if (group.plan(plan) == moveit::core::MoveItErrorCode::SUCCESS)
      {
        if (group.execute(plan) == moveit::core::MoveItErrorCode::SUCCESS)
        {
          RCLCPP_INFO(this->get_logger(), "[%s] success", name.c_str());
          return true;
        }
      }
    }

    RCLCPP_ERROR(this->get_logger(), "[%s] FAILED after 3 attempts", name.c_str());
    return false;
  }

  // -----------------------
  void startupDetectPose()
  {
    std::lock_guard<std::mutex> lock(moveit_mutex_);

    moveit::planning_interface::MoveGroupInterface::Plan plan;

    arm_group_->setNamedTarget("zero");
    arm_group_->plan(plan);
    arm_group_->execute(plan);

    gripper_group_->setNamedTarget("open");
    gripper_group_->plan(plan);
    gripper_group_->execute(plan);
  }

  // -----------------------
  void handle_gripper(
    const std::shared_ptr<GripperCommand::Request> req,
    std::shared_ptr<GripperCommand::Response> res)
  {
    std::lock_guard<std::mutex> lock(moveit_mutex_);

    moveit::planning_interface::MoveGroupInterface::Plan plan;

    gripper_group_->setNamedTarget(req->command);

    bool ok =
      gripper_group_->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS &&
      gripper_group_->execute(plan) == moveit::core::MoveItErrorCode::SUCCESS;

    res->success = ok;
  }

  // -----------------------
  rclcpp_action::GoalResponse handle_pick_goal(
    const rclcpp_action::GoalUUID &,
    std::shared_ptr<const Pick::Goal> goal)
  {
    if (goal->target_tf.empty())
    {
      RCLCPP_ERROR(this->get_logger(), "Empty TF goal");
      return rclcpp_action::GoalResponse::REJECT;
    }

    if (picking_in_progress_)
    {
      RCLCPP_WARN(this->get_logger(), "Already picking");
      return rclcpp_action::GoalResponse::REJECT;
    }

    RCLCPP_INFO(this->get_logger(),
      "Goal accepted: %s", goal->target_tf.c_str());

    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
  }

  rclcpp_action::CancelResponse handle_pick_cancel(
    const std::shared_ptr<GoalHandlePick>)
  {
    return rclcpp_action::CancelResponse::ACCEPT;
  }

  void handle_pick_accepted(const std::shared_ptr<GoalHandlePick> goal_handle)
  {
    std::thread([this, goal_handle]() {
      execute_pick(goal_handle);
    }).detach();
  }

  // -----------------------
  void execute_pick(const std::shared_ptr<GoalHandlePick> goal_handle)
  {
    std::lock_guard<std::mutex> lock(moveit_mutex_);
    picking_in_progress_ = true;

    auto result = std::make_shared<Pick::Result>();
    auto goal = goal_handle->get_goal();

    const std::string target_frame = goal->target_tf;

    geometry_msgs::msg::TransformStamped tf_goal;

    bool success = true;

    try
    {
      tf_goal = tf_buffer_->lookupTransform(
        "base_link",
        target_frame,
        tf2::TimePointZero);
    }
    catch (const tf2::TransformException & ex)
    {
      RCLCPP_ERROR(this->get_logger(), "TF error: %s", ex.what());
      success = false;
    }

    moveit::planning_interface::MoveGroupInterface::Plan plan;

    //----------------- Open Gripper Before Move ---------------------------------------
    geometry_msgs::msg::Pose pose_above;
    if (success)
    {
      gripper_group_->setNamedTarget("open");
      success = planAndExecuteWithRetry(*gripper_group_, plan, "GRIP");
    }

    // ---------------- ABOVE ----------------
    if (success)
    {
      pose_above.position.x = tf_goal.transform.translation.x;
      pose_above.position.y = tf_goal.transform.translation.y;
      pose_above.position.z = 0.4;//tf_goal.transform.translation.z + 0.1;

      tf2::Quaternion q;
      q.setRPY(1.57, 1.3, 0);
      pose_above.orientation = tf2::toMsg(q);

      arm_group_->setPoseTarget(pose_above, "gripper_tcp");
      success = planAndExecuteWithRetry(*arm_group_, plan, "MOVE ABOVE");
    }

    // ---------------- DOWN ----------------
    geometry_msgs::msg::Pose pose_grasp;
    if (success)
    {
      pose_grasp = pose_above;
      pose_grasp.position.z = 0.25;//tf_goal.transform.translation.z + 0.02;

      arm_group_->setPoseTarget(pose_grasp, "gripper_tcp");
      success = planAndExecuteWithRetry(*arm_group_, plan, "MOVE DOWN");
    }

    // ---------------- GRIP ----------------
    if (success)
    {
      gripper_group_->setNamedTarget("close");
      success = planAndExecuteWithRetry(*gripper_group_, plan, "GRIP");
    }

    // ---------------- LIFT ----------------
    if (success)
    {
      arm_group_->setPoseTarget(pose_above, "gripper_tcp");
      success = planAndExecuteWithRetry(*arm_group_, plan, "LIFT");
    }

    // ---------------- RETURN ----------------
    if (success)
    {
      arm_group_->setNamedTarget("detect");
      success = planAndExecuteWithRetry(*arm_group_, plan, "RETURN");
    }

    // ---------------- RESULT ----------------
    if (success)
    {
      result->success = true;
      goal_handle->succeed(result);
      RCLCPP_INFO(this->get_logger(), "Pick SUCCESS");
    }
    else
    {
      RCLCPP_ERROR(this->get_logger(), "Pick FAILED → recovery");

      arm_group_->setNamedTarget("detect");
      moveit::planning_interface::MoveGroupInterface::Plan recovery;
      planAndExecuteWithRetry(*arm_group_, recovery, "RECOVERY");

      result->success = false;
      goal_handle->abort(result);
    }

    picking_in_progress_ = false;
  }
};

// -----------------------
int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<AutoPickServer>(rclcpp::NodeOptions());
  node->initMoveGroups();

  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}

// Test code (ignore)

// #include <memory>
// #include <string>
// #include <thread>
// #include <mutex>

// #include "geometry_msgs/msg/transform_stamped.hpp"
// #include "geometry_msgs/msg/pose.hpp"
// #include "std_msgs/msg/string.hpp"

// #include "moveit/move_group_interface/move_group_interface.h"
// #include "rclcpp/rclcpp.hpp"
// #include "rclcpp_action/rclcpp_action.hpp"

// #include "robot_arm_action/action/pick.hpp"
// #include "robot_arm_action/srv/gripper_command.hpp"

// #include "tf2/exceptions.h"
// #include "tf2_ros/buffer.h"
// #include "tf2_ros/transform_listener.h"
// #include "tf2_ros/transform_broadcaster.h"
// #include "tf2/LinearMath/Quaternion.h"
// #include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

// using Pick = robot_arm_action::action::Pick;
// using GripperCommand = robot_arm_action::srv::GripperCommand;
// using GoalHandlePick = rclcpp_action::ServerGoalHandle<Pick>;

// class AutoPickServer : public rclcpp::Node
// {
// public:
//   AutoPickServer(const rclcpp::NodeOptions & options)
//   : Node("auto_pick_server", options)
//   {
//     tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
//     tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

//     // Dummy TF broadcaster
//     tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

//     // Publisher for TF name string
//     tf_name_pub_ = this->create_publisher<std_msgs::msg::String>(
//       "/detected_object_tf", 10);

//     // Gripper service
//     gripper_service_ = this->create_service<GripperCommand>(
//       "/open_gripper",
//       std::bind(&AutoPickServer::handle_gripper, this,
//                 std::placeholders::_1, std::placeholders::_2));

//     // Pick action server
//     pick_server_ = rclcpp_action::create_server<Pick>(
//       this, "pick_object",
//       std::bind(&AutoPickServer::handle_pick_goal, this, std::placeholders::_1, std::placeholders::_2),
//       std::bind(&AutoPickServer::handle_pick_cancel, this, std::placeholders::_1),
//       std::bind(&AutoPickServer::handle_pick_accepted, this, std::placeholders::_1));

//     RCLCPP_INFO(this->get_logger(), "AutoPickServer Ready");
//   }

//   void initMoveGroups()
//   {
//     arm_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
//       shared_from_this(), "arm");

//     gripper_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
//       shared_from_this(), "gripper");

//     // Move to detect pose on startup
//     std::thread([this]() { startupDetectPose(); }).detach();
//   }

//   // -----------------------
//   // MANUAL FUNCTION: Publish dummy TF + trigger pick
//   // -----------------------
//   void triggerDummyPickTFOnce()
//   {
//     std::lock_guard<std::mutex> lock(moveit_mutex_);

//     // 1️⃣ Publish the dummy TF
//     geometry_msgs::msg::TransformStamped tf_msg;
//     tf_msg.header.stamp = this->get_clock()->now();
//     tf_msg.header.frame_id = "base_link";
//     tf_msg.child_frame_id = "pick_target";  // dummy object frame
//     tf_msg.transform.translation.x = 0.5;
//     tf_msg.transform.translation.y = 0.0;
//     tf_msg.transform.translation.z = 0.1;
//     tf_msg.transform.rotation.w = 1.0;

//     tf_broadcaster_->sendTransform(tf_msg);

//     // 2️⃣ Publish TF name as string
//     auto msg = std::make_shared<std_msgs::msg::String>();
//     msg->data = "pick_target";
//     tf_name_pub_->publish(*msg);

//     RCLCPP_INFO(this->get_logger(), "Dummy TF published ONCE for pick_target.");

//     // 3️⃣ Trigger pick if not already in progress
//     if (!picking_in_progress_) {
//         picking_in_progress_ = true;
//         target_tf_name_ = "pick_target";

//         std::thread([this]() {
//             execute_pick_auto();
//             picking_in_progress_ = false;
//         }).detach();
//     }
//   }

// private:
//   std::mutex moveit_mutex_;
//   bool picking_in_progress_ = false;

//   std::string target_tf_name_;

//   std::shared_ptr<moveit::planning_interface::MoveGroupInterface> arm_group_;
//   std::shared_ptr<moveit::planning_interface::MoveGroupInterface> gripper_group_;

//   std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
//   std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
//   std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
//   rclcpp::Publisher<std_msgs::msg::String>::SharedPtr tf_name_pub_;

//   rclcpp_action::Server<Pick>::SharedPtr pick_server_;
//   rclcpp::Service<GripperCommand>::SharedPtr gripper_service_;

//   // -----------------------
//   // Startup pose
//   // -----------------------
//   void startupDetectPose()
//   {
//     std::lock_guard<std::mutex> lock(moveit_mutex_);
//     moveit::planning_interface::MoveGroupInterface::Plan plan;

//     RCLCPP_INFO(this->get_logger(), "Moving to detect pose...");
//     arm_group_->setNamedTarget("detect");
//     if (arm_group_->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS)
//       arm_group_->execute(plan);

//     gripper_group_->setNamedTarget("open");
//     if (gripper_group_->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS)
//       gripper_group_->execute(plan);

//     RCLCPP_INFO(this->get_logger(), "Detect pose reached.");
//   }

//   // -----------------------
//   // Gripper Service
//   // -----------------------
//   void handle_gripper(
//     const std::shared_ptr<GripperCommand::Request> req,
//     std::shared_ptr<GripperCommand::Response> res)
//   {
//     std::lock_guard<std::mutex> lock(moveit_mutex_);
//     moveit::planning_interface::MoveGroupInterface::Plan plan;

//     gripper_group_->setNamedTarget(req->command);

//     if (gripper_group_->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS &&
//         gripper_group_->execute(plan) == moveit::core::MoveItErrorCode::SUCCESS)
//     {
//       res->success = true;
//       res->message = "Gripper " + req->command;
//     }
//     else
//     {
//       res->success = false;
//       res->message = "Failed";
//     }
//   }

//   // -----------------------
//   // Action callbacks
//   // -----------------------
//   rclcpp_action::GoalResponse handle_pick_goal(
//     const rclcpp_action::GoalUUID &, std::shared_ptr<const Pick::Goal>)
//   {
//     return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
//   }

//   rclcpp_action::CancelResponse handle_pick_cancel(
//     const std::shared_ptr<GoalHandlePick>)
//   {
//     return rclcpp_action::CancelResponse::ACCEPT;
//   }

//   void handle_pick_accepted(const std::shared_ptr<GoalHandlePick> goal_handle)
//   {
//     std::thread([this, goal_handle]() { execute_pick(goal_handle); }).detach();
//   }

//   // -----------------------
//   // PICK LOGIC
//   // -----------------------
//   void execute_pick_auto()
//   {
//     std::lock_guard<std::mutex> lock(moveit_mutex_);

//     moveit::planning_interface::MoveGroupInterface::Plan plan;

//     geometry_msgs::msg::TransformStamped tf_goal;
//     try {
//       tf_goal = tf_buffer_->lookupTransform("base_link", target_tf_name_, tf2::TimePointZero);
//     } catch (const tf2::TransformException & ex) {
//       RCLCPP_ERROR(this->get_logger(), "TF lookup failed: %s", ex.what());
//       return;
//     }

//     // Move ABOVE
//     geometry_msgs::msg::Pose pose_above;
//     pose_above.position.x = tf_goal.transform.translation.x;
//     pose_above.position.y = tf_goal.transform.translation.y;
//     pose_above.position.z = tf_goal.transform.translation.z + 0.2;

//     tf2::Quaternion q;
//     q.setRPY(1.57, 1.3, 0);
//     pose_above.orientation = tf2::toMsg(q);

//     arm_group_->setPoseTarget(pose_above, "gripper_tcp");
//     if (arm_group_->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS)
//       arm_group_->execute(plan);

//     // Move DOWN
//     geometry_msgs::msg::Pose pose_grasp = pose_above;
//     pose_grasp.position.z = tf_goal.transform.translation.z + 0.15;

//     arm_group_->setPoseTarget(pose_grasp, "gripper_tcp");
//     if (arm_group_->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS)
//       arm_group_->execute(plan);

//     // Close Gripper
//     gripper_group_->setNamedTarget("close");
//     if (gripper_group_->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS)
//       gripper_group_->execute(plan);

//     // Lift
//     arm_group_->setPoseTarget(pose_above, "gripper_tcp");
//     arm_group_->plan(plan);
//     arm_group_->execute(plan);

//     // Return
//     arm_group_->setNamedTarget("detect");
//     arm_group_->plan(plan);
//     arm_group_->execute(plan);

//     RCLCPP_INFO(this->get_logger(), "Pick completed!");
//   }

//   void execute_pick(const std::shared_ptr<GoalHandlePick> /*goal_handle*/)
//   {
//     // Optional
//   }
// };

// // -----------------------
// // MAIN
// // -----------------------
// int main(int argc, char ** argv)
// {
//   rclcpp::init(argc, argv);

//   auto node = std::make_shared<AutoPickServer>(rclcpp::NodeOptions());
//   node->initMoveGroups();

//   // 🔥 Manual trigger example:
//   // Press Enter in terminal to trigger dummy TF + pick
//   std::thread([node]() {
//     while (rclcpp::ok()) {
//       std::cout << "\nPress Enter to simulate YOLO pick: ";
//       std::cin.get();
//       node->triggerDummyPickTFOnce();
//     }
//   }).detach();

//   rclcpp::spin(node);
//   rclcpp::shutdown();
//   return 0;
// }
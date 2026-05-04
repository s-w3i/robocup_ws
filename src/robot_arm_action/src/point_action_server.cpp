#include <memory>
#include <string>
#include <thread>
#include <mutex>
#include <cmath>

#include "geometry_msgs/msg/transform_stamped.hpp"
#include "geometry_msgs/msg/pose.hpp"

#include "moveit/move_group_interface/move_group_interface.h"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"

#include "robot_arm_action/action/point.hpp"

#include "tf2/exceptions.h"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

// ----------------------
using Point = robot_arm_action::action::Point;
using GoalHandlePoint = rclcpp_action::ServerGoalHandle<Point>;
// ----------------------

class PointActionServer : public rclcpp::Node
{
public:
  PointActionServer(const rclcpp::NodeOptions & options)
  : Node("point_action_server", options)
  {
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    point_server_ = rclcpp_action::create_server<Point>(
      this,
      "point_object",
      std::bind(&PointActionServer::handle_goal, this,
                std::placeholders::_1, std::placeholders::_2),
      std::bind(&PointActionServer::handle_cancel, this,
                std::placeholders::_1),
      std::bind(&PointActionServer::handle_accepted, this,
                std::placeholders::_1)
    );

    RCLCPP_INFO(this->get_logger(), "PointActionServer Ready");
  }

  void initMoveGroup()
  {
    arm_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
      shared_from_this(), "arm");

    // 🔥 IMPORTANT SETTINGS FOR FULL ARM MOTION
    arm_group_->setPlanningTime(5.0);
    arm_group_->setNumPlanningAttempts(10);
    arm_group_->setMaxVelocityScalingFactor(0.3);
    arm_group_->setGoalPositionTolerance(0.01);
    arm_group_->setGoalOrientationTolerance(0.2);
  }

private:
  std::mutex moveit_mutex_;

  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> arm_group_;

  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  rclcpp_action::Server<Point>::SharedPtr point_server_;

  // ===============================
  // ACTION CALLBACKS
  // ===============================
  rclcpp_action::GoalResponse handle_goal(
    const rclcpp_action::GoalUUID &,
    std::shared_ptr<const Point::Goal> goal)
  {
    if (goal->target_frame.empty()) {
      RCLCPP_ERROR(this->get_logger(), "Rejected: empty target_frame");
      return rclcpp_action::GoalResponse::REJECT;
    }

    RCLCPP_INFO(this->get_logger(),
      "Accepted goal: %s", goal->target_frame.c_str());

    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
  }

  rclcpp_action::CancelResponse handle_cancel(
    const std::shared_ptr<GoalHandlePoint>)
  {
    return rclcpp_action::CancelResponse::ACCEPT;
  }

  void handle_accepted(const std::shared_ptr<GoalHandlePoint> goal_handle)
  {
    std::thread([this, goal_handle]() {
      execute(goal_handle);
    }).detach();
  }

  // ===============================
  // ACTION EXECUTION
  // ===============================
  void execute(const std::shared_ptr<GoalHandlePoint> goal_handle)
  {
    std::lock_guard<std::mutex> lock(moveit_mutex_);

    auto result = std::make_shared<Point::Result>();
    const std::string target_frame = goal_handle->get_goal()->target_frame;

    bool success = execute_point_core(target_frame);

    if (success) {
      result->success = true;
      goal_handle->succeed(result);
      RCLCPP_INFO(this->get_logger(), "POINT SUCCESS");
    } else {
      result->success = false;
      goal_handle->abort(result);
      RCLCPP_ERROR(this->get_logger(), "POINT FAILED");
    }
  }

  // ===============================
  // 🔥 CORE POINTING LOGIC (FULL ARM YAW)
  // ===============================
  bool execute_point_core(const std::string & target_frame)
  {
    geometry_msgs::msg::TransformStamped tf_goal;

    try {
      tf_goal = tf_buffer_->lookupTransform(
        "base_link",
        target_frame,
        tf2::TimePointZero);
    }
    catch (const tf2::TransformException & ex) {
      RCLCPP_ERROR(this->get_logger(), "TF failed: %s", ex.what());
      return false;
    }

    // ---------------- FIXED ARM POSITION ----------------
    const double fixed_x = 0.5;
    const double fixed_z = 0.5;

    // target direction
    double dx = tf_goal.transform.translation.x - fixed_x;
    double dy = tf_goal.transform.translation.y;

    double yaw = atan2(dy, dx);

    geometry_msgs::msg::Pose target_pose;

    target_pose.position.x = fixed_x;
    target_pose.position.y = 0.0;
    target_pose.position.z = fixed_z;

    // ---------------- FULL ARM YAW ORIENTATION ----------------
    tf2::Quaternion q;
    q.setRPY(0, 0, yaw);
    target_pose.orientation = tf2::toMsg(q);

    // ---------------- MOVEIT CONFIG ----------------
    arm_group_->setStartStateToCurrentState();
    arm_group_->setPoseReferenceFrame("base_link");

    arm_group_->setPoseTarget(target_pose);

    moveit::planning_interface::MoveGroupInterface::Plan plan;

    if (arm_group_->plan(plan) != moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_ERROR(this->get_logger(), "Planning failed");
      return false;
    }

    if (arm_group_->execute(plan) != moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_ERROR(this->get_logger(), "Execution failed");
      return false;
    }

    return true;
  }
};

// ===============================
// MAIN
// ===============================
int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<PointActionServer>(rclcpp::NodeOptions());
  node->initMoveGroup();

  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}


//   bool execute_point_core(const std::string & target_frame)
//   {
//     geometry_msgs::msg::TransformStamped tf_goal;

//     try {
//       tf_goal = tf_buffer_->lookupTransform(
//         "base_link",
//         target_frame,
//         tf2::TimePointZero);
//     }
//     catch (const tf2::TransformException & ex) {
//       RCLCPP_ERROR(this->get_logger(), "TF failed: %s", ex.what());
//       return false;
//     }

//     auto current_pose = arm_group_->getCurrentPose("gripper_tcp").pose;

//     // ---------------- direction ----------------
//     double dx = tf_goal.transform.translation.x - current_pose.position.x;
//     double dy = tf_goal.transform.translation.y - current_pose.position.y;

//     double yaw = atan2(dy, dx);

//     // ---------------- pose ----------------
//     geometry_msgs::msg::Pose target_pose = current_pose;

//     target_pose.position.x = 0.5;
//     target_pose.position.z = 0.5;

//     tf2::Quaternion q;
//     q.setRPY(0, 0, yaw);
//     target_pose.orientation = tf2::toMsg(q);

//     // ---------------- planning ----------------
//     moveit::planning_interface::MoveGroupInterface::Plan plan;

//     arm_group_->setPoseTarget(target_pose, "gripper_tcp");

//     if (arm_group_->plan(plan) != moveit::core::MoveItErrorCode::SUCCESS) {
//       return false;
//     }

//     if (arm_group_->execute(plan) != moveit::core::MoveItErrorCode::SUCCESS) {
//       return false;
//     }

//     return true;
//   }
// };

// ===============================
// MAIN
// ===============================
// int main(int argc, char ** argv)
// {
//   rclcpp::init(argc, argv);

//   auto node = std::make_shared<PointActionServer>(rclcpp::NodeOptions());
//   node->initMoveGroup();

//   rclcpp::spin(node);
//   rclcpp::shutdown();
//   return 0;
// }

// old code

// #include <memory>
// #include <string>
// #include <thread>
// #include <mutex>
// #include <cmath>

// #include "geometry_msgs/msg/transform_stamped.hpp"
// #include "geometry_msgs/msg/pose.hpp"

// #include "rclcpp/rclcpp.hpp"
// #include "rclcpp_action/rclcpp_action.hpp"

// #include "moveit/move_group_interface/move_group_interface.h"

// #include "robot_arm_action/action/point.hpp"

// #include "tf2/exceptions.h"
// #include "tf2_ros/buffer.h"
// #include "tf2_ros/transform_listener.h"
// #include "tf2/LinearMath/Quaternion.h"
// #include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

// using Point = robot_arm_action::action::Point;
// using GoalHandlePoint = rclcpp_action::ServerGoalHandle<Point>;

// class PointActionServer : public rclcpp::Node
// {
// public:
//   PointActionServer(const rclcpp::NodeOptions & options)
//   : Node("point_action_server", options)
//   {
//     tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
//     tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

//     point_server_ = rclcpp_action::create_server<Point>(
//       this,
//       "point_object",
//       std::bind(&PointActionServer::handle_goal, this, std::placeholders::_1, std::placeholders::_2),
//       std::bind(&PointActionServer::handle_cancel, this, std::placeholders::_1),
//       std::bind(&PointActionServer::handle_accepted, this, std::placeholders::_1)
//     );

//     RCLCPP_INFO(this->get_logger(), "PointActionServer Ready");
//   }

//   void initMoveGroup()
//   {
//     arm_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
//       shared_from_this(), "arm");
//   }

// private:
//   std::mutex moveit_mutex_;

//   std::shared_ptr<moveit::planning_interface::MoveGroupInterface> arm_group_;

//   std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
//   std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

//   rclcpp_action::Server<Point>::SharedPtr point_server_;

//   // -----------------------
//   // ACTION CALLBACKS
//   // -----------------------
//   // Received goal from dummy
//   // rclcpp_action::GoalResponse handle_goal(
//   //   const rclcpp_action::GoalUUID &,
//   //   std::shared_ptr<const Point::Goal> goal)
//   // {
//   //   RCLCPP_INFO(this->get_logger(), "Received goal: %s", goal->target_frame.c_str());
//   //   return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
//   // }

//   // Received Goal from yolo
//   rclcpp_action::GoalResponse handle_goal(
//     const rclcpp_action::GoalUUID &,
//     std::shared_ptr<const Point::Goal> goal)
//   {
//   // Old logic
//   //   if (goal->target_frame.empty()) {
//   //     RCLCPP_ERROR(this->get_logger(), "Rejected: empty target_frame");
//   //     return rclcpp_action::GoalResponse::REJECT;
//   //   }

//   //   RCLCPP_INFO(this->get_logger(),
//   //     "Goal received (ignored, using fixed frame: empty_chair_1)");
  
//   //   RCLCPP_INFO(this->get_logger(),
//   //     "Received YOLO frame goal: %s",
//   //     goal->target_frame.c_str());
  
//   //   return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
//   // }
//     // ✅ MUST BE INSIDE FUNCTION
//     const std::string allowed_frame = "a_empty_chair_1";

//     if (goal->target_frame.empty()) {
//       RCLCPP_ERROR(this->get_logger(), "Rejected: empty target_frame");
//       return rclcpp_action::GoalResponse::REJECT;
//     }
//     // 🔥 STRICT FILTER (THIS IS THE KEY PART)
//     if (goal->target_frame != allowed_frame) {
//       RCLCPP_WARN(this->get_logger(),
//         "Rejected frame '%s' (only '%s' allowed)",
//         goal->target_frame.c_str(),
//         allowed_frame.c_str());
  
//       return rclcpp_action::GoalResponse::REJECT;
//     }
  
//     // ✅ ACCEPT ONLY THIS FRAME
//     RCLCPP_INFO(this->get_logger(),
//       "Accepted fixed target: %s",
//       allowed_frame.c_str());
  
//     return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
//   }

//   rclcpp_action::CancelResponse handle_cancel(
//     const std::shared_ptr<GoalHandlePoint>)
//   {
//     return rclcpp_action::CancelResponse::ACCEPT;
//   }

//   void handle_accepted(const std::shared_ptr<GoalHandlePoint> goal_handle)
//   {
//     std::thread([this, goal_handle]() { execute(goal_handle); }).detach();
//   }

//   // -----------------------
//   // EXECUTION
//   // -----------------------
//   void execute(const std::shared_ptr<GoalHandlePoint> goal_handle)
//   {
//     std::lock_guard<std::mutex> lock(moveit_mutex_);

//     auto result = std::make_shared<Point::Result>();
//     auto feedback = std::make_shared<Point::Feedback>();

//     // std::string target_frame = goal_handle->get_goal()->target_frame;
//     std::string target_frame = "a_empty_chair_1";

//     feedback->status = "Looking up TF...";
//     goal_handle->publish_feedback(feedback);

//     // geometry_msgs::msg::TransformStamped tf_goal;

//     // try {
//     //   bool got_tf = false;
//     //   for (int i = 0; i < 5; i++) {
//     //     try {
//     //       tf_goal = tf_buffer_->lookupTransform(
//     //         "base_link",
//     //         target_frame,
//     //         tf2::TimePointZero
//     //       );
//     //       got_tf = true;
//     //       break;
//     //     } catch (const tf2::TransformException & ex) {
//     //       RCLCPP_WARN(this->get_logger(),
//     //         "TF not ready, retrying... (%d/5)", i + 1);
//     //       rclcpp::sleep_for(std::chrono::milliseconds(200));
//     //     }
//     //   }

//     //   if (!got_tf) {
//     //     result->success = false;
//     //     result->message = "TF lookup failed for " + target_frame;
//     //     goal_handle->abort(result);
//     //     return;
//     //   }
//     // } catch (const tf2::TransformException & ex) {
//     //   result->success = false;
//     //   result->message = ex.what();
//     //   goal_handle->abort(result);
//     //   return;
//     // }

//     geometry_msgs::msg::TransformStamped tf_goal;

//     try {
//       tf_goal = tf_buffer_->lookupTransform(
//         "base_link",
//         "a_empty_chair_1",
//         tf2::TimePointZero
//       );
//     } catch (const tf2::TransformException & ex) {
//       result->success = false;
//       result->message = "TF 'a_empty_chair_1' not found: " + std::string(ex.what());
//       goal_handle->abort(result);
//       return;
//     }

//     feedback->status = "Computing pointing pose...";
//     goal_handle->publish_feedback(feedback);

//     // 🔥 Get current pose
//     auto current_pose = arm_group_->getCurrentPose("gripper_tcp").pose;

//     double fixed_x = current_pose.position.x;
//     double fixed_z = current_pose.position.z;

//     double target_x = tf_goal.transform.translation.x;
//     double target_y = tf_goal.transform.translation.y;

//     double dx = target_x - fixed_x;
//     double dy = target_y - current_pose.position.y;

//     double yaw = atan2(dy, dx);

//     geometry_msgs::msg::Pose target_pose = current_pose;

//     target_pose.position.x = 0.8; //fixed_x;
//     target_pose.position.z = 0.5; //fixed_z;

//     tf2::Quaternion q;
//     q.setRPY(0, 0, yaw);
//     target_pose.orientation = tf2::toMsg(q);

//     feedback->status = "Planning...";
//     goal_handle->publish_feedback(feedback);

//     moveit::planning_interface::MoveGroupInterface::Plan plan;

//     arm_group_->setPoseTarget(target_pose, "gripper_tcp");

//     if (arm_group_->plan(plan) != moveit::core::MoveItErrorCode::SUCCESS)
//     {
//       result->success = false;
//       result->message = "Planning failed";
//       goal_handle->abort(result);
//       return;
//     }

//     feedback->status = "Executing...";
//     goal_handle->publish_feedback(feedback);

//     arm_group_->execute(plan);

//     result->success = true;
//     result->message = "Pointing completed";

//     goal_handle->succeed(result);
//   }
// };

// // -----------------------
// // MAIN
// // -----------------------
// int main(int argc, char ** argv)
// {
//   rclcpp::init(argc, argv);

//   auto node = std::make_shared<PointActionServer>(rclcpp::NodeOptions());
//   node->initMoveGroup();

//   rclcpp::spin(node);
//   rclcpp::shutdown();
//   return 0;
// }


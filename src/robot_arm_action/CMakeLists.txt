cmake_minimum_required(VERSION 3.8)
project(robot_arm_action)

find_package(ament_cmake REQUIRED)

find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)

find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

find_package(moveit_ros_planning_interface REQUIRED)

find_package(rosidl_default_generators REQUIRED)
find_package(rosidl_default_runtime REQUIRED)

find_package(visualization_msgs REQUIRED)

# =========================
# ACTIONS
# =========================
rosidl_generate_interfaces(${PROJECT_NAME}
  "action/MoveArm.action"
  "action/Pick.action"
  "action/Point.action"
  "srv/ArmPose.srv"
  "srv/GripperCommand.srv"
  DEPENDENCIES geometry_msgs std_msgs
)

# 🔥 Export runtime
ament_export_dependencies(rosidl_default_runtime)

# 🔧 Include generated headers 
include_directories(include)

# =========================
# SERVER
# =========================
add_executable(point_action_server
  src/point_action_server.cpp
)

add_executable(pick_action_server
  src/pick_action_server.cpp
)

ament_target_dependencies(pick_action_server
  rclcpp
  rclcpp_action
  geometry_msgs
  std_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  moveit_ros_planning_interface
)

ament_target_dependencies(point_action_server
  rclcpp
  rclcpp_action
  geometry_msgs
  moveit_ros_planning_interface
)

add_executable(move_arm_server src/move_arm_server.cpp)
ament_target_dependencies(move_arm_server
  rclcpp
  rclcpp_action
  geometry_msgs
  tf2_ros
  moveit_ros_planning_interface
)

add_executable(move_arm_client_tf src/move_arm_client_tf.cpp)
ament_target_dependencies(move_arm_client_tf
  rclcpp
  rclcpp_action
  geometry_msgs
  tf2_ros
  tf2_geometry_msgs
  visualization_msgs
)

add_executable(dummy_tf_pub src/dummy_tf.cpp)
ament_target_dependencies(dummy_tf_pub rclcpp tf2_ros geometry_msgs)
install(TARGETS
  dummy_tf_pub
  DESTINATION lib/${PROJECT_NAME})

# NEW WAY (IMPORTANT FIX)
rosidl_get_typesupport_target(cpp_typesupport_target
  ${PROJECT_NAME} "rosidl_typesupport_cpp"
)

rosidl_target_interfaces(pick_action_server
  ${PROJECT_NAME} "rosidl_typesupport_cpp"
)

target_link_libraries(point_action_server "${cpp_typesupport_target}")

# =========================
# TF NODE
# =========================
add_executable(dummy_point_tf
  src/point_dummy_tf.cpp
)

ament_target_dependencies(dummy_point_tf
  rclcpp
  tf2
  tf2_ros
  tf2_geometry_msgs
)


target_link_libraries(dummy_point_tf "${cpp_typesupport_target}")
target_link_libraries(move_arm_server ${cpp_typesupport_target})
target_link_libraries(move_arm_client_tf ${cpp_typesupport_target})
target_link_libraries(pick_action_server "${cpp_typesupport_target}")

# =========================
# INSTALL
# =========================
install(TARGETS
  point_action_server
  dummy_point_tf
  dummy_tf_pub
  move_arm_server
  move_arm_client_tf
  pick_action_server
  DESTINATION lib/${PROJECT_NAME}
)

# ======================
# 🧪 Testing
# ======================

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()


ament_package()

# cmake_minimum_required(VERSION 3.8)
# project(robot_arm_action)

# if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#   add_compile_options(-Wall -Wextra -Wpedantic)
# endif()

# # 🔧 Dependencies
# find_package(ament_cmake REQUIRED)
# find_package(rclcpp REQUIRED)
# find_package(rclcpp_action REQUIRED)
# find_package(geometry_msgs REQUIRED)
# find_package(tf2_ros REQUIRED)
# find_package(tf2_geometry_msgs REQUIRED)
# find_package(visualization_msgs REQUIRED)
# find_package(moveit_ros_planning_interface REQUIRED)
# find_package(rosidl_default_generators REQUIRED)

# # 🔥 Generate Action
# rosidl_generate_interfaces(${PROJECT_NAME}
# "srv/ArmPose.srv"
# "srv/GripperCommand.srv" 
# "action/Pick.action" 
# "action/MoveArm.action"
#   DEPENDENCIES geometry_msgs
# )


# # 🔥 Export runtime
# ament_export_dependencies(rosidl_default_runtime)

# # 🔧 Include generated headers
# include_directories(include)

# # ======================
# # 🚀 Executables
# # ======================

# add_executable(move_arm_server src/move_arm_server.cpp)
# ament_target_dependencies(move_arm_server
#   rclcpp
#   rclcpp_action
#   geometry_msgs
#   tf2_ros
#   moveit_ros_planning_interface
# )

# add_executable(move_arm_client_tf src/move_arm_client_tf.cpp)
# ament_target_dependencies(move_arm_client_tf
#   rclcpp
#   rclcpp_action
#   geometry_msgs
#   tf2_ros
#   tf2_geometry_msgs
#   visualization_msgs
# )

# add_executable(dummy_tf_pub src/dummy_tf_publisher.cpp)
# ament_target_dependencies(dummy_tf_publisher rclcpp tf2_ros geometry_msgs)
# install(TARGETS
#   dummy_tf_publisher
#   DESTINATION lib/${PROJECT_NAME})

# # 🔥 Link action interfaces properly
# rosidl_get_typesupport_target(cpp_typesupport_target
#   ${PROJECT_NAME} "rosidl_typesupport_cpp")

# target_link_libraries(move_arm_server ${cpp_typesupport_target})
# target_link_libraries(move_arm_client_tf ${cpp_typesupport_target})

# # ======================
# # 📦 Install
# # ======================

# install(TARGETS
#   move_arm_server
#   move_arm_client_tf
#   DESTINATION lib/${PROJECT_NAME}
# )

# # ======================
# # 🧪 Testing
# # ======================

# if(BUILD_TESTING)
#   find_package(ament_lint_auto REQUIRED)
#   set(ament_cmake_copyright_FOUND TRUE)
#   set(ament_cmake_cpplint_FOUND TRUE)
#   ament_lint_auto_find_test_dependencies()
# endif()

# ament_package()


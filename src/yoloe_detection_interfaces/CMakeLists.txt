cmake_minimum_required(VERSION 3.8)
project(yoloe_detection_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Detection3D.msg"
  "msg/Detection3DArray.msg"
  "msg/PeopleTrack2D.msg"
  "msg/PeopleTrack2DArray.msg"
  "msg/PeopleTrack3D.msg"
  "msg/PeopleTrack3DArray.msg"
  "srv/DetectObjectPrompt.srv"
  "srv/SetTracking.srv"
  DEPENDENCIES geometry_msgs
  std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()

cmake_minimum_required(VERSION 3.8)
project(simple_follower_ros2)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/FollowerPosition.msg"
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY param parameters
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY rviz
  DESTINATION share/${PROJECT_NAME}
)

install(FILES package.xml
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS simple_follower_ros2/line_follow.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME line_follow
)

install(PROGRAMS simple_follower_ros2/laserfollower.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME laserfollower
)

install(PROGRAMS simple_follower_ros2/visualTracker.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME visualtracker
)

install(PROGRAMS simple_follower_ros2/visualFollower.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME visualfollow
)

install(PROGRAMS simple_follower_ros2/adjust_hsv.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME adjust_hsv
)

install(PROGRAMS simple_follower_ros2/laserTracker.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME lasertracker
)

install(PROGRAMS simple_follower_ros2/ar_follow.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME arfollower
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()

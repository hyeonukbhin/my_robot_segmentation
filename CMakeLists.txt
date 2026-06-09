cmake_minimum_required(VERSION 3.0.2)
project(my_robot_segmentation)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  cv_bridge
)

catkin_package()

# scripts 폴더 안의 파이썬 파일들을 ROS 실행 파일로 등록
catkin_install_python(PROGRAMS
  scripts/segmentation_node.py
  scripts/inference_engine.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
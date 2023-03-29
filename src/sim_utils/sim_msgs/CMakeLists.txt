cmake_minimum_required(VERSION 3.0.2)
project(sim_msgs)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++14)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  message_generation
)


## Generate messages in the 'msg' folder
add_message_files(
  FILES
  sim_rccmd.msg
  task_state.msg
)



## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
)


catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES sim_msgs
 CATKIN_DEPENDS roscpp rospy std_msgs message_runtime nav_msgs
#  DEPENDS system_lib
)


include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)


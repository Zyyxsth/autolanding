cmake_minimum_required(VERSION 2.8.3)
project(uavsensor)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11 -march=native -DEIGEN_DONT_PARALLELIZE")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall")

find_package(catkin REQUIRED COMPONENTS
    roscpp
    rospy
    std_msgs
    message_generation
    serial
    # quadrotor_msgs
    # quadrotor_common
    nav_msgs
    )

# find_package(catkin REQUIRED genmsg actionlib_msgs actionlib)
add_message_files(
  FILES
  m3508.msg
)
generate_messages(DEPENDENCIES  std_msgs )
# generate_messages(
#     DEPENDENCIES
#     std_msgs
#     geometry_msgs
#     nav_msgs
# )


set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3  REQUIRED)
#find_package(yaml-cpp  REQUIRED)
include_directories(
    include
    ${catkin_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    #${YAML_CPP_INCLUDE_DIR}
    )

catkin_package(
    )


add_executable(relayctrl_node src/relayctrl_node.cpp)
target_link_libraries(relayctrl_node ${catkin_LIBRARIES} )


add_executable(tfluna_node src/tfluna_node.cpp)
target_link_libraries(tfluna_node ${catkin_LIBRARIES} )



add_executable(rccmd_node src/rccmd_node.cpp)
target_link_libraries(rccmd_node ${catkin_LIBRARIES} )




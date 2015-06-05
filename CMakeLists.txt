cmake_minimum_required(VERSION 2.8.3)
project(dense_vo)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -Wall -std=c++11 -Ofast -pipe -march=native -mtune=native")
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp -Ofast -march=native -mtune=native -pipe -Wall")
SET(OpenCV_INCLUDE_DIRS "/usr/include/opencv;/usr/include/opencv2")

find_package(catkin REQUIRED COMPONENTS
  
  roscpp
  rospy
  std_msgs
  geometry_msgs
  sensor_msgs 
  cv_bridge
  image_transport
  message_generation
  camera_info_manager 
  dynamic_reconfigure
  cmake_modules
)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)

find_package(libvisensor 1.1.0 REQUIRED)

find_package(OpenCV REQUIRED)

add_message_files(
  DIRECTORY msg
  FILES visensor_imu.msg 
  visensor_time_host.msg
  visensor_calibration.msg
)

add_service_files(
	FILES
	visensor_calibration_service.srv
)

generate_messages(DEPENDENCIES geometry_msgs)

if(NOT DEFINED CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif(NOT DEFINED CMAKE_BUILD_TYPE)

generate_dynamic_reconfigure_options(cfg/dense_vo_node.cfg)


## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend and a run_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependencies might have been
##     pulled in transitively but can be declared for certainty nonetheless:
##     * add a build_depend tag for "message_generation"
##     * add a run_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     catkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below	
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

## Generate messages in the 'msg' folder
# add_message_files(
#   FILES
#   Message1.msg
#   Message2.msg
# )

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
# generate_messages(
#   DEPENDENCIES
#   std_msgs
# )

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
  INCLUDE_DIRS include ${catkin_INCLUDE_DIRS}
  CATKIN_DEPENDS
  roscpp 
  sensor_msgs 
  cv_bridge 
  std_msgs 
  image_transport 
  camera_info_manager 
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(include)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
#  ${OPENNI2_INCLUDE}
  ${libvisensor_INCLUDE_DIRS}
)



## Declare a cpp library
# add_library(IROS2015
#   src/${PROJECT_NAME}/IROS2015.cpp
# )

## Declare a cpp executable
add_executable(dense_vo_node src/denseVO.cpp)
add_executable(collect_data_node src/collectData.cpp)

#add_executable(dense_vo_node src/denseVO.cpp)

## Add cmake target dependencies of the executable/library
## as an example, message headers may need to be generated before nodes
# add_dependencies(IROS2015_node IROS2015_generate_messages_cpp)

add_dependencies(dense_vo_node ${${PROJECT_NAME}_EXPORTED_TARGETS}})
add_dependencies(collect_data_node ${${PROJECT_NAME}_EXPORTED_TARGETS}})

## Specify libraries to link a library or executable target against
target_link_libraries(dense_vo_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${libvisensor_LIBRARIES}
#  OpenNI2
 # ${Boost_FILESYSTEM_LIBRARY} 
 # ${Boost_SYSTEM_LIBRARY}
)

target_link_libraries(collect_data_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${libvisensor_LIBRARIES}
#  OpenNI2
 # ${Boost_FILESYSTEM_LIBRARY} 
 # ${Boost_SYSTEM_LIBRARY}
)

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables and/or libraries for installation
# install(TARGETS IROS2015 IROS2015_node
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_IROS2015.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)

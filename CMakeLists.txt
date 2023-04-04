cmake_minimum_required(VERSION 3.0.2)
project(inference_package)
#list(APPEND CMAKE_PREFIX_PATH "/home/rio/libtorch")
#list(APPEND CMAKE_PREFIX_PATH "/home/rio/libtorch/share/cmake")
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  roscpp
  sensor_msgs
  std_msgs
)
find_package(OpenCV REQUIRED COMPONENTS 
    highgui 
    dnn
    core 
    imgproc 
    video 
    calib3d 
    features2d
    PATHS "/home/rio/opencv"
)
find_package(Torch REQUIRED PATHS "/home/rio/libtorch" )

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES inference_package
#  CATKIN_DEPENDS cv_bridge image_transport roscpp sensor_msgs std_msgs
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
   ${OpenCV_INCLUDE_DIRS}
)
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/src/*.cpp")
file(GLOB_RECURSE HEADERS CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/src/*.hxx")
add_executable(inference_node ${SOURCES}  )
target_link_libraries(inference_node ${OpenCV_LIBS} ${catkin_LIBRARIES} ${TORCH_LIBRARIES} )
cmake_minimum_required(VERSION 3.0.2)
project(inference_package)

find_package(OpenCV REQUIRED COMPONENTS 
    highgui 
    dnn
    core 
    imgproc 
    video 
    calib3d 
    features2d
    PATHS "."
)
#find_package(Torch REQUIRED 
#    PATHS "." 
#)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
#include_directories(
   #${OpenCV_INCLUDE_DIRS}
#)

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/src/*.cpp")
file(GLOB_RECURSE HEADERS CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/src/*.hxx")

add_executable(inference_node ${SOURCES}  )

target_link_libraries(inference_node ${OpenCV_LIBS} ${TORCH_LIBRARIES} )

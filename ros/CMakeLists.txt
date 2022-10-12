cmake_minimum_required(VERSION 3.0.2)
project(hybridnets_cpp)

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/usr/include/aarch64-linux-gnu/)
link_directories(/usr/lib/aarch64-linux-gnu/)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -g -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

FIND_PACKAGE(CUDA REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  image_transport
  cv_bridge
)
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})
catkin_package(
  INCLUDE_DIRS include
)
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  /usr/local/cuda-11.4/targets/aarch64-linux/include
  ${OpenCV_INCLUDE_DIRS}
)
link_directories(/usr/local/cuda-11.4/targets/aarch64-linux/lib)
link_directories(/usr/local/lib)
cuda_add_executable(hybridnets_cpp src/detector.cpp src/main.cpp src/utils.cpp src/prior_bbox_256x384_sim.cpp)
target_link_libraries(hybridnets_cpp nvinfer)
target_link_libraries(hybridnets_cpp cudart)
target_include_directories(hybridnets_cpp PRIVATE onnx/include)
target_compile_features(hybridnets_cpp PRIVATE cxx_std_11)
target_link_libraries (hybridnets_cpp ${OpenCV_LIBS} ${catkin_LIBRARIES})
target_link_libraries(hybridnets_cpp libonnxruntime.so)
link_directories(${CMAKE_BINARY_DIR})
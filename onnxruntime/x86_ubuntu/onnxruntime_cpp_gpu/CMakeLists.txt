cmake_minimum_required(VERSION 3.14)
project(deploy_test)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -ldl -pthread")

# find opencv
find_package(OpenCV 4.4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# find onnxruntime cpu
include_directories(/root/lib/onnxruntime-linux-x64-gpu-1.5.2/include)
set(ONNXRUNTIME_LIB_PATH "/root/lib/onnxruntime-linux-x64-gpu-1.5.2/lib")
file(GLOB LIBS "${ONNXRUNTIME_LIB_PATH}/*")

add_executable(demo main.cpp)
target_link_libraries(demo ${LIBS}   ${OpenCV_LIBS})


cmake_minimum_required(VERSION "3.16")

# Force OpenCV support
set(DEPTHAI_OPENCV_SUPPORT ON)

# DepthAI Library
add_subdirectory(depthai-core EXCLUDE_FROM_ALL)

# Project declaration
project(oakd_orbslam3)
set(CMAKE_BUILD_TYPE Debug)

# ORB_SLAM3 Library
set(ORB_SLAM3_DIR "./ORB_SLAM3")

# Force the use of system OpenCV installation with ximgproc support
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "/usr/share/cmake-3.22/Modules")
set(OpenCV_DIR "/usr/lib/x86_64-linux-gnu/cmake/opencv4")
find_package(OpenCV 4.0 REQUIRED)

if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV >= 4 not found.")
endif()
find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)

find_library(ORB_SLAM3_LIBS ORB_SLAM3 HINTS "${ORB_SLAM3_DIR}/lib")

include_directories(
    ${ORB_SLAM3_DIR}
    ${ORB_SLAM3_DIR}/include
    ${ORB_SLAM3_DIR}/include/CameraModels
    ${ORB_SLAM3_DIR}/Thirdparty/Sophus
    ${EIGEN3_INCLUDE_DIR}
    ${Pangolin_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/utility
)

# Output dirs
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Target name
set(TARGET_NAME oakd_orbslam3)

# Add main target
add_executable(${TARGET_NAME}
    src/main.cpp
    src/util.cpp
    utility/utility.cpp
)
target_link_libraries(${TARGET_NAME} PUBLIC
    ${OpenCV_LIBS}
    depthai-core
    depthai::opencv
    ${EIGEN3_LIBS}
    ${Pangolin_LIBRARIES}
    ${ORB_SLAM3_LIBS}
    -lboost_serialization
    -lcrypto
)

# Add replay executable
set(REPLAY_TARGET_NAME replay)
add_executable(${REPLAY_TARGET_NAME}
    src/replay.cpp
)
target_link_libraries(${REPLAY_TARGET_NAME} PUBLIC
    ${OpenCV_LIBS}
)
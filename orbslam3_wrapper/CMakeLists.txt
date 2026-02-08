cmake_minimum_required(VERSION 3.14)
project(orbslam3_python LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

#  pybind11 
find_package(pybind11 REQUIRED)

#  OpenCV (must match the one ORB_SLAM3 was built against) ─
find_package(OpenCV 4.4 REQUIRED)

#  Eigen3 
find_package(Eigen3 3 REQUIRED)

#  ORB_SLAM3 paths 
set(ORBSLAM3_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../ORB_SLAM3)
set(ORBSLAM3_LIB ${ORBSLAM3_DIR}/lib/libORB_SLAM3.so)

#  Build the Python module ─
pybind11_add_module(orbslam3_python src/bind.cpp)

# Suppress OpenGL 4.3+ functions that Pangolin headers reference but we don't use
target_compile_definitions(orbslam3_python PRIVATE
    GL_GLEXT_PROTOTYPES
)

target_include_directories(orbslam3_python PRIVATE
    ${ORBSLAM3_DIR}/include
    ${ORBSLAM3_DIR}/include/CameraModels
    ${ORBSLAM3_DIR}
    ${ORBSLAM3_DIR}/Thirdparty/Sophus
    ${ORBSLAM3_DIR}/Thirdparty/DBoW2/lib
    ${ORBSLAM3_DIR}/Thirdparty/g2o
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(orbslam3_python PRIVATE
    ${ORBSLAM3_LIB}
    ${OpenCV_LIBS}
)

# Set the output directory so the .so file lands next to ORB_SLAM3
set_target_properties(orbslam3_python PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../
)

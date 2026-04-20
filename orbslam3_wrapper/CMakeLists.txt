# orbslam3_wrapper — pybind11 module `orbslam3_python` wrapping ORB-SLAM3.
# Build ORB_SLAM3 first (../ORB_SLAM3/lib/libORB_SLAM3.so), then (docs/slam.md):
#   cd orbslam3_wrapper && mkdir -p build && cd build
#   cmake .. -Dpybind11_DIR=$(pybind11-config --cmakedir) -DCMAKE_BUILD_TYPE=Release && make -j3
# The module is written to the pilot_core repo root.

cmake_minimum_required(VERSION 3.14)
project(orbslam3_python LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- pybind11 ---
find_package(pybind11 REQUIRED)

# --- OpenCV (must match the version ORB_SLAM3 was built against) ---
find_package(OpenCV 4.4 REQUIRED)

# --- Eigen3 ---
find_package(Eigen3 3 REQUIRED)

# --- ORB_SLAM3 paths ---
set(ORBSLAM3_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../ORB_SLAM3)
set(ORBSLAM3_LIB ${ORBSLAM3_DIR}/lib/libORB_SLAM3.so)

# --- Python module ---
pybind11_add_module(orbslam3_python src/bind.cpp)

# Declare the OpenGL extension prototypes referenced by Pangolin headers
# (pulled in via System.h; the viewer itself is not used)
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

# Write the .so to the repo root (next to start.py) so it is importable there
set_target_properties(orbslam3_python PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../
)

# C++ Project Setup - OAK-D SLAM

This document provides comprehensive setup instructions for building the C++ components of the OAK-D SLAM project.

## Table of Contents
- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Dependencies](#dependencies)
- [Setup Instructions](#setup-instructions)
- [Build Process](#build-process)
- [Executables](#executables)
- [Troubleshooting](#troubleshooting)

## Overview

The project consists of two main C++ executables:
- **oakd_orbslam3**: Main SLAM application integrating OAK-D cameras with ORB-SLAM3
- **replay**: Utility for replaying recorded data

## System Requirements

- **OS**: Ubuntu 18.04+ or similar Linux distribution
- **CMake**: Version 3.16 or higher
- **C++ Standard**: C++14 or higher
- **Python**: Python 3.6+ with NumPy
- **Memory**: At least 8GB RAM recommended
- **Storage**: At least 20GB free space for dependencies

## Dependencies

### Core Libraries

1. **OpenCV 4.0+**
   - Must include ximgproc module
   - Compiled with C++14 support

2. **Eigen3 3.1.0+**
   - Linear algebra library

3. **Pangolin**
   - For 3D visualization

4. **Boost**
   - Serialization module required

5. **OpenSSL**
   - Crypto library

### SLAM Library

6. **ORB-SLAM3**
   - Visual SLAM library (included as submodule)
   - Requires additional dependencies (see ORB-SLAM3 documentation)

### Camera SDK

7. **DepthAI Core**
   - OAK-D camera integration (included as submodule)

## Setup Instructions

### 1. Clone Repository and Submodules

```bash
# Clone the main repository
git clone <repository-url> pilot_core
cd pilot_core

# Initialize and update submodules
git submodule update --init --recursive

# Initialize DepthAI submodules
cd depthai-core
git submodule update --init --recursive
cd ..
```

### 2. Install System Dependencies

#### Ubuntu/Debian:
```bash
# Update package list
sudo apt update

# Install essential build tools
sudo apt install -y cmake build-essential pkg-config

# Install OpenCV 4.0+
sudo apt install -y libopencv-dev libopencv-contrib-dev

# Install Eigen3
sudo apt install -y libeigen3-dev

# Install Pangolin dependencies
sudo apt install -y libglew-dev libpython3-dev
git clone https://github.com/stevenlovegrove/Pangolin.git
cd Pangolin
mkdir build && cd build
cmake ..
make -j4
sudo make install
cd ../..

# Install Boost
sudo apt install -y libboost-serialization-dev

# Install OpenSSL
sudo apt install -y libssl-dev

# Install Python development headers
sudo apt install -y python3-dev python3-numpy
```

### 3. Build ORB-SLAM3

```bash
cd ORB_SLAM3

# Install ORB-SLAM3 dependencies
sudo apt install -y libglew-dev

# Build ORB-SLAM3
chmod +x build.sh
./build.sh

cd ..
```

**Note**: ORB-SLAM3 may require additional setup. Refer to `ORB_SLAM3/README.md` for detailed instructions.

### 4. Setup Python Modules

```bash
# Make setup script executable and run
chmod +x setup.sh
./setup.sh
```

This will clone the required Python modules:
- `pilot_ingest`
- `pilot_vision` 
- `pilot_controller`

## Build Process

### 1. Create Build Directory

```bash
mkdir -p build
cd build
```

### 2. Configure with CMake

```bash
# Configure the project
cmake ..

# Or with specific options
cmake -DCMAKE_BUILD_TYPE=Release ..
```

### 3. Build the Project

```bash
# Build with multiple cores (adjust -j flag based on your CPU)
make -j4

# Or build specific targets
make oakd_orbslam3
make replay
```

### Alternative Build Script

You can also use the provided build script:

```bash
chmod +x build.sh
./build.sh
```

## Executables

After successful compilation, you'll find the executables in the `build` directory:

### oakd_orbslam3
Main SLAM application that integrates OAK-D camera data with ORB-SLAM3.

**Usage**:
```bash
./oakd_orbslam3 [config_file]
```

### replay
Utility for replaying recorded data.

**Usage**:
```bash
./replay [input_file]
```

## Configuration

- **Main config**: `config.yaml`
- **SLAM settings**: `oak_d_orbslam_settings.yaml`

## Troubleshooting

### Common Issues

1. **CMake version too old**
   ```bash
   # Install newer CMake
   wget https://github.com/Kitware/CMake/releases/download/v3.20.0/cmake-3.20.0-Linux-x86_64.sh
   chmod +x cmake-3.20.0-Linux-x86_64.sh
   sudo ./cmake-3.20.0-Linux-x86_64.sh --prefix=/usr/local --skip-license
   ```

2. **OpenCV not found**
   ```bash
   # Check OpenCV installation
   pkg-config --modversion opencv4
   
   # If not found, install from source or use conda
   ```

3. **ORB-SLAM3 compilation errors**
   - Ensure all dependencies are installed
   - Check C++ standard compatibility
   - May need to modify source for newer compilers

4. **Pangolin not found**
   ```bash
   # Set PKG_CONFIG_PATH if needed
   export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/lib/pkgconfig
   ```

5. **Python integration issues**
   ```bash
   # Ensure Python development headers are installed
   sudo apt install python3-dev python3-numpy
   ```

6. **Submodule issues**
   ```bash
   # Force update submodules
   git submodule update --init --recursive --force
   ```

### Debug Build

For debugging, use a debug build:

```bash
cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make -j4
```

### Verbose Build

To see detailed compilation commands:

```bash
make VERBOSE=1
```

## Additional Resources

- [ORB-SLAM3 Documentation](ORB_SLAM3/README.md)
- [DepthAI Documentation](depthai-core/README.md)
- [Python Components README](README.md)

## Support

For issues related to:
- **ORB-SLAM3**: Check the ORB-SLAM3 repository and documentation
- **DepthAI**: Visit the DepthAI documentation and community forums
- **Project-specific issues**: Create an issue in this repository

---

**Note**: This project integrates multiple complex libraries. Build times can be significant (30+ minutes) depending on your system specifications.
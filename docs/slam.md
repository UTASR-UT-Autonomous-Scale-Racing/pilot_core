# SLAM (ORB-SLAM3)

`SLAMWorker` runs ORB-SLAM3 RGB-D through `orbslam3_wrapper/` (`--enable-slam`).

## Build

About 30 minutes on 4 cores; outputs are git-ignored and built per machine.
System packages (Ubuntu 22.04):

```bash
sudo apt-get install -y build-essential cmake git pkg-config \
    libeigen3-dev libepoxy-dev libglew-dev libgl1-mesa-dev libwayland-dev \
    libopencv-dev libboost-serialization-dev libssl-dev
```

Pangolin (skip if already installed):

```bash
cd /tmp && git clone --depth 1 https://github.com/stevenlovegrove/Pangolin.git
cd Pangolin && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF
make -j3 && sudo make install && sudo ldconfig
```

ORB-SLAM3 and the Python bindings (from the repo root, with `.venv` active):

```bash
git submodule update --init ORB_SLAM3
cd ORB_SLAM3
(cd Thirdparty/DBoW2 && mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j3)
(cd Thirdparty/g2o   && mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j3)
(cd Vocabulary && tar xf ORBvoc.txt.tar.gz)
mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j3
cd ../../orbslam3_wrapper && mkdir -p build && cd build
cmake .. -Dpybind11_DIR=$(pybind11-config --cmakedir) -DCMAKE_BUILD_TYPE=Release && make -j3
```

On a machine with an NVIDIA GPU, build ORB-SLAM3 with CUDA (the `feature/cuda-orb-extraction-stereo` branch of the
ORB_SLAM3 fork; see its README): pyramid, ORB extraction and stereo matching run on the GPU.

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
         -DCMAKE_CUDA_ARCHITECTURES=87      # Jetson Orin; 89 for an RTX 40xx
```

It is used automatically; `ORB_SLAM3_CUDA=0` switches back to the CPU. `SLAMWorker` logs
the mean tracking time every 30 frames, and `scripts/slam_replay.py` replays a TUM RGB-D
sequence through it (tracking time and ATE). On an RTX 4080 / i7-13700, RGB-D tracking
takes 8.0-8.7 ms per frame with CUDA vs 12.3-13.1 ms on the CPU at the same accuracy
(TUM fr1_xyz / fr1_desk).

Check with `python -c "import orbslam3_python"`. The wrapper has an RPATH to the
ORB-SLAM3 libraries, so `LD_LIBRARY_PATH` isn't needed. Without the build `SLAMWorker`
refuses to start: use a map-free control mode, or the AutoDRIVE sim's ground-truth pose.

## Camera settings

`orbslam_oakd_rgbd.yaml` holds the intrinsics of one OAK-D unit; regenerate it when
the camera changes. `CameraWorker` warns if the connected camera doesn't match.

```bash
python scripts/gen_orbslam_settings.py --config config.yaml
```

## Mapping lap

1. Start with `--enable-slam` and keep the car still on the floor for a second:
   SLAM fits the floor plane and publishes in a levelled frame (it logs the camera
   pitch and height; put the height in `depth_ftg.camera_height`).
2. Drive one lap by hand (disarmed, RC transmitter).
3. ORB-SLAM3 closes the loop (or press `l`). After its global bundle adjustment,
   `MappingWorker` extracts the boundaries, SLAM switches to localization and you can
   arm with `a`.

If SLAM starts a new map mid-lap, mapping starts over. Tracking needs texture; while
it is lost no poses are published and the watchdog stops the car.

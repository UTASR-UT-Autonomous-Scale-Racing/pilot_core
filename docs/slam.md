# SLAM (ORB-SLAM3)

`SLAMWorker` runs ORB-SLAM3 in RGB-D mode through the pybind11 bindings in
`orbslam3_wrapper/`. It publishes camera-to-world poses to `pose_buf` while tracking
is good, and the map's point cloud for `MappingWorker`. Enable it with `--enable-slam`
(or `enable_slam: true`).

## Build

Allow about 30 minutes on a 4-core CPU. The build outputs (`ORB_SLAM3/lib/`,
`orbslam3_python*.so`) are git-ignored and must be built on each machine.

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

Check it:

```bash
python -c "import orbslam3_python; print('ok')"
```

The wrapper is built with an RPATH to `ORB_SLAM3/lib` and its third-party libraries, so
`LD_LIBRARY_PATH` is not needed (`scripts/run.sh` still sets it, which does no harm).
Without the build, `SLAMWorker` refuses to start unless the run is synthetic, where it
falls back to a stub that publishes identity poses.

```bash
scripts/run.sh --config config_cpu.yaml --enable-slam
```

## Camera settings

ORB-SLAM3 needs the camera intrinsics at the resolution `CameraWorker` publishes.
`orbslam_oakd_rgbd.yaml` is generated from the connected OAK-D's factory calibration.
Calibration is per unit, so regenerate it whenever the camera is changed:

```bash
python scripts/gen_orbslam_settings.py --config config.yaml
```

The generated file matches what `CameraWorker` sends: undistorted RGB frames, RGB channel
order, depth in metres (`DepthMapFactor: 1.0`). It records the camera's device id
(`Pilot.deviceId`); `CameraWorker` warns at startup if the connected camera or its
intrinsics differ from the file by more than 2%.

## World frame and timestamps

ORB-SLAM3's world frame is the first camera frame, which is pitched toward the floor.
During the first frames `SLAMWorker` fits the floor plane in the depth image (keep the
car still on the floor while it starts) and from then on publishes poses and map points
in a levelled frame: same origin, Y down, the floor parallel to the BEV plane. It logs
the fitted pitch, roll and camera height. Set the measured height in
`depth_ftg.camera_height`, which mapping uses too.

Poses carry the capture time of the frame they came from, so the state estimator
works with the true measurement time.

If ORB-SLAM3 loses the map and starts a new one, `SLAMWorker` signals a map reset.
While mapping, `MappingWorker` drops everything recorded and the lap has to be driven
again; in localization mode, poses are withheld until SLAM relocalises in the mapped map.

## Mapping lap

1. Start with `--enable-slam` and drive one lap of the track by hand.
2. Press `l` in the terminal when the lap is complete (or wait for ORB-SLAM3 to detect
   the loop closure).
3. `MappingWorker` takes the loop-corrected keyframe trajectory, trims it to one lap,
   keeps map points in a height band above the floor and within
   `mapping.max_track_half_width` of the driven line, splits them into left and right
   walls (lap direction from the trajectory, so clockwise laps work too) and publishes
   the smoothed boundaries. SLAM switches to localization-only mode, and you can arm
   with `a`.

## Tips

- The first frames after the camera starts have sparse depth, so `SLAMWorker` skips
  `slam.warmup_frames` (default 15) before initialising.
- Tracking needs texture. A blank wall or a close, featureless scene makes ORB-SLAM3
  lose tracking and re-initialise repeatedly.
- While tracking is lost no poses are published, so the state estimate goes stale and
  the watchdog brings the car to neutral (or the `depth_ftg` fallback takes over, if enabled).

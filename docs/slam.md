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

## RGB-D-inertial mode (`slam.inertial`)

ORB-SLAM3's `IMU_RGBD` sensor adds the OAK-D's own accelerometer and gyro to the RGB-D
frames. Off by default; not yet tried on the car.

```bash
python scripts/gen_orbslam_settings.py --imu    # camera level and still: also checks T_b_c1 against gravity
```

then set `slam.inertial: true` (and `camera.imu_rate_hz` if not 100; regenerate the
settings after changing it). `CameraWorker` streams accel + gyro (all three axes) into
`raw_imu_buf`, apart from `imu_buf`, which keeps whichever IMU `use_oakd_imu` selects, so
the external Arduino IMU can run alongside. `SLAMWorker` hands ORB-SLAM3 the samples
between consecutive frames (waiting up to `imu_wait_s` for the IMU to catch up).

- The noise values written by `--imu` are ORB-SLAM3's D435i example, not measured for
  this IMU; an Allan-variance recording of the still camera gives the real ones.
- While the IMU initialises, ORB-SLAM3 rotates its whole map to gravity (up to three
  times in the first ~15 s). Poses and map points are published relative to the map's
  origin keyframe, which rotates with it, so the published frame stays put; the frame
  of each rotation is withheld.
- The IMU initialisation needs varied motion; a car on a flat floor gives it little.
  Check it on real laps (the log says when the IMU is initialised) before relying on it.
- Tracking losses: upstream ORB-SLAM3 resets the whole active map on any tracking loss
  before the IMU is initialised and before its second inertial BA (~15 s), adds
  keyframes only every 0.25 s until the IMU is initialised, and never for weak visual
  tracking. At car speeds that loops: the local map falls behind within a few frames,
  tracking drops, the map is wiped, repeat. The fork (branch `inertial-depth-recovery`)
  changes this for `IMU_RGBD` / `IMU_STEREO`, where depth fixes the scale: a loss before
  initialisation is relocalised (then a new map) like the RGB-D mode, and weak tracking
  asks for a keyframe. Until the IMU is initialised, frames are tracked with the RGB-D
  mode's rules (upstream uses monocular-inertial ones, which lose a swerve at 4-5 m/s),
  and before the second inertial BA a lost map is given up after `IMU.EarlyLostTimeout`
  (0.5 s). `IMU.UpstreamPolicy: 1` in the settings restores upstream.
- The map is only started once the IMU sees acceleration ("not enough acceleration" in
  the log): standing still does not initialise anything; the IMU initialises ~1 s into
  driving and is fully refined (inertial BA 2) ~15 s later. A tracking loss before then
  starts a new map; after it, short featureless stretches are bridged by the IMU.
- Some OAK-Ds store no IMU calibration (the OAK-D Lite in `ORB_SLAM3/calib_*.json` has
  none): `--imu` then stops and asks for the CAM_A -> IMU transform via
  `--imu-extrinsics` (a 4x4 matrix as a list of rows, p_imu = T p_cam, metres).

## Rear IMU state estimate (`rear_imu`)

The Arduino's ICM-20948 at the back of the car can drive the state estimate:
`StateWorker` then runs `control.ImuEKF` (gyro yaw rate, through SLAM gaps, poses applied
at their capture time) instead of the pose-only EKF. The firmware already sends what it
needs (accel in g, gyro in deg/s, bias-corrected at start-up); `IMUWorker` writes it to
`rear_imu_buf`, dropping polls that repeat the last sample. In the sim this took position
error through 0.3-0.5 s SLAM gaps from 0.3-0.8 m to ~0.1 m and the yaw-rate error from
33-41 deg/s to 0.7 deg/s (`bench/2026-10-29_vio_sim/rear_imu_fusion.md`).

1. On the car, motors off, `start.py` stopped: `python scripts/check_rear_imu.py`. Keep
   still, then turn the car left ~90 deg by hand, then push it forward, when told. It
   prints the sample rate and repeats, the gyro bias at rest, and a `rear_imu` block with
   the mounting rotation (the firmware forces accel to (0, 0, 1 g) at rest whatever the
   mounting, so only motion shows it). The turn should read ~90 deg.
2. Measure `lever_m`: the IMU's position from the camera, in m, x forward, y left.
3. Record with it on (`make_record_config.py`, `rear_imu.enabled: true`) and run
   `python scripts/check_rear_imu.py --recording <dir>` for `delay_s` (its lag behind the
   OAK-D's IMU from the angular-rate magnitudes; the sim's DLPF + read gave 8.5 ms).
4. Paste the block into `config.yaml`. The published state stays the camera's point.

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

# pilot_core

Central control hub for UTASR's autonomous scale racing car. It runs the full
on-vehicle pipeline (camera, IMU, vision, SLAM, mapping, control and actuation)
as a set of Python processes connected by lock-free shared-memory ring buffers.

## Quick start (workstation, CPU-only, OAK-D plugged in)

```bash
git clone https://github.com/UTASR-UT-Autonomous-Scale-Racing/pilot_core.git
cd pilot_core
git submodule update --init pilot_controller pilot_vision pilot_ingest firmware

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python start.py --config config_cpu.yaml            # camera + IMU + vision + control
python start.py --config config_cpu.yaml --debug    # verbose per-worker logging
```

No camera? Run the same pipeline on synthetic data with
`python start.py --config config_dev_local.yaml`.

Stop with **Ctrl+C**. Workers shut down gracefully and shared memory is released.

## Pipeline

```
                 ┌──────────────┐  frame_buf   ┌──────────┐  seg_buf   ┌──────────┐  ctrl_buf  ┌──────────┐
 OAK-D ─────────▶│ CameraWorker │─────────────▶│  Vision  │───────────▶│ Control  │───────────▶│ Actuator │──▶ Arduino (I2C)
 (RGB+depth+IMU) └──────┬───────┘              └──────────┘            └────▲─────┘            └──────────┘
                        │ imu_buf                                           │ pose / boundaries
                        └──────────────────────────────▶ (Control)          │
                                   frame_buf ┌──────────┐  pose_buf    ┌────┴─────┐
                                   ────────▶ │   SLAM   │─────────────▶│ Mapping  │── boundary_buf
                                             │ ORB-SLAM3│  pc_buf      └──────────┘
                                             └──────────┘   (--enable-slam only)
```

Each worker (`workers/`) is a `multiprocessing` process with a
`setup()` / `step()` / `teardown()` lifecycle. `start.py` creates the buffers,
spawns the workers, restarts any that crash (up to 3 times) and handles
shutdown.

### Buffer contracts

| Buffer         | Payload                                                                        |
| -------------- | ------------------------------------------------------------------------------ |
| `frame_buf`    | BGR `uint8` (H,W,3) + depth `float32` metres (H,W), depth aligned to RGB        |
| `imu_buf`      | `[qw, qx, qy, qz, ax, ay, az, s7, s8]`, accel m/s² (see `IMUWorker` / `CameraWorker`) |
| `pose_buf`     | 4×4 **camera-to-world** pose (Twc); only published while SLAM tracking is OK   |
| `ctrl_buf`     | `steering, throttle` both in **[0, 1]**, **0.5 = neutral**: steer 0 = full left, 1 = full right; throttle < 0.5 = brake / reverse, > 0.5 = forward |

`ctrl_buf` follows the car's PWM convention (below neutral = left / brake-reverse,
above = right / forward). `pilot_controller.Controller` maps each channel linearly
over its `[MIN, MAX]` PWM range. Controllers and RL policies work in signed
`[-1, 1]` space with **+steering = left** (ROS / REP-103); `ControlWorker`
converts with `signed_to_unit()`, and the simulator bridge converts back.
`ActuatorWorker` clamps anything out of range as a last line of defence.

## Configuration profiles

| File                    | Use                                                                   |
| ----------------------- | --------------------------------------------------------------------- |
| `config.yaml`           | On the car (Jetson): CUDA, I2C actuator on `/dev/i2c-1`, SLAM + PID     |
| `config_cpu.yaml`       | Workstation: real OAK-D, CPU-only, **actuator output disabled**       |
| `config_dev_local.yaml` | No hardware: synthetic camera/IMU (also used by the dev container)    |
| `config_zmq.yaml`       | AutoDRIVE simulator over ZMQ (see [Simulator](#simulator-autodrive))  |

CLI flags override the config: `--enable-slam`, `--recording`, `--oakd-imu`, `--debug`.

> **Safety:** only `config.yaml` enables I2C. On a laptop or desktop,
> `/dev/i2c-*` are motherboard buses, not the car's Arduino. Never point the
> actuator at them.

### Hardware notes

- The OAK-D can only be opened by **one** process. With `use_oakd_imu: true`
  the on-board IMU is streamed by `CameraWorker`, so no separate IMU worker is started.
- Plug the OAK-D into a **USB 3** port with a USB 3 cable. On USB 2 the device
  still works but bandwidth limits the frame rate.
- `threads_per_worker` (default 1) caps BLAS/OpenMP/OpenCV threads per worker so
  that the separate processes don't oversubscribe small CPUs.

## SLAM (ORB-SLAM3)

Python bindings live in `orbslam3_wrapper/` (pybind11). Build once:

```bash
# System prerequisites (Ubuntu 22.04)
sudo apt-get install -y build-essential cmake git pkg-config \
    libeigen3-dev libepoxy-dev libglew-dev libgl1-mesa-dev libwayland-dev \
    libopencv-dev libboost-serialization-dev libssl-dev
# Pangolin (if not installed)
cd /tmp && git clone --depth 1 https://github.com/stevenlovegrove/Pangolin.git
cd Pangolin && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF
make -j3 && sudo make install && sudo ldconfig
```

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

This produces `ORB_SLAM3/lib/libORB_SLAM3.so` and `orbslam3_python*.so` in the
repo root (git-ignored; always build locally). Allow about 30 minutes on a 4-core CPU.

Generate the SLAM camera settings from the connected camera's factory
calibration. Calibration is per unit, so redo this whenever the camera is swapped:

```bash
python scripts/gen_orbslam_settings.py --config config_cpu.yaml   # writes orbslam_oakd_rgbd.yaml
```

Then run with SLAM (`scripts/run.sh` sets `LD_LIBRARY_PATH` for you):

```bash
scripts/run.sh --config config_cpu.yaml --enable-slam
```

## Simulator (AutoDRIVE)

pilot_core runs against the Unity-based AutoDRIVE simulator through an
`autodrive_bridge` over ZMQ. `ZMQSensorWorker` replaces the camera/IMU workers and
`ActuatorWorker` publishes commands back:

| Port | Direction | Content |
| ---- | --------- | ------- |
| 5555 | sim → pilot_core | camera (JPEG), depth, pose (ground truth), IMU, LiDAR, speed, lap/collision counters |
| 5556 | pilot_core → sim | `{"steering", "throttle", "reset"}` in `[-1, 1]`, steering +1 = left |
| 5557 | pilot_core → sim | telemetry / visualisation (boundaries, pose, segmentation) |

```bash
python start.py --config config_zmq.yaml
```

With `use_zmq_pose: true` the simulator's ground-truth pose replaces SLAM and a
LiDAR projection builds the wall map (`L` + ENTER signals loop closure). Set
`use_saved_map: true` to load boundaries from `saved_map_path` and drive
immediately. `control_mode` can be `pid`, `ftg`, `lidar_pid` or `neural`.

RL training (PPO, needs `pip install gymnasium stable-baselines3 torch`):

```bash
python rl/train.py              # new run → rl/logs/PPO_N, rl/checkpoints/
tensorboard --logdir rl/logs
```

## Tests

```bash
# Synthetic suite (no hardware)
PILOT_SYNTHETIC=1 python -m pytest tests/ --timeout=300

# Include the ORB-SLAM3 closed-loop tests (after building SLAM)
export LD_LIBRARY_PATH=$PWD/ORB_SLAM3/lib:/usr/local/lib:$LD_LIBRARY_PATH
PILOT_SYNTHETIC=1 python -m pytest tests/ --timeout=300

# Hardware tests against a connected OAK-D
OAKD_AVAILABLE=1 python -m pytest tests/test_pipeline_integration.py tests/test_workers.py -k "oakd or camera or slam"
```

## Containers

| Image                   | Use                                                                        |
| ----------------------- | -------------------------------------------------------------------------- |
| `Dockerfile`            | Jetson (L4T, NVIDIA runtime). Entrypoint is `start.py`; see `scripts/Deploy.md` |
| `Dockerfile.dev-local`  | Lightweight synthetic-only test container (`.devcontainer/dev-local/`)     |

## Repository layout

```
pilot_core/
├── start.py              # Orchestrator (entry point)
├── workers/              # Pipeline worker processes + track controllers (PID, FTG)
├── shared_buffer/        # Shared-memory ring buffers (seqlock, single-writer)
├── synthetic/            # Synthetic camera / IMU / closed-loop track generators
├── tests/                # pytest suite
├── orbslam3_wrapper/     # pybind11 bindings for ORB-SLAM3
├── rl/                   # Gymnasium env + PPO training on AutoDRIVE
├── collect_frames.py     # Save simulator camera frames for dataset building
├── scripts/              # Launch, deploy and setup helpers
├── src/ include/ utility/  # Legacy C++ pipeline (scripts/READMECPP.md)
├── config*.yaml          # Runtime profiles
├── vehicle_params.yaml   # Vehicle geometry/limits (F1TENTH placeholders)
└── submodules: pilot_controller, pilot_vision, pilot_ingest, firmware, ORB_SLAM3, depthai-core
```

## Scripts

| Script                        | Purpose                                                  |
| ----------------------------- | -------------------------------------------------------- |
| `run.sh [args]`               | Activate `.venv`, set `LD_LIBRARY_PATH`, run `start.py`   |
| `dev_run.sh [args]`           | `run.sh` with `config_cpu.yaml --debug`                  |
| `jetson_run.sh`               | On the car: performance mode, hard-reset to origin, launch |
| `start.sh`                    | Jetson Docker deploy (pull, validate, build, run)        |
| `gen_orbslam_settings.py`     | ORB-SLAM3 settings from OAK-D calibration                |
| `setup.sh`                    | Init all submodules                                      |
| `branch_checkout.sh [branch]` | Check out a branch in pilot_core and all submodules      |
| `build.sh` / `runCpp.sh` / `replay.sh` | Legacy C++ pipeline                             |

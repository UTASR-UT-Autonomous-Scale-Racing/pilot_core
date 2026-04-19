# pilot_core

Central control hub for UTASR's autonomous scale racing car. It runs the full
on-vehicle pipeline (camera, IMU, vision, SLAM, mapping, control and actuation)
as a set of Python processes connected by lock-free shared-memory ring buffers.

## Quick start (workstation, CPU-only, OAK-D plugged in)

```bash
git clone https://github.com/UTASR-UT-Autonomous-Scale-Racing/pilot_core.git
cd pilot_core
git submodule update --init pilot_vision firmware

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python start.py --config config_cpu.yaml            # camera + IMU + map-free depth_ftg control
python start.py --config config_cpu.yaml --debug    # verbose per-worker logging
```

No camera? Run the same pipeline on synthetic data with
`python start.py --config config_dev_local.yaml`.

Operator keys in the terminal:

| Key       | Action                                               |
| --------- | ---------------------------------------------------- |
| `a`       | **Arm**: actuator outputs follow the controller       |
| `SPACE`   | **Disarm** (any key other than `a`/`l`/`q`): neutral immediately |
| `l`       | Signal loop closure (end of the manual mapping lap)   |
| `q`       | Quit (also **Ctrl+C**). Workers shut down gracefully   |

Outputs are **neutral until armed**.

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
| `frame_buf`    | **RGB** `uint8` (H,W,3) + depth `float32` metres (H,W), depth aligned to RGB    |
| `imu_buf`      | `[qw, qx, qy, qz, ax, ay, az, s7, s8]`, accel m/s² (see `IMUWorker` / `CameraWorker`) |
| `pose_buf`     | 4×4 **camera-to-world** pose (Twc); only published while SLAM tracking is OK   |
| `state_buf`    | `t, x, y, yaw, v, yaw_rate` in the BEV plane (StateWorker EKF)                 |
| `ctrl_buf`     | `steering, throttle` both in **[0, 1]**, **0.5 = neutral**: steer 0 = full left, 1 = full right; throttle < 0.5 = brake / reverse, > 0.5 = forward |

`ctrl_buf` follows the car's PWM convention (below neutral = left / brake-reverse,
above = right / forward). `control.pwm.PWMMapper` maps each channel linearly over
its `[min, max]` pulse width, set in the config `pwm` section. Controllers and RL policies work in signed
`[-1, 1]` space with **+steering = left** (ROS / REP-103); `ControlWorker`
converts with `signed_to_unit()`, and the simulator bridge converts back.
`ActuatorWorker` clamps anything out of range as a last line of defence.

## Driving and control

After the hand-driven mapping lap (`--enable-slam`, press `l` to close the loop),
the track boundaries become a `TrackModel` (centre-line, curvature, width), and a
track controller drives it:

| `control_mode` | Controller | Bench cross-track RMS (twisty track) | Notes |
| -------------- | ---------- | ------------------------------------ | ----- |
| `pure_pursuit` | Look-ahead point on the centre-line (default) | 3.4 cm | Smoothest steering, most robust |
| `stanley`      | Front-axle heading + cross-track error | 1.9 cm | |
| `lqr`          | Speed-scheduled LQR on the path-error model | 0.8 cm | |
| `mpc`          | Linear time-varying kinematic MPC (OSQP, ~2 ms/solve) | 0.8 cm | Steering-angle/rate constraints, delay compensation |
| `policy`       | PPO policy exported to ONNX (`rl/export_onnx.py`) | n/a | Needs a trained policy |
| `depth_ftg`    | Follow-the-Gap on a depth-camera virtual scan | 7 cm | **Map-free**; also usable as the tracking-loss fallback |
| `vae`          | VAE latent of the depth frame → learned steering head (World Models) | 10.9 cm | **Map-free**, learned; train with `scripts/train_vae_policy.py` |
| `bc`           | Behaviour cloning: end-to-end CNN, depth frame → steering (PilotNet) | 4.3 cm | **Map-free**, learned; train with `scripts/train_bc_policy.py` |
| `pid`, `ftg`, `lidar_pid` | Legacy / simulator modes | | |

References for each algorithm: [`docs/references.md`](docs/references.md).

`track_control.fallback: depth_ftg` makes the track modes drive map-free while the
state estimate is stale (e.g. SLAM tracking lost), and hand back once it recovers.

Speed comes from a curvature-based profile (`track_control.speed`) and a PI
controller for a drivetrain **without a brake**: neutral coasts, and a limited
backward input slows the car (never below 0.3 m/s, so it cannot start reversing).
`StateWorker` fuses SLAM poses into a speed and yaw-rate estimate (EKF), since the car has
no wheel encoders.

Every command then passes the **safety supervisor** in `ActuatorWorker` (config `safety`):

1. **Arming:** neutral until `a` is pressed.
2. **Watchdog:** no fresh command for 150 ms gives neutral throttle with steering held,
   so the car coasts to a stop (also when SLAM tracking is lost).
3. **Speed cap:** 1.5 m/s. Throttle tapers near the cap and is bounded by
   `speed_cap / speed_per_throttle`.
4. **Slew limits** on steering and throttle.

> **Calibrate `speed_per_throttle`** (`track_control.speed.longitudinal`) on the car:
> the steady-state speed per unit of forward throttle. Both the speed controller
> and the speed-cap bound depend on it. `vehicle_params.yaml` still holds F1TENTH
> placeholder values.

Evaluate controllers without hardware on the closed-loop bench (kinematic bicycle
model, 15 Hz noisy delayed poses, the same estimator, controller and safety code):

```bash
python scripts/sim_bench.py                       # all controllers × tracks
python scripts/sim_bench.py --mode stanley --plot bench.png
python scripts/sim_bench.py --mode depth_ftg      # renders depth images, no pose/map
```

Training the learned depth controllers (bench demonstrations → model → ONNX). Both scripts
can share one cached dataset:

```bash
python scripts/train_bc_policy.py  --data demos.npz --out bc_policy.onnx    # control_mode: bc
python scripts/train_vae_policy.py --data demos.npz --out vae_policy.onnx   # control_mode: vae
```

Deploying a trained RL policy:

```bash
python rl/export_onnx.py --run PPO_3 --out policy.onnx   # bakes in VecNormalize + clipping
# then set control_mode: policy and track_control.policy.model: policy.onnx
```

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
immediately. Any track mode works with the simulator's pose; `lidar_pid` uses its LiDAR.

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
├── start.py              # Orchestrator (entry point, operator keys)
├── control/              # Control library: track model, controllers, EKF, safety, sim bench
├── workers/              # Pipeline worker processes + track controllers (PID, FTG)
├── shared_buffer/        # Shared-memory ring buffers (seqlock, single-writer)
├── synthetic/            # Synthetic camera / IMU / closed-loop track generators
├── tests/                # pytest suite
├── orbslam3_wrapper/     # pybind11 bindings for ORB-SLAM3
├── rl/                   # Gymnasium env + PPO training on AutoDRIVE
├── collect_frames.py     # Save simulator camera frames for dataset building
├── scripts/              # Launch, deploy and setup helpers
├── config*.yaml          # Runtime profiles
├── vehicle_params.yaml   # Vehicle geometry/limits (F1TENTH placeholders)
└── submodules: pilot_vision (segmentation models), firmware (Arduino), ORB_SLAM3
```

## Scripts

| Script                        | Purpose                                                  |
| ----------------------------- | -------------------------------------------------------- |
| `run.sh [args]`               | Activate `.venv`, set `LD_LIBRARY_PATH`, run `start.py`   |
| `dev_run.sh [args]`           | `run.sh` with `config_cpu.yaml --debug`                  |
| `jetson_run.sh`               | On the car: performance mode, hard-reset to origin, launch |
| `start.sh`                    | Jetson Docker deploy (pull, validate, build, run)        |
| `gen_orbslam_settings.py`     | ORB-SLAM3 settings from OAK-D calibration                |
| `sim_bench.py`                | Closed-loop controller benchmarks (no hardware)          |
| `setup.sh`                    | Init all submodules                                      |
| `branch_checkout.sh [branch]` | Check out a branch in pilot_core and all submodules      |

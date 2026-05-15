# Changelog

## Unreleased: release clean-up (branch `release/cleanup`)

Based on `migration_multiprocessing_in_python`, with the AutoDRIVE simulator branch
merged in. Nothing here has been driven on the car yet; see "Not yet verified"
below.

### Control

- A control library (`control/`): track model, pose EKF, speed profile with a PI
  speed controller (no brake: neutral coasts, limited reverse to slow down).
- Track followers: Pure Pursuit (default), Stanley, speed-scheduled LQR,
  linear time-varying MPC (OSQP), and an ONNX PPO policy trained in AutoDRIVE.
- Map-free controllers from the depth image: Follow-the-Gap (`depth_ftg`),
  behaviour cloning (`bc`) and a VAE latent policy (`vae`).
- A closed-loop bicycle-model bench (`scripts/sim_bench.py`) running the same
  estimator → controller → safety chain as the car.

### Safety

- Safety supervisor in the actuator: terminal arming, a 150 ms command watchdog
  (stale, future-stamped and non-finite commands), speed cap with a taper, a reverse
  cap and slew limits.
- The actuator is silent on I2C while disarmed, so the RC transmitter drives the
  mapping lap. Firmware (submodule branch `jetson-command-timeout`): Jetson
  commands time out after 250 ms and any RC stick movement takes over.
- Workers stop on SIGTERM/SIGHUP and when the orchestrator dies. A crash on the
  command path disarms the car, and SLAM is not restarted after mapping.
- The map-free controllers go neutral on invalid depth or a close obstacle.

### Perception and mapping

- Frames and poses are stamped with the camera's capture time, and all buffers use
  one monotonic clock.
- SLAM publishes in a floor-levelled world frame, detects map resets, and signals a
  loop closure only after global bundle adjustment. The wrapper loads without
  `LD_LIBRARY_PATH`.
- Mapping uses the loop-corrected keyframe trajectory, the floor plane fitted to the
  map, and places each wall at the nearest dense layer of points. It handles
  clockwise laps. On the dev simulator the boundaries are within 1.2 cm (median) of
  the true walls.
- The camera re-opens after disconnects, and the I2C IMU reads the firmware's
  actual 60-byte frame.
- Vision: drivable-track segmentation candidates (Fast-SCNN, LR-ASPP, DeepLabV3,
  U-Net, training-free `ground_plane`) with training and ONNX export. Vision is off
  by default since nothing consumes the mask yet.

### Recording and data

- A single recorder process writes every buffer, rotates files instead of
  overwriting them, and misses no samples. Buffer publication is hardened for the
  Jetson's weak memory ordering.
- `scripts/read_recording.py`, `scripts/recording_to_dataset.py` (BC and vision
  datasets), and `scripts/render_recording.py` (the README figures and demo GIF).

### Simulation

- AutoDRIVE bridge and RL training with the same safety settings as the car.
- A dev simulator: the synthetic camera ray-casts a closed track, so the whole
  pipeline runs without hardware (`config_dev_local.yaml`).

### Project

- `pyproject.toml`, a CI workflow, `.dockerignore`, Dockerfiles that build ORB-SLAM3
  (JetPack 6 base), and scripts that use pinned submodules.
- `start.py --config` is required; `start.py` builds the pipeline with
  `build_pipeline()` and adds a `--control-mode` override.
- Everything runs from any working directory (`control.types.resolve_path`).
- Documentation: README with dev-simulator evidence, `docs/architecture.md`,
  control, vision, SLAM, simulator, deployment and references.

### Removed

- Legacy `pid`, `ftg` and `lidar_pid` modes, the web publisher, the `pilot_vision`,
  `pilot_controller`, `pilot_ingest` and `depthai-core` submodules, and the old C++
  pipeline.

### Not yet verified

- The firmware change has not been compiled or bench-tested
  (`firmware/README.md` has the checklist). The parent repo still pins the old
  firmware commit.
- Nothing has been driven on the car: calibrate `speed_per_throttle`, measure the
  vehicle parameters (F1TENTH placeholders now), and do the camera checks listed in
  `docs/slam.md` and `docs/control.md`.
- No license has been chosen (ORB-SLAM3 is GPLv3).

# Control

This guide covers how the car is controlled: the data flow, conventions, the
available controllers, the safety layer, the closed-loop bench, training the
learned models, and what to check before driving the real car.

## Data flow

```
pose_buf ──▶ StateWorker (PoseEKF) ──state_buf──┐
boundary_buf ──▶ TrackModel ────────────────────┼──▶ ControlWorker ──ctrl_buf──▶ ActuatorWorker
frame_buf (depth) ──────────────────────────────┘      controller            SafetySupervisor → PWMMapper → I2C
```

- **State:** `StateWorker` runs `control.PoseEKF` on the SLAM poses and publishes
  position, heading, speed and yaw rate at 50 Hz. The car has no wheel encoders,
  so speed comes from the pose history. If poses stop for 0.5 s the estimate is
  marked stale and nothing is published.
- **Track:** once mapping is done, `control.TrackModel` builds a smoothed centre-line
  from the mapped boundaries (arc length, curvature, width) and orients it with the
  car's driving direction.
- **Controller:** `ControlWorker` runs the configured `control_mode` and writes the command
  to `ctrl_buf`.
- **Actuation:** every 20 ms `ActuatorWorker` passes the newest command through
  `control.SafetySupervisor`, maps it to pulse widths with `control.PWMMapper` and
  writes them to the Arduino.

All algorithms live in `control/` as plain Python classes, which you can import and
test without any hardware or worker processes.

## Conventions

| Quantity | Convention |
| -------- | ---------- |
| Ground plane | 2-D `(x, y)` = two world axes chosen by `mapping.bev_axes` (SLAM camera frame: `[0, 2]`) |
| Heading | Radians, counter-clockwise from +x. The car's forward axis is `mapping.forward_axis` (camera: `2`) |
| Signed command | Steering **+1 = full left**, throttle +1 = full forward, −1 = full backward, 0 = neutral |
| `ctrl_buf` | Both channels in [0, 1], 0.5 = neutral; steering 0 = full left, 1 = full right |
| PWM | Linear over each channel's `[min, max]` pulse width (config `pwm`); below neutral = left / backward |

`workers.control_worker.signed_to_unit()` / `unit_to_signed()` convert between
signed commands and `ctrl_buf`.

The drivetrain has no separate brake. Neutral (throttle 0) lets the car coast and
slow down; a backward input pushes it backwards and would reverse once stopped.

## Controllers

### Track followers (need SLAM and a mapped track)

| `control_mode` | Class | Notes |
| -------------- | ----- | ----- |
| `pure_pursuit` | `control.PurePursuit` | Steers at a point `lookahead` metres ahead on the centre-line. Smoothest steering. Default |
| `stanley` | `control.Stanley` | Heading error plus cross-track error at the front axle |
| `lqr` | `control.lqr.LQR` | Discrete LQR on the path-error model, gains scheduled on speed, curvature feed-forward |
| `mpc` | `control.mpc.MPC` | 1 s-horizon linear time-varying MPC with steering angle/rate limits and delay compensation (OSQP, about 2 ms per step) |
| `policy` | `control.policy.PolicyFollower` | PPO policy exported to ONNX; builds the same observation as training |

The lateral controllers are combined with the speed planner in `control.TrackFollower`:

- **Speed profile:** `v ≤ sqrt(a_lat / |κ|)`, capped at `track_control.speed.v_max`,
  with forward and backward passes so accelerating and slowing down stay within limits.
- **Speed control:** a PI controller with feed-forward `v / speed_per_throttle`. It
  uses a limited backward input to slow down, and never below 0.3 m/s so it cannot start reversing.

Set `track_control.fallback: depth_ftg` to drive map-free while the state estimate
is stale (for example when SLAM loses tracking). Control returns to the track follower
when the estimate recovers.

### Map-free (depth camera only)

| `control_mode` | Class | Notes |
| -------------- | ----- | ----- |
| `depth_ftg` | `control.depth_ftg.DepthGapFollower` | Builds a virtual scan from the depth image (floor removed by a height band) and steers at the centre of the best free gap |
| `bc` | `control.imitation.DepthPolicyFollower` + `control.bc` | End-to-end CNN, depth frame → steering |
| `vae` | `control.imitation.DepthPolicyFollower` + `control.vae` | VAE latent of the depth frame → steering head |

The map-free modes drive at a cruise speed that drops with steering. `depth_ftg`
assumes a level camera at `depth_ftg.camera_height` above the ground.

### Legacy and simulator modes

`pid` and `ftg` steer on raw poses and boundaries (`workers/track_pid_controller.py`,
`workers/track_ftg_controller.py`). The PID only aligns heading with the local boundary
direction and has no cross-track term, so it drifts; prefer the track followers.
`lidar_pid` centres between walls using the simulator's LiDAR.

## Safety supervisor

`control.SafetySupervisor` runs inside `ActuatorWorker` on every output tick (config `safety`):

1. **Arming:** outputs are neutral until the operator presses `a`; `SPACE` disarms immediately.
2. **Watchdog:** if the newest command is older than `command_timeout` (150 ms), throttle goes
   neutral and steering is held, so the car coasts to a stop.
3. **Speed cap:** forward throttle tapers to zero over the last `cap_margin` below
   `speed_cap` (1.5 m/s), and never exceeds `speed_cap / speed_per_throttle`, the throttle
   whose steady-state speed is the cap. The second limit holds even when the speed estimate lags.
   Without a speed estimate, forward throttle is capped at `max_throttle_no_speed`.
4. **Slew limits** on steering and throttle. Dropping to neutral is never delayed.

## Configuration

All keys are commented in `config.yaml`. The control-related sections are:

| Section | Contents |
| ------- | -------- |
| `track_control` | Per-controller gains (`pure_pursuit`, `stanley`, `lqr`, `mpc`), `policy.model`, `speed` (profile + `longitudinal`), `fallback` |
| `depth_ftg`, `bc`, `vae` | Map-free controllers (camera height, cruise speed, model paths) |
| `estimator` | EKF noise settings and staleness timeout |
| `safety` | Watchdog, speed cap, slew rates |
| `pwm` | Servo / ESC pulse widths |
| `vehicle_params.yaml` | Wheelbase, steering limits, friction, etc. |

`track_control.speed.longitudinal.speed_per_throttle` is the one drivetrain
calibration. The speed controller, the speed cap and the map-free modes all use it.

## Closed-loop bench

`control/sim.py` drives the real estimator → controller → safety chain against a
kinematic bicycle model with steering delay and rate limit, a no-brake drivetrain,
and simulated SLAM poses (15 Hz, noisy, 60 ms latency). For the map-free modes it
renders depth images of the track walls instead.

```bash
python scripts/sim_bench.py                          # every controller on both tracks
python scripts/sim_bench.py --mode mpc --track twisty --plot mpc.png
python scripts/sim_bench.py --mode depth_ftg
```

The same scenarios run as acceptance tests in `tests/test_control_bench.py` and
`tests/test_control_models.py`. Typical results on the twisty track:

| Controller | Cross-track RMS / max | Off track | Speed |
| ---------- | --------------------- | --------- | ----- |
| `pure_pursuit` | 3.4 / 11.6 cm | 0 % | 1.3 m/s |
| `stanley` | 1.9 / 4.5 cm | 0 % | 1.3 m/s |
| `lqr` | 0.8 / 3.2 cm | 0 % | 1.3 m/s |
| `mpc` | 0.8 / 2.6 cm | 0 % | 1.3 m/s |
| `depth_ftg` | 6.8 / 12.7 cm | 0 % | 0.5 m/s |
| `bc` | 4.3 / 16.0 cm | 0 % | 0.7 m/s |
| `vae` | 10.9 / 26.5 cm | 0 % | 0.7 m/s |

MPC matches LQR's accuracy with about half the steering activity.

## Learned models

### Behaviour cloning (`bc`) and VAE (`vae`)

Both learn depth frame → steering from demonstrations
(`control/imitation.py`). An LQR expert drives random bench tracks; noise is injected into
the executed steering so the data contains recoveries, while the labels are the expert's
clean commands; frames are mirrored to balance left and right turns.

```bash
python scripts/train_bc_policy.py  --data demos.npz --out bc_policy.onnx    # about 8 min on 4 cores
python scripts/train_vae_policy.py --data demos.npz --out vae_policy.onnx   # about 12 min
```

Each script collects the data (or loads the `--data` cache), trains, exports an `.onnx`
file plus a `.json` sidecar, and prints closed-loop results on held-out tracks. Point
`bc.model` / `vae.model` in the config at the exported file.

These models learn **rendered** depth. To drive the real car, retrain on recorded
real depth frames with expert labels.

### RL policy (`policy`)

`rl/train.py` trains PPO in the AutoDRIVE simulator ([simulator.md](simulator.md)).
Export a run for the car:

```bash
python rl/export_onnx.py --run PPO_3 --out policy.onnx
```

The exported graph includes observation normalisation and action clipping, and the
sidecar records the observation settings the policy was trained with. Set
`control_mode: policy` and `track_control.policy.model`.

## Adding a controller

1. **Track follower:** implement `steer(state, track, projection) -> wheel angle (rad, + = left)`
   and `reset()` (see `control/lqr.py`), then add a factory to `LATERAL_CONTROLLERS` in
   `control/controllers.py` and the mode name to `TRACK_MODES` in `workers/control_worker.py`.
   The speed planner, fallback and safety come for free.
2. **Map-free controller:** implement `step(depth) -> Command` and register a setup/step pair
   in `ControlWorker._modes()`.
3. Add its config section, a bench acceptance test (`tests/test_control_bench.py` style)
   and a reference in `docs/references.md`.

## Before driving the car

1. **Calibrate `speed_per_throttle`:** hold a few fixed throttles on a straight and read
   the speed from `state_buf` (SLAM running).
2. **Measure the vehicle:** wheelbase and maximum steering angle in `vehicle_params.yaml`,
   the camera's offset from the rear axle (the controllers treat the pose as the rear axle),
   and the camera height and pitch for `depth_ftg`.
3. **Bench test on stands:** wheels off the ground, check arm/disarm response and steering
   direction (steering above 0.5 in `ctrl_buf` must turn right).
4. **First laps:** a low `track_control.speed.v_max` (e.g. 0.6 m/s) on the mapped track.

## Known limitations

- The EKF uses poses only. Fusing the OAK-D IMU would reduce the speed-estimate lag.
- `depth_ftg` can alternate between two similar gaps in cluttered scenes.
- `rl/train.py` defines `TrackCNNExtractor` but trains the default MLP policy.

# Control models: assessment and implementation plan

Status: **all controllers implemented** (P0–P4, C1–C6, plus LQR, a VAE and a BC model), tested in the
closed-loop bench and on the local camera/SLAM pipeline. Not yet run on the car.
Remaining: on-car calibration (§6) and training an RL policy for C5.
References: [`references.md`](references.md).

## 1. Where control stands today

| Controller | Location | State |
| --- | --- | --- |
| `pilot_controller` models (`MeanIntensity`, `RNN`, RNN trainer) | removed submodule | Placeholders (constant output / stubs). The submodule was removed; its PWM mapping now lives in `control/pwm.py` and `neural` mode is gone. |
| `TrackPIDController` | `workers/track_pid_controller.py` | Runs. Since the sim merge it aligns the heading with the local boundary tangent, but has **no cross-track term**, so lateral drift is never corrected. In a closed-loop test at 0.84 m/s it drifted off the track within about one lap (on track 43% of the time). Also: no `dt`, no anti-windup, constant throttle, Gaussian noise injected into real commands. |
| `TrackFTGController` | `workers/track_ftg_controller.py` | Implemented, unit-tested, selectable with `control_mode: ftg` (since the merge). Not yet evaluated closed-loop. |
| RL (PPO) + `LidarWallController` | `rl/`, `workers/lidar_wall_controller.py` (merged) | Gymnasium env over AutoDRIVE, PPO with a 1-D CNN over boundary points; LIDAR wall-centering PID (`control_mode: lidar_pid`). |

Fixed during the release clean-up (so they're not repeated below): the PID [-1,1] vs
actuator [0,1] range crash, the SLAM Tcw-vs-Twc pose convention, the PID heading
convention, and lost-tracking frames published as the identity pose.

**What's missing across the board:** a vehicle state estimate (speed in
particular), a track representation that controllers can query cheaply, a
safety layer between control and actuation, and any closed-loop way to
evaluate a controller before it touches the car.

## 2. Prerequisites (shared by every controller)

These come first. Every controller below depends on them.

### P0a. Signed actuation contract (brake + reverse) — **done**
- `ctrl_buf`: both channels `[0, 1]`, **0.5 = neutral**, car PWM convention
  (steer 0 = full left, 1 = full right; throttle < 0.5 = brake / reverse).
- `control.pwm.PWMMapper` (formerly `pilot_controller.Controller`) maps throttle linearly over
  `[THROTTLE_MIN, THROTTLE_MAX]`, like steering; 0.5 lands exactly on
  `THROTTLE_NEURAL`. Steering offsets unchanged (intended).
- Signed controller convention: **+steering = left** (ROS / REP-103), matching
  the controllers' geometry and AutoDRIVE. `signed_to_unit()` negates steering
  into `ctrl_buf`, and the ZMQ actuator converts back. Before this, PID/FTG would
  have steered the *car* the wrong way (they only worked in the simulator).
- Drivetrain (confirmed): input is forward or backward; neutral (untouched) coasts
  and slows the car, and there is no separate brake.

### P0b. Actuation safety layer: `ActuatorWorker` — **done** (`control/safety.py`)
- **Command watchdog:** if the newest `ctrl_buf` entry is older than `command_timeout_ms` (e.g. 150 ms), send neutral throttle and hold steering. Today a stalled controller or lost SLAM tracking leaves the car on its last command.
- **Rate / slew limits** on steering and throttle (configurable), applied after the controller.
- **Arming:** actuator starts disarmed (neutral) and only arms on explicit operator action: terminal key `a`; `SPACE` disarms immediately.
- Tests: watchdog timing, slew limiting, disarmed-by-default.

### P1. Vehicle state estimator: new `StateWorker` → `state_buf` — **done** (pose-only EKF, `control/estimator.py`)
- Implemented as a pose-only EKF (unicycle model) published at 50 Hz. IMU fusion is a follow-up (§6): it would cut the ~150 ms speed-estimate lag, but needs the OAK-D IMU→camera extrinsics.
- Gives controllers **speed** (currently unknown) and smooth, high-rate pose between SLAM frames; bridges short tracking losses with a bounded covariance.
- (The removed `pilot_controller/lap_engine.EKFLocalization` double-integrated world-frame acceleration and drifted; not reused.)

### P2. Track model: `TrackModel` (library, not a worker) — **done** (`control/track.py`)
- Built once from `boundary_buf`: order and resample boundaries, compute the centre-line, arc length `s`, curvature `κ(s)` and left/right widths.
- `project(x, z) → (s, lateral_error, heading_error)` with index continuity (no global nearest-point search every step).
- Later: minimum-curvature race line as an alternative reference path.
- Reuse `workers/boundary_resampler.py` (merged from the sim branch).

### P3. Controller interface + selection — **done** (`ControlWorker` mode registry, `control/controllers.py`)
```python
class TrackController(Protocol):
    def reset(self) -> None: ...
    def step(self, state: VehicleState, track: TrackModel) -> Command: ...  # steer, throttle ∈ [-1,1] (P0a)
```
- `ControlWorker` gets a registry keyed by `control_mode` (`pid`, `ftg`, `lidar_pid`, `pure_pursuit`, `stanley`, `lqr`, `mpc`, `policy`, `depth_ftg`, `vae`) instead of the current if/else chain.

### P4. Closed-loop evaluation harness — **done** (`control/sim.py`, `tests/test_control_bench.py`, `scripts/sim_bench.py`)
- Pure-Python kinematic bicycle simulator on synthetic tracks (`conftest.track_boundaries`, `synthetic/closed_loop_generator.py`) → `pytest` benchmarks: lap completion, lap time, cross-track RMS / max, steering jerk.
- Every controller must pass it before car testing. Seconds per run on CPU.
- AutoDRIVE (via the ZMQ workers) is the higher-fidelity stage. A fake ZMQ bridge with a kinematic bicycle model (F1TENTH params) already closes the loop around the unmodified pipeline. It was used to validate the merge and should become this harness.

## 3. Controllers to implement (recommended order)

### C1. Pure Pursuit, the new baseline (replaces PID as the default) — **done**, default in `config.yaml`
- Geometric path tracking on the `TrackModel` centre-line: look-ahead point at `L_d = clamp(k·v + L_min, L_min, L_max)`, steer `δ = atan(2·L·sin α / L_d)`, normalised by max steering angle.
- Fixes the PID's core flaw: the reference is ahead of the car, so there is no singularity on the line.
- Needs: P1 (speed), P2, wheelbase `L` and max steering angle from `vehicle_params.yaml`.
- Effort: **S** (≈1–2 days incl. tests).

### C2. Speed planner / longitudinal control — **done** (no-brake drivetrain: neutral coasts, limited backward input)
- Curvature-based velocity profile: `v_max(s) = sqrt(a_lat_max / |κ(s)|)` (with `a_lat_max ≈ μ·g` from `friction_coeff`), forward/backward pass with `max_accel` / `max_decel` limits. Throttle via a PI controller on `v_ref − v`, with feed-forward from a measured throttle→speed map. With P0a, negative output brakes, so corners can be entered at the planned speed.
- Pairs with C1/C3. Today throttle is a constant 0.3.
- Needs: P1 and a one-off throttle-map calibration run (log PWM vs. estimated speed).
- Effort: **S–M**.

### C3. Stanley controller (alternative lateral controller) — **done**
- `δ = ψ_e + atan(k·e / (k_soft + v))` on the front axle. Often better than Pure Pursuit at low speed and tight tracking.
- Same interface and inputs as C1, so it's cheap once C1 exists, and A/B-able in the harness.
- Effort: **S**.

### C4. Reactive depth controller (no map needed) — **done** (`depth_ftg`, also the optional tracking-loss fallback)
- FTG (already written) fed by a *virtual scan built from the depth image* (ground-plane removal → per-column nearest obstacle) instead of from mapped boundaries. The sim branch's `depth_projection_worker.py` does this.
- The mapping lap stays manual (decided), so this is mainly a **fallback when SLAM tracking is lost** and a map-free mode for new tracks. Lower priority.
- Effort: **M**.

### C5. Learned policy deployment (`policy` mode) — **done** (deployment path; no trained policy yet)
- Load a trained policy exported to **ONNX** and run it on CPU with onnxruntime (tiny MLP/1-D CNN: well under 1 ms). The observation builder reuses `TrackModel` to produce the sim branch's boundary-point observation.
- Start from the merged PPO work in `rl/`. Sim-to-real gap is the main risk; gate behind the P0b safety layer and the P4 harness.
- Optional follow-on: behaviour cloning from `BufferRecorderWorker` recordings of C1/C2 laps (an action-predicting model; see also the VAE model).
- Effort: **M** for deployment; training effort is open-ended.

### C6. MPC — **done** (`mpc`, linear time-varying kinematic MPC; plus `lqr`)
- Kinematic-bicycle MPC (CasADi/acados) tracking the race line with input constraints. Best lap-time potential, but needs an identified vehicle model and more CPU. Revisit after C1–C3 are characterised on the car.
- Effort: **L**.

## 4. Decisions
1. **Steering / throttle sign (22 Sep):** set by the neutral values in the control
   config: below neutral = left / backward, above = right / forward. → P0a.
2. **Vehicle parameters (22 Sep):** F1TENTH simulator values for now
   (`vehicle_params.yaml`, placeholders until the car is measured).
3. **Mapping lap (22 Sep):** driven by hand. C4 drops in priority.
4. **Throttle (22 Sep):** forward / backward input, neutral coasts, no separate brake. → P0a, C2.
5. **Simulator:** AutoDRIVE. `simulator_integration` merged; the PyBullet
   `scale_racing_simulator` is outdated and ignored.
6. **Steering offsets:** intended. Linear `[MIN, MAX]` mapping kept.
7. **Arming:** terminal key press. **Safety defaults:** 150 ms watchdog, slew limits, 1.5 m/s cap.
8. **Frames:** RGB.

### Simulator follow-ups found during the merge
- `config_zmq.yaml` references `pilot_vision/models/deeplab/config.yaml`, which only
  exists on `pilot_vision`'s `deeplab` branch (the submodule is pinned to `main`).
  Not needed in the default sim mode (LiDAR mapping skips vision).
- `simulator_orbslam_settings.yaml` is referenced but was never committed (only
  needed for SLAM-in-sim).
- ~~RGB vs BGR~~: resolved. `frame_buf` is **RGB** everywhere (the vision models expect RGB);
  CameraWorker converts from DepthAI's BGR and ORB-SLAM3 uses `Camera.RGB: 1`.

## 5. Results (closed-loop bench, 60 s, 15 Hz noisy poses with 60 ms latency)

| Track | Controller | Cross-track RMS / max | Off track | Speed mean / max |
| ----- | ---------- | --------------------- | --------- | ---------------- |
| oval | Pure Pursuit | 1.4 / 3.7 cm | 0 % | 1.29 / 1.43 m/s |
| oval | Stanley | 1.6 / 3.8 cm | 0 % | 1.29 / 1.43 m/s |
| twisty | Pure Pursuit | 3.4 / 11.6 cm | 0 % | 1.29 / 1.43 m/s |
| twisty | Stanley | 1.9 / 4.5 cm | 0 % | 1.29 / 1.43 m/s |

| oval | LQR | 0.7 / 1.7 cm | 0 % | 1.29 / 1.43 m/s |
| oval | MPC | 0.7 / 1.9 cm | 0 % | 1.29 / 1.43 m/s |
| twisty | LQR | 0.8 / 3.2 cm | 0 % | 1.29 / 1.43 m/s |
| twisty | MPC | 0.8 / 2.6 cm | 0 % | 1.29 / 1.43 m/s |
| oval | depth FTG (map-free) | 5.1 / 15.2 cm | 0 % | 0.52 / 0.53 m/s |
| twisty | depth FTG (map-free) | 6.8 / 12.7 cm | 0 % | 0.51 / 0.52 m/s |
| oval | VAE (map-free, learned) | 7.3 / 19.1 cm | 0 % | 0.73 / 0.78 m/s |
| twisty | VAE (map-free, learned) | 10.9 / 26.5 cm | 0 % | 0.71 / 0.79 m/s |
| oval | BC (map-free, learned) | 4.3 / 6.7 cm | 0 % | 0.74 / 0.78 m/s |
| twisty | BC (map-free, learned) | 4.3 / 16.0 cm | 0 % | 0.72 / 0.79 m/s |

BC model (`bc`): PilotNet-style CNN trained end to end by `scripts/train_bc_policy.py` (8 min) on the **same** dataset as the VAE model. Its steering error on held-out frames is 0.018 (VAE 0.038), and it tracks tighter with smoother steering. The VAE's advantage is that its encoder can also learn from unlabelled frames.

VAE model (`vae`): trained by `scripts/train_vae_policy.py` in about 11 min on 4 cores
(21,600 depth frames from 16 random tracks, mirrored; LQR expert with DART noise).
The VAE reconstructs depth to 13 cm RMS, and the steering head's error is 0.038 on held-out frames. The
closed-loop results above are on the oval / twisty tracks, which were **not** in the training
set. The model learns rendered depth, so for the car the same pipeline must be retrained
on recorded real depth frames with expert labels.

MPC steers about half as busily as LQR (steering rate RMS 1.5 vs 2.8 /s) at the same accuracy.
The depth FTG uses only rendered depth images (no pose, no map). On the real OAK-D it
produces commands at 15 Hz; in cluttered scenes it can alternate between near-equal
gaps, and a gap-hysteresis term is a candidate improvement.

For comparison, the legacy PID (heading only) left the track within about one lap at 0.84 m/s
in the same kind of test. Stanley tracks tighter on twisty tracks; Pure Pursuit
steers about half as busily. End to end through `start.py` (processes, shared memory,
arming keys) Pure Pursuit held 100 % on track at 1.33 m/s and coasted to a stop
on disarm.

Findings fixed along the way:
- A speed cap based only on the (lagging) pose speed estimate let the car
  overshoot to 2.3 m/s. The cap now also bounds forward throttle at
  `speed_cap / speed_per_throttle`, and holds at ≤ 1.5 m/s.
- Planning at exactly the cap made the planner and the cap fight (limit cycle), so the
  planner now targets 1.3 m/s.
- Curvature from raw 5 cm differences read 15 % high, so a ±0.25 m stencil is used.

## 6. Before driving the car
1. **Calibrate `speed_per_throttle`:** hold a few fixed throttles on a straight and
   record the SLAM speed. Both the speed controller and the speed-cap bound use it.
2. **Measure the vehicle:** wheelbase and max steering angle in `vehicle_params.yaml`
   (F1TENTH placeholders now). Also the camera offset from the rear axle: the
   controllers treat the pose as the rear axle.
3. **First runs:** wheels off the ground (arm → disarm response), then low
   `v_max` (e.g. 0.6 m/s) on the mapped track.
4. **Measure the camera height and pitch** for `depth_ftg` (level camera assumed).
5. **Follow-ups:** IMU in the EKF (lower speed lag); gap hysteresis for `depth_ftg`;
   train a PPO policy (`rl/train.py`) and export it for `policy` mode; retrain
   the VAE model on real recorded depth frames. Note that
   `train.py` defines `TrackCNNExtractor` but doesn't pass it to PPO (it trains a plain MLP).

## 7. Sequencing

| Phase | Scope | Outcome |
| --- | --- | --- |
| 1 | P0b safety, P3 interface, P4 harness | **Done** |
| 2 | P1 estimator, P2 track model, C1 Pure Pursuit | **Done** (bench-verified; not yet on the car) |
| 3 | C2 speed planner, C3 Stanley | **Done** (bench-verified; not yet on the car) |
| 4 | C5 policy deployment, C4 reactive depth | RL on car; tracking-loss fallback |
| 5 | C6 MPC | Lap-time optimisation |

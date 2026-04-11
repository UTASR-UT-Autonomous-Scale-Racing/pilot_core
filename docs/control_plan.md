# Control models: assessment and implementation plan

Status: **proposal.** Done so far: P0a (signed actuation / brake-reverse), the `simulator_integration` merge, and FTG selection (`control_mode: ftg`, from that branch). Everything else is not started.

## 1. Where control stands today

| Controller | Location | State |
| --- | --- | --- |
| `MeanIntensity` (default "neural") | `pilot_controller/models/default` | Placeholder: always returns `(0.5, 0.5)`. Runs an IMU-integrating EKF + matplotlib mapper as a side effect. |
| `RNN` | `pilot_controller/models/rnn/.py` | Stub (`tanh(mean(history))`). File is literally named `.py` / `.yaml`, so it can't be loaded. |
| RNN trainer | `pilot_controller/Controller/` | Torch GRU/LSTM scaffold that predicts the next *input* vector, not actions. No dataset, not wired in. |
| `TrackPIDController` | `workers/track_pid_controller.py` | Runs. Since the sim merge it aligns the heading with the local boundary tangent, but has **no cross-track term**, so lateral drift is never corrected. In a closed-loop test at 0.84 m/s it drifted off the track within about one lap (on track 43% of the time). Also: no `dt`, no anti-windup, constant throttle, Gaussian noise injected into real commands. |
| `TrackFTGController` | `workers/track_ftg_controller.py` | Implemented, unit-tested, selectable with `control_mode: ftg` (since the merge). Not yet evaluated closed-loop. |
| RL (PPO) + `LidarWallController` | `rl/`, `workers/lidar_wall_controller.py` (merged) | Gymnasium env over AutoDRIVE, PPO with a 1-D CNN over boundary points; LIDAR wall-centering PID (`control_mode: lidar_pid`). |

Fixed during the release clean-up (so they're not repeated below): the PID [-1,1] vs
actuator [0,1] range crash, the SLAM Tcw-vs-Twc pose convention, the PID heading
convention, lost-tracking frames published as the identity pose, and neural models not receiving `timestamp`.

**What's missing across the board:** a vehicle state estimate (speed in
particular), a track representation that controllers can query cheaply, a
safety layer between control and actuation, and any closed-loop way to
evaluate a controller before it touches the car.

## 2. Prerequisites (shared by every controller)

These come first. Every controller below depends on them.

### P0a. Signed actuation contract (brake + reverse) — **done**
- `ctrl_buf`: both channels `[0, 1]`, **0.5 = neutral**, car PWM convention
  (steer 0 = full left, 1 = full right; throttle < 0.5 = brake / reverse).
- `pilot_controller.Controller` maps throttle linearly over
  `[THROTTLE_MIN, THROTTLE_MAX]`, like steering; 0.5 lands exactly on
  `THROTTLE_NEURAL`. Steering offsets unchanged (intended).
- Signed controller convention: **+steering = left** (ROS / REP-103), matching
  the controllers' geometry and AutoDRIVE. `signed_to_unit()` negates steering
  into `ctrl_buf`, and the ZMQ actuator converts back. Before this, PID/FTG would
  have steered the *car* the wrong way (they only worked in the simulator).
- **Bench check still needed:** many RC ESCs need brake → neutral → reverse
  before reverse engages. The firmware passes PWM straight through, so verify the
  ESC's behaviour and, if needed, add that sequence in `ActuatorWorker`.

### P0b. Actuation safety layer: `ActuatorWorker`
- **Command watchdog:** if the newest `ctrl_buf` entry is older than `command_timeout_ms` (e.g. 150 ms), send neutral throttle and hold steering. Today a stalled controller or lost SLAM tracking leaves the car on its last command.
- **Rate / slew limits** on steering and throttle (configurable), applied after the controller.
- **Arming:** actuator starts disarmed (neutral) and only arms on explicit operator action. Replaces the current `input()` gate in `start.py`.
- Tests: watchdog timing, slew limiting, disarmed-by-default.

### P1. Vehicle state estimator: new `StateWorker` → `state_buf`
- EKF fusing SLAM pose (15 Hz, Twc) with OAK-D IMU (100 Hz: gyro yaw-rate, accel) → `VehicleState(t, x, z, yaw, v, yaw_rate)` at IMU rate.
- Gives controllers **speed** (currently unknown) and smooth, high-rate pose between SLAM frames; bridges short tracking losses with a bounded covariance.
- Reuse the idea (not the code) of `pilot_controller/lap_engine.EKFLocalization`, which double-integrates world-frame acceleration and will drift.

### P2. Track model: `TrackModel` (library, not a worker)
- Built once from `boundary_buf`: order and resample boundaries, compute the centre-line, arc length `s`, curvature `κ(s)` and left/right widths.
- `project(x, z) → (s, lateral_error, heading_error)` with index continuity (no global nearest-point search every step).
- Later: minimum-curvature race line as an alternative reference path.
- Reuse `workers/boundary_resampler.py` (merged from the sim branch).

### P3. Controller interface + selection
```python
class TrackController(Protocol):
    def reset(self) -> None: ...
    def step(self, state: VehicleState, track: TrackModel) -> Command: ...  # steer, throttle ∈ [-1,1] (P0a)
```
- `ControlWorker` gets a registry keyed by `control_mode` (`pid`, `ftg`, `lidar_pid`, `pure_pursuit`, `stanley`, `policy`, `neural`) instead of the current if/else chain.

### P4. Closed-loop evaluation harness
- Pure-Python kinematic bicycle simulator on synthetic tracks (`conftest.track_boundaries`, `synthetic/closed_loop_generator.py`) → `pytest` benchmarks: lap completion, lap time, cross-track RMS / max, steering jerk.
- Every controller must pass it before car testing. Seconds per run on CPU.
- AutoDRIVE (via the ZMQ workers) is the higher-fidelity stage. A fake ZMQ bridge with a kinematic bicycle model (F1TENTH params) already closes the loop around the unmodified pipeline. It was used to validate the merge and should become this harness.

## 3. Controllers to implement (recommended order)

### C1. Pure Pursuit, the new baseline (replaces PID as the default)
- Geometric path tracking on the `TrackModel` centre-line: look-ahead point at `L_d = clamp(k·v + L_min, L_min, L_max)`, steer `δ = atan(2·L·sin α / L_d)`, normalised by max steering angle.
- Fixes the PID's core flaw: the reference is ahead of the car, so there is no singularity on the line.
- Needs: P1 (speed), P2, wheelbase `L` and max steering angle from `vehicle_params.yaml`.
- Effort: **S** (≈1–2 days incl. tests).

### C2. Speed planner / longitudinal control
- Curvature-based velocity profile: `v_max(s) = sqrt(a_lat_max / |κ(s)|)` (with `a_lat_max ≈ μ·g` from `friction_coeff`), forward/backward pass with `max_accel` / `max_decel` limits. Throttle via a PI controller on `v_ref − v`, with feed-forward from a measured throttle→speed map. With P0a, negative output brakes, so corners can be entered at the planned speed.
- Pairs with C1/C3. Today throttle is a constant 0.3.
- Needs: P1 and a one-off throttle-map calibration run (log PWM vs. estimated speed).
- Effort: **S–M**.

### C3. Stanley controller (alternative lateral controller)
- `δ = ψ_e + atan(k·e / (k_soft + v))` on the front axle. Often better than Pure Pursuit at low speed and tight tracking.
- Same interface and inputs as C1, so it's cheap once C1 exists, and A/B-able in the harness.
- Effort: **S**.

### C4. Reactive depth controller (no map needed)
- FTG (already written) fed by a *virtual scan built from the depth image* (ground-plane removal → per-column nearest obstacle) instead of from mapped boundaries. The sim branch's `depth_projection_worker.py` does this.
- The mapping lap stays manual (decided), so this is mainly a **fallback when SLAM tracking is lost** and a map-free mode for new tracks. Lower priority.
- Effort: **M**.

### C5. Learned policy deployment (`policy` mode)
- Load a trained policy exported to **ONNX** and run it on CPU with onnxruntime (tiny MLP/1-D CNN: well under 1 ms). The observation builder reuses `TrackModel` to produce the sim branch's boundary-point observation.
- Start from the merged PPO work in `rl/`. Sim-to-real gap is the main risk; gate behind the P0b safety layer and the P4 harness.
- Optional follow-on: behaviour cloning from `BufferRecorderWorker` recordings of C1/C2 laps (fixes the `pilot_controller` RNN scaffold into an action-predicting model).
- Effort: **M** for deployment; training effort is open-ended.

### C6 (later). MPC
- Kinematic-bicycle MPC (CasADi/acados) tracking the race line with input constraints. Best lap-time potential, but needs an identified vehicle model and more CPU. Revisit after C1–C3 are characterised on the car.
- Effort: **L**.

## 4. Decisions
1. **Steering / throttle sign (22 Sep):** set by the neutral values in the control
   config: below neutral = left / brake-reverse, above = right / forward. → P0a.
2. **Vehicle parameters (22 Sep):** F1TENTH simulator values for now
   (`vehicle_params.yaml`, placeholders until the car is measured).
3. **Mapping lap (22 Sep):** driven by hand. C4 drops in priority.
4. **Braking and reverse (22 Sep):** supported and to be used. → P0a, done.
5. **Simulator:** AutoDRIVE. `simulator_integration` merged; the PyBullet
   `scale_racing_simulator` is outdated and ignored.
6. **Steering offsets:** intended. Linear `[MIN, MAX]` mapping kept.

### Simulator follow-ups found during the merge
- `config_zmq.yaml` references `pilot_vision/models/deeplab/config.yaml`, which only
  exists on `pilot_vision`'s `deeplab` branch (the submodule is pinned to `main`).
  Not needed in the default sim mode (LiDAR mapping skips vision).
- `simulator_orbslam_settings.yaml` is referenced but was never committed (only
  needed for SLAM-in-sim).
- The sim camera path delivers **RGB** frames while the OAK-D path delivers **BGR**.
  Left as is until it's clear how the sim-trained vision model expects its input.

## 5. Suggested sequencing

| Phase | Scope | Outcome |
| --- | --- | --- |
| 1 | P0b safety, P3 interface, P4 harness | Safe to test any controller; benchmarks in CI |
| 2 | P1 estimator, P2 track model, C1 Pure Pursuit | Reliable autonomous laps on a mapped track |
| 3 | C2 speed planner, C3 Stanley | Faster laps; lateral controller A/B |
| 4 | C5 policy deployment, C4 reactive depth | RL on car; tracking-loss fallback |
| 5 | C6 MPC | Lap-time optimisation |

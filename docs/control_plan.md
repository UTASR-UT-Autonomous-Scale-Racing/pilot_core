# Control models: assessment and implementation plan

Status: **proposal, awaiting a decision.** Nothing in this document has been implemented yet.

## 1. Where control stands today

| Controller | Location | State |
| --- | --- | --- |
| `MeanIntensity` (default "neural") | `pilot_controller/models/default` | Placeholder: always returns `(0.5, 0.5)`. Runs an IMU-integrating EKF + matplotlib mapper as a side effect. |
| `RNN` | `pilot_controller/models/rnn/.py` | Stub (`tanh(mean(history))`). File is literally named `.py` / `.yaml`, so it can't be loaded. |
| RNN trainer | `pilot_controller/Controller/` | Torch GRU/LSTM scaffold that predicts the next *input* vector, not actions. No dataset, not wired in. |
| `TrackPIDController` | `workers/track_pid_controller.py` | Runs, but: target = nearest centre-line point (no look-ahead), so the error is ill-defined when on the line; no `dt`; no anti-windup; constant throttle; Gaussian noise injected into real commands. |
| `TrackFTGController` | `workers/track_ftg_controller.py` | Implemented and unit-tested, but **not selectable**. `ControlWorker` only knows `neural` / `pid`. |
| RL (PPO) + `LidarWallController` | `simulator_integration` branch (unmerged) | Gymnasium env over the AutoDRIVE simulator, PPO with a 1-D CNN over boundary points. Diverged from this branch (4 commits). |

Fixed during the release clean-up (so they're not repeated below): the PID [-1,1] vs
actuator [0,1] range crash, the SLAM Tcw-vs-Twc pose convention, the PID heading
convention, lost-tracking frames published as the identity pose, and neural models not receiving `timestamp`.

**What's missing across the board:** a vehicle state estimate (speed in
particular), a track representation that controllers can query cheaply, a
safety layer between control and actuation, and any closed-loop way to
evaluate a controller before it touches the car.

## 2. Prerequisites (shared by every controller)

These come first. Every controller below depends on them.

### P0a. Signed actuation contract (enables braking + reverse)
PWM is defined relative to the neutral values in `pilot_controller/control_output.py`:
below neutral = left / brake-reverse, above neutral = right / forward.
Today `Controller` maps throttle `[0,1]` onto neutral→max only (`THROTTLE_MIN`
is never used), and `ControlWorker` clips negative throttle to 0, so **software
cannot brake or reverse**. Steering is mapped linearly over `[MIN, MAX]`, so
"centre" (0.5) lands at 1465 µs instead of the configured 1461 µs.

- Change `ctrl_buf` to **signed** `steering, throttle ∈ [-1, 1]` (0 = neutral for both).
- Map piecewise around neutral: `pwm = NEUTRAL + u·(MAX−NEUTRAL)` for `u ≥ 0`,
  `pwm = NEUTRAL + u·(NEUTRAL−MIN)` for `u < 0`, so 0 is exactly the configured neutral
  and the full range of each side is reachable.
- Touches: `pilot_controller/control_output.py` (submodule), `ActuatorWorker`,
  `ControlWorker` (drop `signed_to_unit`), `ControlBuffer` docs, an adapter for
  `[0,1]` neural models, tests. PID/FTG/RL already produce signed output.
- Bench check: many RC ESCs need brake → neutral → reverse before reverse engages.
  The firmware passes PWM straight through, so verify the ESC's behaviour and, if
  needed, add that sequence in `ActuatorWorker`.

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
- The `simulator_integration` branch already has `boundary_resampler.py`, which is worth porting.

### P3. Controller interface + selection
```python
class TrackController(Protocol):
    def reset(self) -> None: ...
    def step(self, state: VehicleState, track: TrackModel) -> Command: ...  # steer, throttle ∈ [-1,1] (P0a)
```
- `ControlWorker` gets a registry keyed by `control_mode` (`pid`, `pure_pursuit`, `stanley`, `ftg`, `policy`, `neural`) instead of the current if/else.
- Wire the existing FTG in on day one.

### P4. Closed-loop evaluation harness
- Pure-Python kinematic bicycle simulator on synthetic tracks (`conftest.track_boundaries`, `synthetic/closed_loop_generator.py`) → `pytest` benchmarks: lap completion, lap time, cross-track RMS / max, steering jerk.
- Every controller must pass it before car testing. Seconds per run on CPU.
- AutoDRIVE / `scale_racing_simulator` (via the sim branch's ZMQ workers) is the higher-fidelity stage.

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
- Start from the PPO work on `simulator_integration` (merge/rebase that branch first). Sim-to-real gap is the main risk; gate behind the P0b safety layer and the P4 harness.
- Optional follow-on: behaviour cloning from `BufferRecorderWorker` recordings of C1/C2 laps (fixes the `pilot_controller` RNN scaffold into an action-predicting model).
- Effort: **M** for deployment; training effort is open-ended.

### C6 (later). MPC
- Kinematic-bicycle MPC (CasADi/acados) tracking the race line with input constraints. Best lap-time potential, but needs an identified vehicle model and more CPU. Revisit after C1–C3 are characterised on the car.
- Effort: **L**.

## 4. Decisions (22 Sep 2026)
1. **Steering / throttle sign:** set by the neutral values in the control config.
   Below neutral = left / brake-reverse, above neutral = right / forward. → P0a.
2. **Vehicle parameters:** use the F1TENTH simulator values for now
   (`vehicle_params.yaml`, marked as placeholders until the car is measured).
3. **Mapping lap:** driven by hand. C4 drops in priority.
4. **Braking and reverse:** supported by the car. → P0a.

**Still open**
- **Simulator:** AutoDRIVE (Unity, via ZMQ, on `simulator_integration`, Feb–Apr 2026)
  vs. the older PyBullet `scale_racing_simulator` (Apr 2025, 2 commits). P4's kinematic
  simulator is needed either way for fast CI checks.
- **`simulator_integration` branch:** merge it into this branch (brings RL, ZMQ,
  LiDAR, depth projection) or keep it separate?

## 5. Suggested sequencing

| Phase | Scope | Outcome |
| --- | --- | --- |
| 1 | P0a signed actuation, P0b safety, P3 interface (+ FTG wiring), P4 harness | Brake/reverse available; safe to test any controller; benchmarks in CI |
| 2 | P1 estimator, P2 track model, C1 Pure Pursuit | Reliable autonomous laps on a mapped track |
| 3 | C2 speed planner, C3 Stanley | Faster laps; lateral controller A/B |
| 4 | C5 policy deployment, C4 reactive depth | RL on car; tracking-loss fallback |
| 5 | C6 MPC | Lap-time optimisation |

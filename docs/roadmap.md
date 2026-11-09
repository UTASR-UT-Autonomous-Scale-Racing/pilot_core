# Roadmap

Where the work stands and what is next. Results: [results.md](results.md). The gaps between
the sim and the car, axis by axis: [sim_to_real.md](sim_to_real.md).

## Done

- **One pipeline.** The car and the sim map and drive with the same segmentation, scan, mapping,
  control, state and actuator workers; a test keeps it that way.
- **Camera-only mapping and racing in the sim.** Walls from an occupancy grid over the
  segmentation-edge scans, in the loop: the practice track maps cleanly in one lap and races 10
  laps without a hit; the compete track maps every time ([results.md](results.md)).
- **A SLAM-like sim pose** (`sim_pose_degrade`) for testing the controllers against late, noisy,
  drifting poses with dropouts.
- **Deployment tooling.** `scripts/jetson_selftest.py` before every start, `scripts/stage_latency.py`
  for per-stage latency from any recording, deploys from `car-v*` tags, the Jetson image built in
  CI ([deployment.md](deployment.md)).

## Car

Each step needs the car, the Jetson and the track; in this order, nothing moves a wheel before
steps 1 and 2 pass.

1. **Firmware bench test.** Compile (`arduino-cli compile -b arduino:esp32:nano_nora firmware/`)
   and run the five wheels-off checks in `firmware/README.md`: RC only, Jetson only, stick
   override, Jetson timeout (servo to neutral within 250 ms), transmitter off. Then remove its warning.
2. **Jetson bring-up.** `scripts/jetson_selftest.py` all PASS; OAK-D on USB 3; regenerate
   `orbslam_oakd_rgbd.yaml` at the car's frame rate (30 fps requested, the file says 15); try the
   Pro's IR dot projector for depth indoors; pin `depthai` ([oakd.md](oakd.md#open-items)).
3. **Measure on the Orin.** A recorded run through `scripts/stage_latency.py`, `SLAMWorker`'s
   tracking time, CPU / GPU load and thermals at maximum performance, next to the sim's table in
   [results.md](results.md#stage-latency).
4. **Calibrate.** `speed_per_throttle` from a straight run, `vehicle_params.yaml` (F1TENTH
   placeholders now), and the camera height and pitch (SLAM logs them at start).
5. **Hand-driven recordings.** Five or more RC laps with `--recording`; bootstrap masks from them,
   correct a few hundred by hand, train the segmentation on real frames and score it on a held-out
   lap ([datasets.md](datasets.md#still-to-collect)). Until then the car uses `ground_plane`.
6. **ORB-SLAM3 on the recordings.** Tracked frames and ATE against AprilTags (`scripts/vio_bench`);
   refit `sim_pose_degrade` from the measured latency, noise and dropouts. Drive the rear-IMU
   estimator; `lqr` / `mpc` stay sim-only until its yaw rate is measured.
7. **Static scan check.** The car standing still, segmentation-edge scan against tape-measured
   wall distances at 0.5, 1, 2 and 3 m.
8. **Autonomous lap.** Mapping lap on the scan, then three laps of `pure_pursuit` with the speed cap
   at 1.0 m/s; a kill test at low speed (SIGKILL every process, the firmware timeout stops the car).
9. **Release.** Tag the version that did it `car-v1.0.0` ([deployment.md](deployment.md#releasing-a-car-version));
   the Jetson runs that tag from then on.

## Simulator

1. **Race settings against the degraded pose.** With `sim_pose_degrade` on, the compete race goes
   from 0–1 to 12–15 collisions and 2–3 s slower laps, and latency and dropouts are the cause, not
   noise ([results.md](results.md#degraded-pose-2026-11-09)). Separate latency from dropouts, then
   tune the speed profile and lookahead (or the estimator) until the degraded race is clean; that is
   the setting the car should start from.
2. **Compete-track racing lines on live maps.** The live maps are as accurate as the offline one,
   but their racing lines pass 0.33–0.42 m from a wall (0.45 m offline) and hit the left hairpins.
   Keep the line 0.45 m off on any map, or map two laps before racing.
3. **Policy collisions.** About 30 % of live episodes collide, most likely because
   `rl/bicycle_env.py` models the car as circles of r = 0.14 m. Retrain with the real footprint
   and a wall margin, then benchmark against `pure_pursuit` at the same speed cap.
4. **ORB-SLAM3 in the sim** needs Joshua Choi's build (a front camera with depth); the stock
   camera cannot support it ([results.md](results.md#what-did-not-work)). Until then the degraded
   pose stands in.
5. **Sim-side neutral on disconnect.** If every pilot_core process dies, the stock sim keeps
   applying the last command. An issue for the simulator's repository (`Tinker-Twins/AutoDRIVE`)
   is drafted in the maintainers' notes.

## Repository

- `scripts/autodrive_texture_sim.py` still hard-codes `/workspaces`; the other sim scripts read
  `AUTODRIVE_ROOT` and `AUTODRIVE_LOG_DIR`.

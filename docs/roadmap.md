# Roadmap

Where the work stands and what is next. Results so far: [results.md](results.md).

## Car (real hardware)

1. **OAK-D on USB 3** and a matching SLAM settings file at the car's frame rate
   ([oakd.md](oakd.md#open-items)).
2. **First recordings on the track**: hand-driven mapping laps (`--recording`), then BC
   data and a real segmentation set from them ([datasets.md](datasets.md#still-to-collect)).
3. **Autonomous mapping lap** on the segmentation-edge scan (no LiDAR on the car): needs the
   real segmentation model from (2), `track_control.mapping_drive` and `vision.enabled`.
   Before driving it, check the scan with the car standing still against tape-measured wall distances.
4. **Before driving**: calibrate `speed_per_throttle`, measure `vehicle_params.yaml` (still
   F1TENTH placeholders) and the camera height and pitch, and bench-test arm / disarm and
   the steering direction with the wheels off the ground.

## Simulator

1. **Policy collisions.** About 30 % of live episodes collide, most likely because
   `rl/bicycle_env.py` models the car as circles of r = 0.14 m. Retrain with the real
   footprint and a wall margin, then benchmark against pure_pursuit at the same speed cap
   (3 runs × 2 laps, practice and compete).
2. **Drive the segmentation-edge scan live** in the sim (`seg_scan.source: segmentation`): it
   is only scored offline so far.
3. **ORB-SLAM3 in the sim** needs Joshua Choi's build (a front camera with depth). The stock
   camera cannot support it ([results.md](results.md#what-did-not-work)).
4. **Sim-side command timeout.** If every pilot_core process dies, the stock sim keeps
   applying the last command. Raise this upstream.
5. Optional: re-run controller × v_max on a GPU without other jobs (that stage had 0–90 %
   sim stalls), and find out why on-the-fly LiDAR maps lap about 0.3 s faster than the
   saved map.

## Repository

- **Sim location.** The sim scripts default to `/workspaces/autodrive_sim` and
  `/workspaces/logs` (`autodrive_bench.py`, `autodrive_texture_sim.py`, the supervisor
  scripts). Move these to one setting or environment variable.
- Upload the data bundles ([datasets.md](datasets.md)) and choose a license.

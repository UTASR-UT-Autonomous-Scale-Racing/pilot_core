# RGB-D-inertial, sim: ORB-SLAM3 (CUDA fork) vs cuVSLAM 17.0.1 (2026-10-29)

> **Superseded.** These are the first runs: replayed ~3x faster than real time on an
> unpinned, CPU-throttled container, with the upstream inertial policy. The ORB-SLAM3 numbers
> below are not current; see [inertial_fix.md](inertial_fix.md) (real-time, pinned, fork fixes).
> Caveats for all VIO numbers here: the sim and rendered sequences are synthetic (rendered
> images, exact depth and the IMU all come from the same ground truth), ORB-SLAM3's IMU noise
> scale was chosen on the sequences it is scored on, and cuVSLAM had one run per sequence.

Sequence: textured AutoDRIVE compete track, 3 race laps (10.4-10.9 s each, up to ~8 m/s),
31.7 s, 1297 RGB-D frames at ~53 fps (384x216: the sim camera upscaled, depth from the
LiDAR), synthetic 200 Hz IMU. vast.ai box: NVIDIA L40S, AMD EPYC 7713. 3 repeats, mean ±
sd. How the sequence is built and its limits: [scripts/vio_bench/README.md](../../scripts/vio_bench/README.md).

| run | tracked | ATE RMSE (m) | drift over 1 s | rot. drift (deg / 1 s) | latency mean / p50 / p95 / p99 / max (ms) |
| --- | --- | --- | --- | --- | --- |
| ORB-SLAM3 CUDA, `IMU_RGBD` | 49.1 ± 0.2 % | 6.62 ± 0.00 | 4.70 m, ~100 % | 60.5 | 9.7 / 8.6 / 30.6 / 55.3 / 79.6 |
| ORB-SLAM3 CPU, `IMU_RGBD` | 49.6 ± 0.1 % | 6.61 ± 0.01 | 4.71 m, ~100 % | 59.3 | 21.7 / 12.7 / 69.7 / 80.2 / 94.0 |
| ORB-SLAM3 CUDA, `RGBD` | 42.5 ± 4.1 % | 5.54 ± 0.72 | 5.44 m, 49 % | 95.4 | 10.0 / 7.2 / 35.4 / 44.9 / 57.9 |
| cuVSLAM, Multisensor RGB-D + IMU + SLAM | 100 % | 5.15 ± 0.91 | 0.86 m, 15.1 % | 7.2 | 4.0 / 3.6 / 5.7 / 8.8 / 190 |
| cuVSLAM, `RGBD` + SLAM | 100 % | 4.87 ± 0.45 | 2.15 m, 17.5 % | 20.9 | 3.5 / 3.2 / 5.8 / 8.7 / 25.4 |

- **ORB-SLAM3 `IMU_RGBD` never initialised its IMU** (cause and fork fix: [inertial_fix.md](inertial_fix.md)). Within ~1 s after starting it needs
  50 inliers per frame; these images often give fewer, and in the inertial modes any
  tracking failure before IMU initialisation (and before inertial BA 2) resets the active
  map (`Tracking.cc`, "IMU is not or recently initialized"). ~150 resets per run; more
  features (2000 / 3000) and 10-300x IMU noise did not break the loop. `RGBD` mode, which
  relocalises instead, tracks for seconds at a time.
- **cuVSLAM tracked every frame**, and the IMU cut its rotation drift 3x (7.2 vs 20.9
  deg/s). Its speed is ~15 % low against ground truth with zero lag and 0.98 turn-rate
  correlation: the sim depth's scale error, which inflates every ATE here.
- **Latency:** cuVSLAM's tracking call is ~2.4x faster at the median (3.6 vs 8.6 ms) and
  ~5x at p95 (5.7 vs 30.6 ms), except one ~190 ms call when its IMU fusion starts.
  ORB-SLAM3's CUDA path vs its CPU path: 8.6 vs 12.7 ms median, 30.6 vs 69.7 ms p95.
  At the car's 640x400 these grow; measure on the Orin.
- Not a verdict on accuracy: the sim depth, IMU and textures are synthetic. The real-track
  recordings are.

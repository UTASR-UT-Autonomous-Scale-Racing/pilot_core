# AutoDRIVE benchmark results

Each cell: repeated `start.py` runs on the live stock sim (practice track unless noted), racing a saved LiDAR map after a reset to the start line. Lap times are the sim's own (sim clock, which pauses while the sim is frozen); lap 1 of a run is a standing start. Cross-track error is the sim's own pose against the map raced on. mean ± sd over repeats.

![summary](summary.png)

## Stage A

| cell | laps | best lap (s) | mean lap (s) | collisions | xtrack RMS (cm) | xtrack max (cm) | off-track % | speed mean (m/s) | frames/s | sim frozen % | pose age p50/p95 (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control_mode=pure_pursuit, v_max=4.0 | 2.0 ± 0.0 | 13.41 ± 0.93 | 14.22 ± 0.54 | 0.0 ± 0.0 | 7.7 ± 0.9 | 17.7 ± 2.5 | 0.0 ± 0.0 | 2.27 ± 0.06 | 15.4 ± 5.8 | 17 ± 29 | 31 ± 4 / 152 ± 148 |
| control_mode=mpc, v_max=4.0 | 2.0 ± 0.0 | 13.28 ± 0.08 | 14.35 ± 0.13 | 0.0 ± 0.0 | 6.1 ± 0.2 | 22.4 ± 2.7 | 0.0 ± 0.0 | 2.27 ± 0.02 | 10.2 ± 7.4 | 30 ± 52 | 38 ± 1 / 135 ± 94 |
| control_mode=pure_pursuit, v_max=2.5 | 2.0 ± 0.0 | 13.83 ± 0.01 | 14.57 ± 0.44 | 0.0 ± 0.0 | 7.1 ± 0.1 | 17.3 ± 1.6 | 0.0 ± 0.0 | 2.18 ± 0.03 | 17.0 ± 1.7 | 8 ± 12 | 30 ± 1 / 92 ± 57 |
| control_mode=lqr, v_max=1.5 | 2.0 ± 0.0 | 20.83 ± 0.07 | 21.12 ± 0.44 | 0.0 ± 0.0 | 2.4 ± 0.3 | 10.7 ± 2.6 | 0.0 ± 0.0 | 1.49 ± 0.02 | 18.8 ± 0.7 | 2 ± 4 | 28 ± 1 / 60 ± 3 |
| control_mode=pure_pursuit, v_max=1.5 | 2.0 ± 0.0 | 20.58 ± 0.00 | 21.26 ± 0.03 | 0.0 ± 0.0 | 4.1 ± 0.1 | 9.8 ± 0.2 | 0.0 ± 0.0 | 1.48 ± 0.01 | 17.7 ± 3.0 | 0 ± 0 | 32 ± 5 / 72 ± 25 |
| control_mode=mpc, v_max=1.5 | 2.0 ± 0.0 | 20.44 ± 0.09 | 21.31 ± 0.24 | 0.0 ± 0.0 | 2.1 ± 0.2 | 9.7 ± 0.0 | 0.0 ± 0.0 | 1.49 ± 0.02 | 10.5 ± 0.7 | 32 ± 5 | 38 ± 2 / 98 ± 13 |
| control_mode=stanley, v_max=1.5 | 2.0 ± 0.0 | 23.00 ± 4.09 | 23.59 ± 3.93 | 0.0 ± 0.0 | 4.1 ± 0.9 | 15.3 ± 5.4 | 0.0 ± 0.0 | 1.40 ± 0.14 | 9.9 ± 7.9 | 46 ± 41 | 39 ± 10 / 317 ± 221 |
| control_mode=pure_pursuit, v_max=1.0 | 2.0 ± 0.0 | 30.52 ± 0.77 | 31.02 ± 0.67 | 0.0 ± 0.0 | 2.8 ± 0.1 | 9.4 ± 0.0 | 0.0 ± 0.0 | 1.01 ± 0.02 | 15.1 ± 6.2 | 15 ± 25 | 34 ± 7 / 140 ± 133 |
| control_mode=stanley, v_max=1.0 | 2.0 ± 0.0 | 30.37 ± 0.36 | 31.18 ± 0.07 | 0.0 ± 0.0 | 2.9 ± 0.1 | 9.7 ± 0.0 | 0.0 ± 0.0 | 1.00 ± 0.00 | 15.8 ± 1.2 | 13 ± 5 | 31 ± 1 / 86 ± 16 |
| control_mode=lqr, v_max=1.0 | 2.0 ± 0.0 | 30.88 ± 0.36 | 31.50 ± 0.12 | 0.0 ± 0.0 | 1.5 ± 0.3 | 9.5 ± 0.1 | 0.0 ± 0.0 | 1.00 ± 0.01 | 15.5 ± 5.1 | 17 ± 20 | 32 ± 6 / 138 ± 139 |
| control_mode=mpc, v_max=1.0 | 2.0 ± 0.0 | 31.29 ± 0.61 | 32.92 ± 2.56 | 0.0 ± 0.0 | 1.8 ± 0.5 | 9.8 ± 0.2 | 0.0 ± 0.0 | 0.97 ± 0.05 | 13.4 ± 8.1 | 27 ± 41 | 34 ± 9 / 187 ± 195 |
| control_mode=mpc, v_max=2.5 | 1.7 ± 0.6 | 14.71 ± 1.15 | 15.36 ± 0.58 | 0.0 ± 0.0 | 4.2 ± 0.2 | 13.9 ± 1.7 | 0.0 ± 0.0 | 2.15 ± 0.00 | 5.9 ± 8.1 | 30 ± 52 | 38 ± 1 / 81 ± 7 |
| control_mode=lqr, v_max=4.0 | 2.0 ± 0.0 | 13.63 ± 0.25 | 14.61 ± 0.25 | 0.3 ± 0.6 | 9.8 ± 0.8 | 35.1 ± 6.1 | 0.0 ± 0.0 | 2.28 ± 0.04 | 19.4 ± 0.5 | 1 ± 1 | 28 ± 0 / 56 ± 2 |
| control_mode=stanley, v_max=2.5 | 2.0 ± 0.0 | 14.16 ± 0.11 | 15.11 ± 0.05 | 0.3 ± 0.6 | 7.5 ± 0.4 | 29.1 ± 8.7 | 0.0 ± 0.0 | 2.17 ± 0.01 | 17.6 ± 3.7 | 8 ± 15 | 30 ± 3 / 103 ± 82 |
| control_mode=lqr, v_max=2.5 | 2.0 ± 0.0 | 14.76 ± 0.53 | 15.64 ± 0.29 | 0.7 ± 0.6 | 9.1 ± 1.9 | 33.7 ± 2.0 | 0.0 ± 0.0 | 2.11 ± 0.04 | 17.7 ± 2.1 | 7 ± 12 | 29 ± 1 / 68 ± 15 |
| control_mode=stanley, v_max=4.0 | 2.0 ± 0.0 | 13.59 ± 0.14 | 14.99 ± 0.52 | 1.3 ± 2.3 | 11.5 ± 4.1 | 35.3 ± 9.8 | 0.0 ± 0.0 | 2.26 ± 0.02 | 14.7 ± 1.7 | 12 ± 10 | 34 ± 4 / 87 ± 14 |

## Stage B

| cell | laps | best lap (s) | mean lap (s) | collisions | xtrack RMS (cm) | xtrack max (cm) | off-track % | speed mean (m/s) | frames/s | sim frozen % | pose age p50/p95 (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control_mode=pure_pursuit, v_max=4.0, control_rate_hz=50 | 2.0 ± 0.0 | 12.90 ± 0.00 | 13.96 ± 0.00 | 0.0 ± 0.0 | 7.0 ± 0.1 | 15.2 ± 0.4 | 0.0 ± 0.0 | 2.30 ± 0.02 | 16.7 ± 0.3 | 0 ± 0 | 34 ± 0 / 68 ± 2 |
| control_mode=pure_pursuit, v_max=4.0, control_rate_hz=100 | 2.0 ± 0.0 | 12.89 ± 0.00 | 13.96 ± 0.01 | 0.0 ± 0.0 | 7.4 ± 0.2 | 15.8 ± 0.5 | 0.0 ± 0.0 | 2.29 ± 0.01 | 15.2 ± 0.8 | 0 ± 0 | 35 ± 1 / 69 ± 2 |
| control_mode=pure_pursuit, v_max=4.0, control_rate_hz=30 | 2.0 ± 0.0 | 12.90 ± 0.02 | 13.98 ± 0.00 | 0.0 ± 0.0 | 6.4 ± 0.1 | 14.5 ± 0.3 | 0.0 ± 0.0 | 2.29 ± 0.01 | 16.6 ± 0.5 | 0 ± 0 | 34 ± 1 / 68 ± 1 |
| control_mode=mpc, v_max=4.0, control_rate_hz=50 | 2.0 ± 0.0 | 13.14 ± 0.01 | 14.23 ± 0.02 | 0.0 ± 0.0 | 3.8 ± 0.3 | 13.5 ± 1.3 | 0.0 ± 0.0 | 2.28 ± 0.01 | 17.4 ± 0.1 | 0 ± 0 | 32 ± 1 / 62 ± 2 |
| control_mode=mpc, v_max=4.0, control_rate_hz=100 | 2.0 ± 0.0 | 13.15 ± 0.01 | 14.24 ± 0.02 | 0.0 ± 0.0 | 3.8 ± 0.5 | 11.8 ± 3.6 | 0.0 ± 0.0 | 2.28 ± 0.02 | 16.1 ± 0.4 | 0 ± 0 | 34 ± 1 / 65 ± 2 |
| control_mode=mpc, v_max=4.0, control_rate_hz=30 | 2.0 ± 0.0 | 13.22 ± 0.07 | 14.32 ± 0.07 | 0.0 ± 0.0 | 5.8 ± 1.2 | 19.5 ± 5.1 | 0.0 ± 0.0 | 2.28 ± 0.01 | 15.9 ± 0.6 | 0 ± 0 | 34 ± 2 / 67 ± 3 |

## Stage C

| cell | laps | best lap (s) | mean lap (s) | collisions | xtrack RMS (cm) | xtrack max (cm) | off-track % | speed mean (m/s) | frames/s | sim frozen % | pose age p50/p95 (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control_mode=pure_pursuit, v_max=4.0, control_rate_hz=50, pose_map=saved_map | 2.0 ± 0.0 | 12.90 ± 0.01 | 13.94 ± 0.06 | 0.0 ± 0.0 | 7.0 ± 0.1 | 15.9 ± 0.1 | 0.0 ± 0.0 | 2.29 ± 0.00 | 16.3 ± 0.7 | 0 ± 0 | 33 ± 1 / 73 ± 3 |
| control_mode=pure_pursuit, v_max=4.0, control_rate_hz=50, pose_map=lidar_mapping | 2.0 ± 0.0 | 12.63 ± 0.25 | 14.62 ± 0.31 | 0.0 ± 0.0 | 7.2 ± 0.1 | 15.8 ± 0.8 | 0.0 ± 0.0 | 2.36 ± 0.03 | 17.0 ± 0.6 | 0 ± 0 | 32 ± 1 / 69 ± 5 |
| control_mode=pure_pursuit, v_max=4.0, control_rate_hz=50, pose_map=orbslam3_cuda | FAILED: replay ATE 2.5-2.7 m over an 8.5 m lap, Sim(3) scale 42-56 (metric RGB-D should be 1): features are horizon and view-dependent floor glare, the car fills ~55 % of the 192x108 view | | | | | | | | | | |
| control_mode=pure_pursuit, v_max=4.0, control_rate_hz=50, pose_map=orbslam3_cpu | FAILED: as orbslam3_cuda (High quality, CPU: 50 % tracked, 12 maps, ATE 2.67 m, scale 67) | | | | | | | | | | |

## Stage D

| cell | laps | best lap (s) | mean lap (s) | collisions | xtrack RMS (cm) | xtrack max (cm) | off-track % | speed mean (m/s) | frames/s | sim frozen % | pose age p50/p95 (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control_mode=pure_pursuit, v_max=4.0, control_rate_hz=100, track=practice | 2.0 ± 0.0 | 12.89 ± 0.01 | 13.96 ± 0.01 | 0.0 ± 0.0 | 7.5 ± 0.1 | 16.0 ± 0.4 | 0.0 ± 0.0 | 2.29 ± 0.01 | 15.9 ± 0.4 | 0 ± 0 | 34 ± 0 / 67 ± 2 |
| control_mode=pure_pursuit, v_max=4.0, control_rate_hz=50, track=practice | 2.0 ± 0.0 | 12.89 ± 0.01 | 13.98 ± 0.03 | 0.0 ± 0.0 | 7.0 ± 0.1 | 15.1 ± 0.6 | 0.0 ± 0.0 | 2.29 ± 0.02 | 15.9 ± 1.8 | 0 ± 0 | 34 ± 2 / 66 ± 2 |
| control_mode=pure_pursuit, v_max=4.0, control_rate_hz=50, track=compete | 2.0 ± 0.0 | 19.54 ± 0.00 | 20.73 ± 0.02 | 0.0 ± 0.0 | 11.9 ± 0.2 | 26.8 ± 0.1 | 0.0 ± 0.0 | 2.23 ± 0.03 | 19.2 ± 0.9 | 0 ± 0 | 29 ± 1 / 57 ± 3 |
| control_mode=pure_pursuit, v_max=4.0, control_rate_hz=100, track=compete | 2.0 ± 0.0 | 19.53 ± 0.01 | 20.74 ± 0.03 | 0.0 ± 0.0 | 12.4 ± 0.4 | 27.9 ± 0.8 | 0.0 ± 0.0 | 2.22 ± 0.01 | 18.2 ± 1.0 | 0 ± 0 | 31 ± 2 / 62 ± 5 |
| control_mode=pure_pursuit, v_max=4.0, control_rate_hz=50, track=joshua | SKIPPED: needs Joshua's build (not available) | | | | | | | | | | |
| control_mode=pure_pursuit, v_max=4.0, control_rate_hz=100, track=joshua | SKIPPED: needs Joshua's build (not available) | | | | | | | | | | |


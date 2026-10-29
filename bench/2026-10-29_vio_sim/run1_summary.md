# RGB-D-inertial benchmark: compete_textured_race

1297 frames, 31.7 s; AutoDRIVE sim run /workspace/bench/runs/compete_textured_race

| run | mode | GPU | tracked | first pose (s) | IMU init (s) | ATE RMSE (m) | ATE max (m) | drift 1 s (m / %) | rot drift 1 s (deg) | latency mean / p50 / p95 / p99 / max (ms) |
|---|---|---|---|---|---|---|---|---|---|---|
| orb_cuda_imu | IMU_RGBD | NVIDIA L40S | 49.3 % | 0.06 | – | 6.622 | 10.220 | 4.698 / 100.8 | 60.48 | 9.4 / 8.4 / 28.0 / 47.6 / 78.0 |
| orb_cpu_imu | IMU_RGBD | NVIDIA L40S | 49.4 % | 0.06 | – | 6.598 | 10.230 | 4.725 / 100.7 | 59.22 | 21.7 / 12.6 / 70.2 / 81.9 / 94.0 |
| orb_cuda_rgbd | RGBD | NVIDIA L40S | 48.3 % | 0.00 | – | 5.688 | 13.113 | 6.562 / 55.0 | 89.80 | 10.0 / 7.1 / 35.9 / 45.4 / 61.9 |
| cuv_imu | Multisensor RGB-D + IMU + SLAM | NVIDIA L40S | 100.0 % | 0.00 | – | 4.093 | 10.285 | 0.864 / 15.5 | 6.26 | 3.9 / 3.6 / 5.4 / 8.5 / 186.0 |
| cuv_rgbd | RGBD + SLAM | NVIDIA L40S | 100.0 % | 0.00 | – | 4.233 | 7.418 | 1.405 / 17.1 | 19.23 | 3.5 / 3.2 / 5.7 / 8.5 / 19.4 |

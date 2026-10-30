# Dual IMU, sim: rear-of-car IMU + ORB-SLAM3 in a vehicle-state EKF (2026-10-29)

Option A of the dual-IMU plan: ORB-SLAM3 keeps the OAK-D's IMU (device-timestamped with the
frames); the rear IMU (the Arduino's ICM-20948) drives `control.ImuEKF`, corrected by the
SLAM poses, for the controller's pose, speed, yaw rate and sideslip.

- Rear IMU: `scripts/vio_bench/make_rear_imu.py`, at the rear axle, ICM-20948 noise and
  post-calibration bias, its ~24 Hz DLPF, the Jetson polling the latest sample at 100 Hz
  with 1 ms jitter, stamped when the read returns (as `IMUWorker` does today).
- Replay and scoring: `scripts/vio_bench/fuse_rear_imu.py`. SLAM poses arrive at frame
  time + their measured tracking time + 5 ms; every rear-IMU tick reads what each
  estimator would hand the controller then, against ground truth. The rear IMU's time
  offset is estimated first by cross-correlating its yaw rate with the camera IMU's
  (found 8.5 ms: the DLPF delay plus the read).
- SLAM: the ORB-SLAM3 RGB-D-inertial runs of the fork (7121458), real-time replay.

ImuEKF settings: no-slip pseudo-measurement vy_std 0.1 m/s (truth at the rear axle: vy RMS
0.015 m/s), pose gate chi2 16.3, re-anchor to SLAM after 2 s of rejected poses (0.5 s
re-anchored onto SLAM's wrong poses after a blackout; 1-3 s: same elsewhere).

Medians over runs (clean 12, blackouts at 3/9 s 12, blackouts at 20/26 s 6):

| laps | estimator | pos RMSE / max (m) | pos in SLAM gaps (m) | yaw RMSE (deg) | speed RMSE (m/s) | yaw rate RMSE (deg/s) |
|---|---|---|---|---|---|---|
| clean | ImuEKF (rear IMU + SLAM) | **0.105** / 0.25 | – | **0.91** | **0.081** | **0.74** |
| clean | PoseEKF (the car today) | 0.116 / 0.27 | – | 2.24 | 0.603 | 32.6 |
| clean | latest SLAM pose | 0.168 / 0.47 | – | 2.20 | – | – |
| blackouts 3 s, 9 s | ImuEKF | **0.132** / 0.34 | **0.121** | **0.98** | **0.088** | **0.73** |
| blackouts 3 s, 9 s | PoseEKF | 0.388 / 3.90 | 0.809 | 5.21 | 1.054 | 41.1 |
| blackouts 3 s, 9 s | latest SLAM pose | 0.485 / 3.88 | 1.412 | 5.18 | – | – |
| blackouts 20 s, 26 s | ImuEKF | **0.131** / 0.28 | **0.090** | **1.11** | **0.078** | **0.76** |
| blackouts 20 s, 26 s | PoseEKF | 0.143 / 0.49 | 0.317 | 2.99 | 0.611 | 33.9 |
| blackouts 20 s, 26 s | latest SLAM pose | 0.216 / 1.52 | 0.911 | 2.62 | – | – |

- Through a 0.3-0.5 s SLAM gap the ImuEKF stays within ~0.1 m (PoseEKF 0.3-0.8 m, the held
  pose 0.9-1.4 m).
- After a blackout ORB-SLAM3's first poses can be 1-2 m off for up to ~1.7 s; the gate
  rejects them (without it the ImuEKF followed them: max error 3.7 m).
- The car's PoseEKF gets yaw rate from differencing poses: 33-41 deg/s RMS error, useless
  for a controller that uses it (LQR / MPC). The gyro gives 0.7 deg/s.
- Sideslip at the rear axle is estimated to 0.013 m/s (truth ~0 here: the sim car barely
  slips; a sliding car is not tested).

Not yet: the car side (firmware per-sample timestamps and a wider DLPF, a raw rear-IMU
buffer with gyro, StateWorker on ImuEKF, extrinsics), and a real recording with both IMUs.

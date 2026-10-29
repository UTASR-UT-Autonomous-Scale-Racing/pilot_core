# ORB-SLAM3 RGB-D-inertial reset loop: cause and fork fix (2026-10-29)

vast.ai L40S / EPYC 7713. ORB-SLAM3 = the UTASR CUDA fork, upstream policy
(`run_orbslam3.py --upstream-policy`) vs branch `inertial-depth-recovery` (e8bacc9).
Values are per run (3 repeats), `a/b/c`.

## Cause

Upstream ORB-SLAM3, in every inertial mode:

1. resets the active map on any tracking loss before the IMU is initialised and before
   inertial BA 2 (~15 s after it);
2. until the IMU is initialised, inserts keyframes only every 0.25 s, and never for weak
   visual tracking (also not after initialisation: the IMU is meant to carry it).

At 4-8 m/s the camera moves 1-2 m between keyframes; inliers fall from ~300 to under 50
within 4-8 frames, `TrackLocalMap` fails, (1) wipes the map, and the IMU never gets the
~1 s / 10 keyframes it needs to initialise: 84-154 resets per 32 s, IMU never initialised,
each new map tracked for ~0.1 s.

## Fix (fork, `IMU_RGBD` and `IMU_STEREO` only; `IMU.UpstreamPolicy: 1` restores upstream)

Depth gives the visual map its scale without the IMU, so a loss is handled as in the
RGB-D mode: no reset, relocalise while the IMU is uninitialised, a new map after 5 s
lost (reset only below 10 keyframes); weak visual tracking asks for a keyframe; no
keyframes from a pose-less frame while lost. (Relocalising after IMU initialisation as
well was tried and dropped: it segfaulted in 3 of 6 runs.)

## Sequences

- `render_clean`: the 3 race laps of the sim run re-rendered by `render_track.py` with
  exact depth and a non-repeating texture, same ground truth and synthetic IMU.
- `render_blackout`: the same with 0.3 s of featureless frames at 3 s and 9 s.
- `compete_textured_race`: the sim sequence itself (LiDAR depth). Only ~35-65 % of ORB
  matches between consecutive frames fit its depth: ORB-SLAM3 loses tracking in every
  mode (RGB-D 35 % tracked), so it says nothing about the inertial policy.

## Results

| sequence | run | tracked % | maps | resets | IMU init (s) | drift 1 s (%) |
|---|---|---|---|---|---|---|
| render_clean | upstream, IMU noise x1 | 73/73 (1 crash) | 1 | 84/84 | never | 102 |
| render_clean | fix, x1 | 31/24/28 | 3-5 | 0 | 1.2/1.2/7.3 | 19/39/36 |
| render_clean | fix, x10 | 53/82/83 | 2-3 | 0 | 1.2 | 2/1/1 |
| render_clean | RGB-D only | 100 | 1 | 0 | - | 1 |
| render_blackout | upstream, x1 | 71/70/71 | 1 | 84-85 | never | 102 |
| render_blackout | fix, x10 | 67/67/67 | 2 | 0 | 1.2/1.2/7.0 | 1/1/2 |
| render_blackout | RGB-D only | 81 | 3 | 0 | - | 1 |
| compete_textured_race | upstream, x1 | 49/49 (1 crash) | 1 | 152/154 | never | 101 |
| compete_textured_race | fix, x1 | 4/4/4 | 1 | 0 | never | 101 |

Upstream's "tracked" counts frames of maps that live ~0.1 s each: its trajectory is
fragments (drift ~100 %). "x10" multiplies the IMU noise densities and random walks given
to ORB-SLAM3 (`--imu-noise-scale 10`); `gen_orbslam_settings.py --imu` already writes
values 35-60x the synthetic IMU's.

- The reset loop is gone: 0 resets, the IMU initialises at ~1.2 s, maps persist and,
  while tracked, drift is 1-2 % of distance per second instead of ~100 %.
- Still not as robust as RGB-D only: 53-83 % tracked against 100 %. After IMU
  initialisation the tracker leans on the IMU prediction; its speed estimate drifts low
  (7-28 % within 2 s at x1) and tracking collapses. Inflating the IMU noise helps (x10)
  but runs vary (48-83 % on the same sequence). Inertial BA 2 was reached in 2 of 9 fix
  runs.
- ORB-SLAM3 crashed (segfault) in ~1 of 9 runs on this box in every mode, upstream policy
  and RGB-D only included: the fork's build here, not the policy.

Serial latency on `render_clean`, tracking call, ms (mean / p50 / p95 / p99 / max):

| run | tracked % | ATE RMSE (m) | drift 1 s (%) | latency |
|---|---|---|---|---|
| ORB-SLAM3 fix, IMU_RGBD x10 | 48 | 6.52 (3 maps) | 5.0 | 11.0 / 7.7 / 42.3 / 60.9 / 65.1 |
| ORB-SLAM3 RGB-D | 100 | 0.107 | 0.7 | 12.3 / 8.4 / 41.4 / 48.9 / 58.6 |
| cuVSLAM RGB-D + IMU + SLAM | 100 | 0.900 | 0.6 | 38.0 / 2.8 / 285.1 / 700.5 / 837.9 |
| cuVSLAM RGB-D + IMU, odometry only | 100 | 0.245 | 0.4 | 3.0 / 2.7 / 4.4 / 5.4 / 149.3 |

On this richly textured sequence cuVSLAM's SLAM back end runs inside `track()`: 110 calls
over 100 ms. Its odometry alone stays at 3-4 ms.

## Verdict

Use ORB-SLAM3 in RGB-D mode on the car for now. The inertial mode no longer destroys its
map, but at race speeds it is less robust than RGB-D alone; check it on a real recording
(real IMU, real depth) with `--imu-noise-scale` 1 and 10 before enabling `slam.inertial`.

## Update: real-time replay, CPU budget, startup (fork 7cb97e4)

Two benchmark-harness faults distorted everything above:

- The runners fed frames as fast as tracking allowed (~3x real time for ORB-SLAM3): its
  local-mapping thread (keyframes, IMU initialisation, inertial BA) fell behind. Now the
  sequence's clock (`--speed 1`).
- The container reports 255 cores but is capped at 30.6 CPUs: ORB-SLAM3's ~500 threads got
  the process throttled, which was its 40-60 ms p95. Now pinned (`cpulimit.py`, `--cpus 8`).

And a fork fix for the start: before the IMU is initialised the inertial modes used
monocular-inertial frame tracking (10 px search window, poses accepted from any number of
matches); through a 4-5 m/s swerve 0.8 s in it lost tracking in 2-7 of 9 runs. With depth,
the RGB-D rules now apply until the IMU is initialised: 18 of 18 runs tracked the first
10 s with no loss, IMU initialised at ~1.1 s. A young lost map (before inertial BA 2) is
given up after 0.5 s instead of 5 s (`IMU.EarlyLostTimeout`).

Same mode for both: RGB-D + IMU + SLAM. Real time, 8 pinned cores, 3 runs sharing the GPU.
ORB-SLAM3 with IMU noise x10 (`--imu-noise-scale 10`; `gen_orbslam_settings.py --imu`
writes values 35-60x the synthetic IMU's for the car).

| sequence | system | tracked % | maps | ATE RMSE (m) | drift 1 s (%) | latency p50 / p95 (ms) |
|---|---|---|---|---|---|---|
| clean | ORB-SLAM3 CUDA (3 runs) | 100/100/100 | 1 | 0.12/0.10/0.17 | 0.7/0.7/0.6 | 7.8 / 9.9 |
| clean | cuVSLAM | 100 | 1 | 0.24 | 1.8 | 7.0 / 1837 |
| blackouts 20 s, 26 s | ORB-SLAM3 CUDA (3 runs) | 98/98/98 | 1 | 0.19/0.14/0.14 | 0.9/0.7/1.1 | 7.8 / 10.0 |
| blackouts 20 s, 26 s | cuVSLAM | 100 | 1 | 1.84 | 4.7 | 4.0 / 7.1 |
| blackouts 3 s, 9 s | ORB-SLAM3 CUDA (3 runs) | 96/96/96 | 3 | 6.45/6.44/6.50 | 1.1 | 7.7 / 10.0 |
| blackouts 3 s, 9 s | cuVSLAM | 100 | 1 | 3.95 | 9.8 | 4.2 / 7.2 |

- ORB-SLAM3's tracked % excludes the blank frames (it gives no pose there; cuVSLAM gives
  IMU-only poses). A blackout before inertial BA 2 (~16 s) still costs it a new map, hence
  the ATE over 3 maps; afterwards it rides through on the IMU.
- With the true IMU noise (x1) it tracks 93-95 % in 3-4 maps: inflate the noise.
- cuVSLAM's SLAM back end stalls its tracking call on this texture (p95 1.8 s on the clean
  laps); its odometry alone: 7.4 / 15.2 ms, ATE 1.58 m.

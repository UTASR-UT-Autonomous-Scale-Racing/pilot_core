# RGB-D-inertial benchmark: ORB-SLAM3 (CUDA fork) vs cuVSLAM

Both systems run offline on the same recorded sequence, in the same `rgbdi` format
([rgbdi.py](rgbdi.py)), and are scored by one script against ground truth.

| step | script | runs in |
| ---- | ------ | ------- |
| sim run to sequence | `scripts/export_recording.py <run>`, then [sim_to_rgbdi.py](sim_to_rgbdi.py) | pilot_core env |
| sim sequence re-rendered with exact depth (same track, trajectory, IMU) | [render_track.py](render_track.py) | pilot_core env |
| simulated rear-of-car IMU (ICM-20948, DLPF, I2C polling) | [make_rear_imu.py](make_rear_imu.py) | pilot_core env |
| vehicle state from the rear IMU + SLAM (control.ImuEKF vs PoseEKF), causal replay | [fuse_rear_imu.py](fuse_rear_imu.py) | pilot_core env |
| car config for recording (frames + 6-axis IMU, nothing dropped) | [make_record_config.py](make_record_config.py) | pilot_core env |
| car recording to sequence | [car_to_rgbdi.py](car_to_rgbdi.py) | pilot_core env |
| ORB-SLAM3, `IMU_RGBD` (or `RGBD` with `--no-imu`) | [run_orbslam3.py](run_orbslam3.py) | pilot_core env, built `orbslam3_python` |
| cuVSLAM, Multisensor RGB-D + IMU + SLAM (or `RGBD` with `--no-imu`) | [run_cuvslam.py](run_cuvslam.py) | its own venv: the PyCuVSLAM 17.x wheel pins numpy |
| accuracy against ground truth (sim), coverage, latency | [score.py](score.py) | pilot_core env (+ matplotlib) |
| accuracy from AprilTags + lap closure (real track) | [markers.py](markers.py) | pilot_core env |

```bash
python scripts/autodrive_bench.py run --config config_autodrive_compete.yaml --out runs/vio --laps 3 \
    --set sim_depth.camera=sim_camera/camera.yaml --set sim_depth.scale=2 --set recording_config.frame.sample_k=1
python scripts/export_recording.py runs/vio
python scripts/vio_bench/sim_to_rgbdi.py runs/vio --out seq/vio
python scripts/vio_bench/run_orbslam3.py seq/vio --out res/orb_imu
ORB_SLAM3_CUDA=0 python scripts/vio_bench/run_orbslam3.py seq/vio --out res/orb_cpu_imu
/venv/cuvslam/bin/python scripts/vio_bench/run_cuvslam.py seq/vio --out res/cuv_imu
python scripts/vio_bench/score.py seq/vio res/* --out res/score
# the same laps rendered with exact depth (and with forced 0.3 s tracking losses)
python scripts/vio_bench/render_track.py seq/vio runs/vio --out seq/vio_render
python scripts/vio_bench/render_track.py seq/vio runs/vio --out seq/vio_blackout --blackout 3.0:0.3 --blackout 9.0:0.3
```

`run_orbslam3.py --upstream-policy` runs upstream ORB-SLAM3's inertial policy (reset the
map on a tracking loss before the IMU is initialised, keyframes by time only) instead of
the fork's depth-inertial one (relocalise, keyframes also when visual tracking weakens).

Metrics ([score.py](score.py)): ATE after one rigid (no scale) alignment, RPE over 1 s
(translation in m and % of distance, rotation in deg), fraction of frames tracked, time to
the first pose and to IMU initialisation (ORB-SLAM3), and the per-frame tracking call's
latency (mean, p50, p95, p99, max). Latency is the tracking call only: both systems map
and close loops in background threads.

## Real track (the verdict)

No motion capture: accuracy comes from AprilTags placed around the track
([markers.py](markers.py)) and from laps that pass them repeatedly.

**Before the session**

1. Print 8-12 tag36h11 tags (ids 0..N, any free generator), black square **24 cm or
   larger** (A3): a 16 cm tag is only good to ~1.5 cm within 1.5 m of the camera, and
   sightings beyond 2.5 m or under 30 px are dropped. Note the printed square edge.
2. Mount them upright on the walls facing the racing line, at camera height (~15-30 cm),
   spread round the whole lap, some on straights and some in corners, flat and not
   reflective. Never move them during the session.
3. Optional, for absolute accuracy: tape-measure each tag centre in one track frame
   (origin on the start line, x along it, y left, z up) into `markers.yaml` (format in
   [markers.py](markers.py)); at least 3 tags, not in a line. Without it you still get
   lap-to-lap repeatability and closure.
4. Settings with the IMU block at the recording rate, and the recording config:

```bash
python scripts/gen_orbslam_settings.py --imu                 # camera level and still: checks T_b_c1
python scripts/vio_bench/make_record_config.py config.yaml -o config_vio_record.yaml   # IMU 200 Hz
```

(`make_record_config.py --imu-rate` must match the rate the settings were generated
for: regenerate with `camera.imu_rate_hz` set, or pass `--imu-rate 100`.)

**Recording**

5. Car still for ~3 s, then 5+ laps at the speeds you race at (a slow lap first if you
   want both), then stop on the start mark (for `--closed`). Over USB 3.

```bash
python start.py --config config_vio_record.yaml              # [q] when done
```

**Afterwards** (on the vast.ai box or the Orin; latency on the Orin is the one that counts)

```bash
python scripts/vio_bench/car_to_rgbdi.py /tmp/pilot_recordings/<session> --settings orbslam_oakd_rgbd.yaml --out seq/track1
python scripts/vio_bench/run_orbslam3.py seq/track1 --out res/orb_imu
python scripts/vio_bench/run_orbslam3.py seq/track1 --out res/orb_rgbd --no-imu
/venv/cuvslam/bin/python scripts/vio_bench/run_cuvslam.py seq/track1 --out res/cuv_imu
/venv/cuvslam/bin/python scripts/vio_bench/run_cuvslam.py seq/track1 --out res/cuv_rgbd --no-imu
python scripts/vio_bench/score.py seq/track1 res/* --out res/score            # latency, coverage
python scripts/vio_bench/markers.py seq/track1 res/* --survey markers.yaml --closed --out res/markers
```

`score.py` has no ground truth here, so it reports coverage and latency only; the
accuracy is in `markers.md`: repeatability (the same tag placed on different laps), its
growth lap by lap, absolute error against the survey, and start/end closure.

## Sim sequences: what they can and can't show

- The stock AutoDRIVE build has an untextured floor with camera-fixed glare and black
  walls; use the textured copy (`scripts/autodrive_texture_sim.py --variant compete`).
- Depth comes from the 2-D LiDAR (walls extruded, floor plane) through a fitted camera
  model. Against ground truth, cuVSLAM's speed comes out ~15 % low with zero time lag and
  0.98 turn-rate correlation: the depth's scale is off, so absolute accuracy in the sim is
  pessimistic for both systems.
- The IMU is synthesised from the ground-truth pose (low-passed at 6 Hz, then
  differentiated, plus consumer-MEMS noise): no vibration, no real bias behaviour.
- The car-body region of the image is painted flat: it moves with the camera.
- The LiDAR depth and the sim's images do not agree closely: between consecutive frames
  only ~35-65 % of ORB matches fit the depth (more for the most distinctive matches), so
  a tracker that builds its map from depth (ORB-SLAM3) loses it within a few frames in
  every mode, while cuVSLAM, which tracks 2-D features and uses depth for scale, holds
  on. [render_track.py](render_track.py) renders the same track and trajectory with a
  non-repeating texture and exact depth, keeping the sequence's IMU and ground truth:
  that one compares the trackers, not the sim's depth.

The sim exercises the pipeline and gives comparable latency; the real-track recordings
(markers + lap closure) are the accuracy verdict.

## Building ORB-SLAM3 on the vast.ai box

On the AMD EPYC 7713 / Ubuntu 24.04 / gcc 13 box, ORB-SLAM3 built with its default
`-march=native` (ORB_SLAM3, g2o, DBoW2) aborts with "double free or corruption" in
`g2o::HyperGraph::clear` after the first pose optimisation, the C++ `rgbd_tum` example
included. Without `-march=native` in the three CMakeLists it runs. The cause is not
pinned down.

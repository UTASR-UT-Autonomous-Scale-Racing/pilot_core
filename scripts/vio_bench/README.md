# RGB-D-inertial benchmark: ORB-SLAM3 (CUDA fork) vs cuVSLAM

Both systems run offline on the same recorded sequence, in the same `rgbdi` format
([rgbdi.py](rgbdi.py)), and are scored by one script against ground truth.

| step | script | runs in |
| ---- | ------ | ------- |
| sim run to sequence | `scripts/export_recording.py <run>`, then [sim_to_rgbdi.py](sim_to_rgbdi.py) | pilot_core env |
| ORB-SLAM3, `IMU_RGBD` (or `RGBD` with `--no-imu`) | [run_orbslam3.py](run_orbslam3.py) | pilot_core env, built `orbslam3_python` |
| cuVSLAM, Multisensor RGB-D + IMU + SLAM (or `RGBD` with `--no-imu`) | [run_cuvslam.py](run_cuvslam.py) | its own venv: the PyCuVSLAM 17.x wheel pins numpy |
| accuracy, coverage, latency | [score.py](score.py) | pilot_core env (+ matplotlib) |

```bash
python scripts/autodrive_bench.py run --config config_autodrive_compete.yaml --out runs/vio --laps 3 \
    --set sim_depth.camera=sim_camera/camera.yaml --set sim_depth.scale=2 --set recording_config.frame.sample_k=1
python scripts/export_recording.py runs/vio
python scripts/vio_bench/sim_to_rgbdi.py runs/vio --out seq/vio
python scripts/vio_bench/run_orbslam3.py seq/vio --out res/orb_imu
ORB_SLAM3_CUDA=0 python scripts/vio_bench/run_orbslam3.py seq/vio --out res/orb_cpu_imu
/venv/cuvslam/bin/python scripts/vio_bench/run_cuvslam.py seq/vio --out res/cuv_imu
python scripts/vio_bench/score.py seq/vio res/* --out res/score
```

Metrics ([score.py](score.py)): ATE after one rigid (no scale) alignment, RPE over 1 s
(translation in m and % of distance, rotation in deg), fraction of frames tracked, time to
the first pose and to IMU initialisation (ORB-SLAM3), and the per-frame tracking call's
latency (mean, p50, p95, p99, max). Latency is the tracking call only: both systems map
and close loops in background threads.

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

The sim exercises the pipeline and gives comparable latency; the real-track recordings
(markers + lap closure) are the accuracy verdict.

## Building ORB-SLAM3 on the vast.ai box

On the AMD EPYC 7713 / Ubuntu 24.04 / gcc 13 box, ORB-SLAM3 built with its default
`-march=native` (ORB_SLAM3, g2o, DBoW2) aborts with "double free or corruption" in
`g2o::HyperGraph::clear` after the first pose optimisation, the C++ `rgbd_tum` example
included. Without `-march=native` in the three CMakeLists it runs. The cause is not
pinned down.

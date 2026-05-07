# AutoDRIVE simulator

pilot_core can drive the Unity-based AutoDRIVE simulator through an
`autodrive_bridge` over ZMQ. `ZMQSensorWorker` replaces the camera and IMU workers,
and `ActuatorWorker` sends the commands back.

```bash
python start.py --config config_zmq.yaml
```

| Port | Direction | Content |
| ---- | --------- | ------- |
| 5555 | simulator → pilot_core | camera (JPEG), depth, ground-truth pose, IMU, LiDAR, speed, lap/collision counters |
| 5556 | pilot_core → simulator | `{"steering", "throttle", "reset"}` in [−1, 1], steering +1 = left |
| 5557 | pilot_core → simulator | telemetry for visualisation (boundaries, pose, point cloud, segmentation as a binary mask) |

Useful settings in `config_zmq.yaml`:

- `use_zmq_pose: true`: use the simulator's ground-truth pose instead of SLAM. A LiDAR
  projection builds the wall map; press `l` to signal loop closure.
- `use_saved_map: true`: load boundaries from `saved_map_path` and drive straight away.
- `mapping.bev_axes: [0, 1]`, `mapping.forward_axis: 0`: the simulator's frame is Z-up,
  X-forward (the SLAM camera frame uses `[0, 2]` and `2`).
- `control_mode`: any track mode works with the simulator's pose.
- `depth_raw_scale`: metres per unit, needed only if the bridge sends raw `uint16` depth.
- SLAM, `depth_ftg` and `vision.model: ground_plane` need the simulated camera's
  intrinsics in an ORB-SLAM3 settings file (`slam.settings`). None ships with the repo;
  without one they would use the OAK-D intrinsics, which are wrong for the simulator.

## RL training

`rl/` contains a Gymnasium environment over the simulator and a PPO training script.
It needs `pip install -e ".[rl]"`.

```bash
python rl/train.py                 # new run → rl/logs/PPO_N, rl/checkpoints/PPO_N
python rl/train.py --vis-port 5558 # also stream observations for rl/vis_obs_ros2.py
tensorboard --logdir rl/logs
python rl/export_onnx.py --run PPO_N --out policy.onnx   # for control_mode: policy
```

Training runs the same safety supervisor as the car (speed cap, taper, slew limits),
with the settings built from `--pilot-config` (default `config_zmq.yaml`) the same way
`start.py` does. The exported `policy.json` sidecar records the observation layout, the
policy rate (`policy_hz`, which `ControlWorker` steps the policy at) and the safety
settings used in training. `ControlWorker` warns if they differ from the car's.

`rl/publish_boundaries_ros2.py` and `rl/vis_obs_ros2.py` visualise boundaries and
observations in RViz (run them from a repo checkout; they import `control`).
`scripts/collect_frames.py` saves simulator camera frames into a timestamped folder per
session.

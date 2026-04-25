# AutoDRIVE simulator

pilot_core can drive the Unity-based AutoDRIVE simulator through an
`autodrive_bridge` over ZMQ. `ZMQSensorWorker` replaces the camera and IMU workers,
and `ActuatorWorker` sends the commands back.

```bash
python start.py --config config_zmq.yaml
```

| Port | Direction | Content |
|- |------ |---- |
| 5555 | simulator → pilot_core | camera (JPEG), depth, ground-truth pose, IMU, LiDAR, speed, lap/collision counters |
| 5556 | pilot_core → simulator | `{"steering", "throttle", "reset"}` in [−1, 1], steering +1 = left |
| 5557 | pilot_core → simulator | telemetry for visualisation (boundaries, pose, segmentation) |

Useful settings in `config_zmq.yaml`:

- `use_zmq_pose: true`: use the simulator's ground-truth pose instead of SLAM. A LiDAR
  projection builds the wall map; press `l` to signal loop closure.
- `use_saved_map: true`: load boundaries from `saved_map_path` and drive straight away.
- `mapping.bev_axes: [0, 1]`, `mapping.forward_axis: 0`: the simulator's frame is Z-up,
  X-forward (the SLAM camera frame uses `[0, 2]` and `2`).
- `control_mode`: any track mode works with the simulator's pose; `lidar_pid` uses its LiDAR.

## RL training

`rl/` contains a Gymnasium environment over the simulator and a PPO training script.
It needs `pip install gymnasium stable-baselines3 torch`.

```bash
python rl/train.py                 # new run → rl/logs/PPO_N, rl/checkpoints/PPO_N
tensorboard --logdir rl/logs
python rl/export_onnx.py --run PPO_N --out policy.onnx   # for control_mode: policy
```

`rl/publish_boundaries_ros2.py` and `rl/vis_obs_ros2.py` visualise boundaries and
observations in RViz. `scripts/collect_frames.py` saves simulator camera frames for
building datasets.

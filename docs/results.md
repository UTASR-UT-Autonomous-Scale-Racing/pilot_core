# Results: AutoDRIVE RoboRacer sim (October 2026)

pilot_core was validated end to end against the stock AutoDRIVE RoboRacer `2026-iros`
build over the direct link ([autodrive.md](autodrive.md)), then benchmarked. Sim at 640×360,
Vulkan, Low quality, on a shared-GPU cloud box. Full tables: [bench/2026-10-20/results.md](../bench/2026-10-20/results.md).

## Validation

| check | result |
| ----- | ------ |
| Link: rates, pose, LiDAR, camera | PASS. Pose error 2.5e-7 m; 100 % of LiDAR points within 10 cm of the mapped walls |
| Disarmed hold / armed command echo | PASS / PASS |
| Steering sign / negative throttle | PASS / PASS |
| Reset | PASS, 0.0001 m from the start |
| Stop: SIGTERM or SIGKILL `start.py` | PASS / PASS. Neutral sent; the car rolled 0.34 / 0.30 m |
| Stop: SIGKILL every pilot_core process | **FAIL**. The sim has no command timeout: 8.51 m driven, 15 wall hits in 21.2 s. Needs a sim-side timeout like the firmware's 250 ms |
| Calibration | 24.80 m/s per unit throttle; effective wheelbase 0.342 m |
| Map, then race 3 laps | PASS: 0 collisions; cross-track RMS 4.1 cm |
| pytest | 357 passed, 7 skipped (all extras installed) |

## Benchmark (84 runs, 0 errors; 3 repeats × 2 laps per cell)

![Benchmark summary](../bench/2026-10-20/summary.png)

- **Controller × v_max.** At v_max 4.0, pure_pursuit and mpc are fastest (mean lap
  14.2–14.4 s) with 0 collisions. mpc has the lowest cross-track error from 1.5 m/s up.
  Collisions happen only at v_max ≥ 2.5, with lqr and stanley; stanley at 4.0 is the worst
  cell (1.3 ± 2.3 per run). The car never reaches 4.0 m/s: its mean speed is about 2.3 m/s.
- **Loop rate, 30 / 50 / 100 Hz.** Barely matters. pure_pursuit best lap 12.89–12.90 s;
  mpc is about 0.25 s slower per lap with 3.8 cm cross-track.
- **Pose and map source.** Saved map: best lap 12.90 s. LiDAR mapping on the fly, then
  racing: 12.63 ± 0.25 s. This gap is unexplained; the two maps differ in length by only 0.2 m.
  ORB-SLAM3: **FAILED** (below).
- **Track.** Practice best lap 12.89 s; compete best lap 19.53 s; 0 collisions on both.
- **Sim stalls.** 0–89.8 % of a run's wall time in controller × v_max, when other GPU jobs
  shared the box; 0 % in every other stage. Each run reports `stalled_pct`.

## Learned components

- **Track segmentation** (Fast-SCNN, ONNX, trained on sim runs): IoU 0.941 on the
  compete track it never saw, 0.939 live in `VisionWorker`, with lap times unchanged.
- **PPO policy** (bicycle-model pretrain, PPO_1): zero-shot on the live sim, lap 1 in 17.58 s
  against pure_pursuit's 18.09 s at the same 2.0 m/s cap. About 30 % of live episodes
  collide, most likely because the bicycle env's collision body (circles r = 0.14 m) is
  smaller than the car. A live fine-tune of 26.5k steps did not improve this.

## What did not work

- **ORB-SLAM3 on the stock sim camera.** 30–63 % of frames tracked, ATE 2.5–2.7 m over an
  8.5 m lap, Sim(3) scale 42–67 (metric RGB-D should give 1). The 192×108 camera sits near the
  rear axle, so the car body fills about 55 % of the frame; once it is masked, the features
  left are the horizon and floor glare. It needs a camera ahead of the body with real
  depth, which Joshua Choi's build provides.
- **CPU rendering of the sim.** The build ships only HDRP Vulkan shaders, so lavapipe and
  GL are impossible, and the sim shares the NVIDIA driver with other jobs.

Raw runs, GIFs and logs are archived with the maintainers ([datasets.md](datasets.md)).

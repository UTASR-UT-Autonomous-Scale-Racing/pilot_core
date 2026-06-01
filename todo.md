# TODO: pilot_core ↔ AutoDRIVE (where the work stopped, 2026-10-20)

Branch `autodrive-direct`, 21 commits on top of `main` (`38f0032`). Nothing has been
pushed: the vast box's gh token has no UTASR access. The plan is in
`/workspaces/HANDOFF_PILOT_CORE.md`, and the sim notes are in `docs/autodrive.md`.

## Done

- **Step 0.** Sim setup (`scripts/vast_autodrive_setup.sh`, supervisor services, noVNC
  on 127.0.0.1:6080) and the protocol spike (`scripts/autodrive_spike.py`).
  - Finding: the sim's client is EIO=4 websocket-only, pings from the client side and
    never sends `40`. python-socketio 4 and 5 both fail, so pilot_core speaks the
    subset directly over `websockets`.
- **Step 1.**
  - `AutoDRIVEWorker`, the actuator output buffer, and `start.py` wiring (key `r` =
    reset).
  - `config_autodrive.yaml` and `vehicle_params_roboracer.yaml`.
  - The fake sim, tests, and the `TrackEnv` port.
  - LiDAR gap-follower mapping lap and auto loop closure.
  - Full pytest: 326 passed, 16 skipped (`/workspaces/proof/logs/pytest_after_zmq_removal.log`).
- **Step 2.** Live checks (evidence in `/workspaces/proof/runs`, GIFs in `/workspaces/proof/gifs`):
  - **2a PASS** (`runs/2a/link_results.json`)
  - **2b PASS**, **2c PASS**, **2d reset PASS** (`runs/2bcd/results.json`)
  - **2d stops** (`runs/2d_kill/kill_results.json`): SIGTERM and SIGKILL of `start.py`
    PASS. SIGKILL of every pilot_core process **FAIL**: the stock sim has no command
    timeout and keeps driving the last command.
  - **2e** (`runs/2e_calibration`): 24.8 m/s per unit throttle; effective wheelbase 0.342 m.
  - **2f PASS**, before and after the ZMQ removal (`runs/2af`, `runs/2f_recheck`):
    3 laps, 0 collisions, 4.1 cm cross-track RMS. The map is in `maps/autodrive_practice`.
  - The ZMQ bridge path is removed (`b8754ef`).
- **Step 3 Stage A.** All 48 runs done (16 cells × 3) in
  `/workspaces/proof/bench_runs/2026-10-20/A`; the summary has not been run.

## Left to do

1. **Stage A summary.** Run
   `python scripts/autodrive_bench.py summary /workspaces/proof/bench_runs/2026-10-20`.
   This writes `results.csv`, `summary.png` and `results.md`. Copy them to
   `/workspaces/proof/bench/`, and possibly to `bench/2026-10-20/` in the repo.
2. **Stages B and D.** Run
   `python scripts/autodrive_bench.py matrix bench/matrix.yaml --stage B --out /workspaces/proof/bench_runs/2026-10-20`,
   then `--stage D`.
   - D switches the sim to the compete build and maps that track first.
   - The Joshua cells are SKIPPED.
   - Best and worst benchmark cell GIFs: `autodrive_bench.py gif <rep dir> --prefix ...`.
3. **Stage C / ORB-SLAM3. Not working yet.**
   - The stock build has no depth and only a 192×108 front camera, which sits near the
     rear axle and sees the chassis in the bottom ~40 % of the image. Verified on the
     raw wire JPEG (`tests/data/autodrive_bridge_sample.json`).
   - The adapter is committed (`37160f0`):
     - `workers/lidar_depth.py`: LiDAR depth, chassis mask, CLAHE.
     - `sim_camera/`: fitted camera (vfov 35°, pitch 10.5°, height 0.17 m).
     - `AutoDRIVEWorker` uses it when `sim_depth` is set.
   - ORB-SLAM3 is built with CUDA (sm_120) and imports.
   - Offline replay (`scripts/autodrive_slam_replay.py`) with the car fully hidden:
     30 % of frames tracked, ATE 2.7 m over an 8.7 m path. The scene is too dark and
     texture-poor; the earlier 78 % "tracking" was locked onto the car body.
   - **Next:** test the HUD "Scene Light" toggle and Low/High Quality for brighter, more
     textured camera frames, and record frames at `sample_k: 1`. If that's still not
     enough, say so and mark the ORB-SLAM3 cells FAILED with the replay evidence. Then:
     - matrix `pose_map` entries for `orbslam3_*` (`sim_pose: false`, `sim_depth`,
       `resolution: [384, 216]`, `slam.settings: sim_camera/orbslam_settings.yaml`,
       `mapping.bev_axes: [0, 2]`)
     - CUDA vs CPU (`ORB_SLAM3_CUDA=0`)
     - remove the orbslam3 entries from `skip:`
   - Note to the user: they say the camera is front-mounted. That's true of the real car
     or Joshua's build, not this stock build.
4. **Proof write-up.**
   - `/workspaces/proof/validation.md`: per check PASS/FAIL, command, key numbers,
     evidence file. All numbers are in the JSON files listed above.
   - `/workspaces/proof/logs`: copy the sim, pilot_core and pytest logs it relies on.
   - `/workspaces/proof/bench/`
   - Re-read `validation.md` against the evidence.
5. **`/workspaces/REPORT_PILOT_CORE.md`.** Cover:
   - commits
   - pytest
   - validation table
   - benchmark headlines
   - SKIPPED and FAILED items: Joshua build cells, SIGKILL-all, ORB-SLAM3
   - deviations:
     - websockets instead of python-socketio
     - LiDAR mapping driver and auto loop closure
     - sim under `/workspaces`
     - lavapipe/GL impossible
   - reproduce commands
6. **Before the final pytest run**, add websockets/gymnasium notes. CI needs
   `websockets` (it's in the `test` extra).

## Environment caveats

- **Sim freezes.** The sim freezes for seconds at a time when other GPU jobs in this
  container load the NVIDIA driver (graphics thread in `D` state in
  `os_acquire_rwlock_read`): 0–90 % of a run's wall time. Every run reports `stalled_pct`.
- **ORB-SLAM3 import.** Importing `orbslam3_python` needs `import ssl` first (libcrypto
  clash); `SLAMWorker` does this.
- **Sim settings.** These live in `/workspaces/autodrive_sim/sim.env` (practice,
  640×360, Vulkan).

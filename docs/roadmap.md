# TODO: pilot_core ↔ AutoDRIVE (status 2026-10-21)

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
    3 laps, 0 collisions each; cross-track RMS 4.4 cm before, 4.1 cm after. The map is in `maps/autodrive_practice`.
  - The ZMQ bridge path is removed (`b8754ef`).
- **Step 3 Stage A.** All 48 runs done (16 cells × 3) in
  `/workspaces/proof/bench_runs/2026-10-20/A`; the summary has not been run.

## Done 2026-10-21 (on a new vast box, `/workspaces` -> `/workspace` symlink)

- Stages A-D all run (84 runs OK). Results: `bench/2026-10-20/`, `/workspaces/proof/bench/`.
- Compete-track mapping fixed (`gap_select: area`, `8e398b7`).
- ORB-SLAM3 cells marked FAILED with the HUD-quality replays (`dbc1571`).
- `/workspaces/proof/validation.md` and `/workspaces/REPORT_PILOT_CORE.md` written.
- Final pytest: 357 passed, 7 skipped.

## Done 2026-10-21, later (vision, policy, telemetry export)

- **Vision.** VisionWorker runs in the sim with sim_depth (`d0a249e`).
  - Fixed: with sim_depth on, every frame was too big for the buffer, and the pose was held back with it.
  - Fast-SCNN trained on sim runs: IoU 0.94 live on the unseen compete track (`d6e88ea`).
- **Policy.**
  - Bicycle-model pretraining, PPO_1 (`076feec`).
  - Zero-shot on the live sim: lap 1 in 17.58 s, against pure_pursuit's 18.09 s at the same 2.0 m/s cap.
  - Live fine-tune: 26.5k steps, 121 episodes, about 30 % of episodes collided from the start to the end.
    No improvement; stopped. Checkpoints at 4.51M and 4.52M; PPO_1 at 4.5M is untouched.
  - TrackEnv reset fix (`9d1070d`).
  - The policy model key is `track_control.policy.model`.
- **Telemetry export.** `scripts/export_recording.py` exports all buffers on one clock plus timeline CSVs.
  The showcase run is in `/workspaces/proof/showcase`.

## Left to do

0. Policy: the ~30 % collision rate live is most likely the bicycle env's small collision body (circles r=0.14 m).
   Retrain in rl/bicycle_env.py with the real footprint and a wall margin, then benchmark
   (3 runs x 2 laps, practice + compete) against pure_pursuit at the same cap.
1. Push `autodrive-direct` once a gh token with UTASR access is available.
2. Optional: re-run Stage A without other GPU jobs, since it had 0-89.8 % sim stalls.
3. Optional: find out why the lidar_mapping laps (best 12.34-12.78 s) beat the saved map
   (12.89 s).
4. ORB-SLAM3 needs Joshua's build: a camera ahead of the car body, with depth.

## Environment caveats

- **Sim freezes.** The sim freezes for seconds at a time when other GPU jobs in this
  container load the NVIDIA driver (graphics thread in `D` state in
  `os_acquire_rwlock_read`): 0–90 % of a run's wall time. Every run reports `stalled_pct`.
- **ORB-SLAM3 import.** Importing `orbslam3_python` needs `import ssl` first (libcrypto
  clash); `SLAMWorker` does this.
- **Sim settings.** These live in `/workspaces/autodrive_sim/sim.env` (practice,
  640×360, Vulkan).

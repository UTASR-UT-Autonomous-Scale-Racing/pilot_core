# AutoDRIVE re-run and steering-cap sweep (2026-10-29)

`scripts/autodrive_bench.py run --config config_autodrive_compete.yaml --laps 10`, stock 2026-iros
builds, L40S. Scored like the published final: race time + 10 s per collision.

| run | laps | collisions | laps (s) | adjusted (s) | place of 23 |
|---|---|---|---|---|---|
| qualifier, practice track | 10 | 0 | 9.19, then 6.89-6.98 | ~71.6 | qualified |
| final, compete, steer_lat_accel 5.0 (config) | 10 | 0 | 13.39, then 10.52-10.76 | **109.07** | **15** |
| compete, steer_lat_accel 5.5 | 10 | 6 | 13.98, 10.43-11.62 | 172.85 | 22 |
| compete, steer_lat_accel 6.0 | 8 (stopped at 10+ hits) | 11 | 10.93-12.05 | - | - |
| compete, steer_lat_accel 7.0 | 5 (stopped) | 11 | 10.78-11.75 | - | - |
| compete, steer_lat_accel 8.0 | 5 (stopped) | 11 | 10.60-12.20 | - | - |

Published final: 1st 71.82 s (best lap 7.16 s), 14th 107.77 s, 15th 113.19 s.

- A higher cap hits the wall at the exit of the bottom hairpin, every time at the same spot
  (x 2.4, y -15.3: clipping the end of the inner wall at ~2 m/s), and at 6.0 also out of the
  top hairpin (fig_steer_collisions.png). The zig-zags never touched.
- Even clean it would gain little: the best lap at 5.5 was 10.43 s against 10.52 s.
- Where the time goes, flying lap at 5.0: mean 4.1 m/s (the winners ~6.1 m/s); hairpins mean
  2.3 m/s (min 1.6), zig-zag 3.8 m/s, the long straight peaks at 8.8 m/s (v_max 12, accel and
  decel 5 m/s^2 never let it get there).

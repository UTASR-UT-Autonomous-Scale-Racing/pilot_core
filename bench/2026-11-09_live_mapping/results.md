# In-loop mapping and the degraded pose (2026-11-09)

Stock AutoDRIVE RoboRacer `2026-iros` builds, L40S, 640×360, `config_autodrive_compete.yaml`,
`scripts/cluster/autodrive_job.sh` stages `live` and `degraded`. No LiDAR, no sim speed. Scored
like the published final: race time + 10 s per collision. Each live run is one job: a mapping lap
with follow-the-gap on the segmentation-edge scan, the walls from the occupancy grid
(`mapping.wall_source: occupancy`), the racing line from that map (margin 0.65 m), then 10 laps on it.

## Mapping lap in the loop

| track | run | map built after (s) | hits during the mapping lap | racing line's closest wall (m) |
| --- | --- | --- | --- | --- |
| practice | 1 / 2 / 3 | 54.9 / 54.9 / 55.5 | 0 / 0 / 0 | 0.50 / 0.50 / 0.48 |
| compete | 1 / 2 / 3 | 89.0 / 92.8 / 87.8 | 5 / 4 / 5 | 0.42 / 0.33 / 0.37 |

Walls against the LiDAR-built map (used only to score them), median / p90 / max, m:

| map | outer | inner |
| --- | --- | --- |
| practice live 1 / 2 / 3 | 0.17 / 0.36–0.39 / 0.82–1.07 | 0.08 / 0.19–0.24 / 0.71–0.72 |
| compete live 1 / 2 / 3 | 0.10–0.13 / 0.39–0.40 / 0.56–0.60 | 0.15–0.18 / 0.31–0.33 / 0.65–0.74 |
| compete, offline from five recordings (2026-10-31) | 0.13 / 0.30 / 0.51 | 0.10 / 0.28 / 0.42 |

After the compete mapping lap, runs 1 and 3 drove their one centre-line lap into a respawn trap
at a hairpin (the car respawned too close to the line's corner and hit again every few tenths of
a second, 900+ hits until the timeout). That lap had no racing line yet; the races below use one.

## Racing on the live maps (10 laps)

| track | run | collisions | first lap, then (s) | raw (s) | adjusted (s) |
| --- | --- | --- | --- | --- | --- |
| practice | 1 | 0 | 9.55, 7.44–7.52 | 76.89 | 76.89 |
| practice | 2 | 0 | 9.50, 7.37–7.48 | 76.20 | 76.20 |
| practice | 3 | 0 | 9.63, 7.44–7.51 | 76.94 | 76.94 |
| compete | 1 | 4 | 13.40, 11.84–12.92 | 125.60 | 165.60 |
| compete | 2 | 1 | 13.52, 11.76–12.84 | 122.20 | 132.20 |
| compete | 3 | 7 | 13.16, 11.86–12.69 | 124.06 | 194.06 |

On the practice track the live map races like the offline one (qualifier 75.21 s, 0 collisions).
On the compete track it does not yet: the lines on the live maps pass 0.33–0.42 m from a wall
against 0.45 m on the offline map, and the hits are at the two left hairpins, as in October.

## Degraded pose (compete, offline map, 10 laps)

`sim_pose_degrade` defaults: +33 ms, tracking p50 8.4 / p95 41.4 ms, 1 cm and 0.005 rad white
noise, a bias of 0.07 m per axis reverting over 2 s, 0.15 dropouts per second of 0.3 s. The two
partial runs switch off one part each.

| pose | run 1 / 2 / 3 collisions | raw (s) | adjusted (s) | laps after the first (s) |
| --- | --- | --- | --- | --- |
| ground truth (2026-10-31, finals) | 1 / 0 / 0 | 120.04–120.64 | 120.04–130.64 | 11.48–12.56 |
| noise and bias only | 0 / 2 / 0 | 120.94–122.52 | 120.94–142.52 | 11.43–12.79 |
| latency and dropouts only | 4 / 12 / 7 | 144.70–151.00 | 185.62–271.00 | 12.88–16.92 |
| everything | 15 / 14 / 12 | 151.52–157.65 | 271.52–307.65 | 13.87–18.60 |

The race settings are tuned on a pose that is exact and on time. Noise and a bounded bias of
the size ORB-SLAM3 showed barely change the race; latency and dropouts cost about 2–3 s a lap
and several hits, although the state estimator already updates at the capture time and predicts
to the present. An earlier version of the degradation let the bias wander without bound (a random
walk); that is not how SLAM localising against a saved map behaves, and it was replaced.

## Not settled

- **Which of latency and dropouts.** They were switched off together.
- **Earlier attempts.** The first compete mapping runs never built a map: a failed extraction
  slept 2 s, the next pose then looked like a respawn, and the scans around it were dropped. With
  the retry made non-blocking and teleports told apart by speed, the runs above mapped every time.
  Practice runs 2 and 3 are from before that fix; they mapped on the first try, so it did not
  touch them. Four jobs failed to start the sim on two cluster nodes and were rerun on others.

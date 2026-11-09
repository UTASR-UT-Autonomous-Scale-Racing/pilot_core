# AutoDRIVE no-LiDAR race runs (2026-10-31)

The first valid compete-track runs under the no-LiDAR rule: map from the camera's
segmentation, no LiDAR buffer, no sim speed fused (`use_pose_speed: false`). The sim's pose is
still ground truth (ORB-SLAM3 does not run in the sim). Stock 2026-iros builds, L40S,
640x360, about 23 frames/s, `scripts/cluster/autodrive_job.sh`. Scored like the published
final: race time + 10 s per collision.

| run | laps | collisions | first lap, then | raw (s) | adjusted (s) |
|---|---|---|---|---|---|
| final 1 | 10 | 1 | 13.74, 11.48-12.56 | 120.64 | 130.64 |
| final 2 | 10 | 0 | 14.01, 11.59-12.00 | 120.04 | **120.04** |
| final 3 | 10 | 0 | 13.97, 11.62-12.08 | 120.46 | **120.46** |
| Ultra quality | 10 | 1 | 13.53, 11.52-12.68 | 119.52 | 129.52 |
| competition start (sim restarted, auto_arm, no keys) | 10 | 1 | 22.51, 11.57-12.63 | 129.53 | 139.53 |
| qualifier, practice track | 10 | 0 | 9.37, 7.30-7.33 | 75.21 | 75.21 |

Published final: 1st 71.82 s (best lap 7.16 s), 14th 107.77 s, 15th 113.19 s, 23 teams. 120 s
is behind 15th. The qualifier needs 10 clean laps, which it did, in one run.

The LiDAR-map runs (bench/2026-10-29_autodrive) did 107.17-107.62 s, flying laps 10.4-10.7 s.
They do not count: LiDAR-built map, sim speed fused. Flying laps here are about 11.7 s; at
margin 0.5 on the camera map below they were about 11.1 s, so the margin costs some 0.6 s a lap.

## The margin decides it

Same five recordings, same settings otherwise. Before the reach limit and margin 0.65:

| run (margin 0.5, no reach limit) | collisions | raw (s) | adjusted (s) |
|---|---|---|---|
| final 1 / 2 / 3 | 21 / 6 / 9 | 126.78 / 114.66 / 117.16 | 336.78 / 174.66 / 207.16 |
| Ultra | 5 | 112.72 | 162.72 |
| competition start | 26 | 142.09 | 402.09 |

All hits were at the two left hairpins. The camera map had a pocket past the real wall at
about (-0.3, 3.5), so the line used room that isn't there, and at the bottom hairpin it passed
0.27 m from the LiDAR map's inner tip (the clean LiDAR-map line: 0.46 m). After a hit the car
respawned at (0.8, 3.7), too close to the line's corner to recover, and hit again every 0.8 s:
20 of the 26 in the competition start.

## The map

- The mapping lap never got round clean: 2-3 hits a lap in every setting tried (more wall
  clearance, a wider bubble, slower, edge memory up to 5 s; `sweep` stage), down from nearly 5.
- The cloud MappingWorker collects is not usable on this track: within 15 cm of the LiDAR
  walls by range (replayed masks), straights / corners:

| scan range (m) | straights | corners |
|---|---|---|
| 0.6-0.9 | 99 % | 68 % |
| 0.9-1.2 | 91 % | 42 % |
| 1.2-1.5 | 76 % | 40 % |
| 1.5-2.0 | 66 % | 31 % |
| 2.0-3.0 | 61 % | 22 % |

- `scripts/seg_map.py` builds the map from the recordings instead (five mapping runs, about 7
  laps): an occupancy grid where rays free the floor they cross, masks near a hit or with the
  car standing dropped, no wall where the car drove, outer wall closed, at most 0.9 m from the
  driven path. Against the LiDAR map, used only to score it: outer wall median 0.13 m, p90 0.30,
  max 0.51; inner 0.10, 0.28, 0.42. Half-width min 0.40 m.
- Line (margin 0.65): 44.4 m, max curvature 2.97 1/m, closest wall 0.45 m on the camera map.

Thresholds (`--ratio`, `--min-occ`, `--edge-range`) were picked looking at the distance to
the LiDAR map, so that much of the method saw it; nothing from that map is in the result.

## Not settled

- **The obstacle near (0.4, 4.0).** The top hairpin's hits line up with a straight edge the
  camera sees across the outer side at y about 4. It may be a lighting edge or something
  physical; the LiDAR map shows no wall there. The margin avoids it, the cause is unknown.
- **Not like-for-like with the leaderboard.** The track was known, the map came from 5
  runs on it, settings were tuned on it, and the segmentation model's labels were LiDAR-derived
  (training only). The sim's pose is ground truth.
- **Ultra here is 640x360.** At 1920x1080 Ultra on the vast.ai box the sim ran at 0.52x, 5.2
  frames/s, with 9 collisions, CPU-bound on one core; the organisers' resolution is unknown.
- **Clean is not guaranteed.** 2 of the 5 compete runs had no hit, the others one each.

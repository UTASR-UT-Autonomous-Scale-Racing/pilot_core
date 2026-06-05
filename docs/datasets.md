# Datasets, models and recordings

Data and trained models stay out of git (`.gitignore`). These are the bundles the team
needs, with where each file goes in a checkout and how to rebuild it from scratch.

**Status (2026-10-21):** all three bundles are built and checksummed, but **not yet uploaded**.
Proposed home: GitHub Release `data-v1` on this repo; every bundle is under the 2 GB
asset limit. After upload:

```bash
gh release download data-v1 -R UTASR-UT-Autonomous-Scale-Racing/pilot_core -D /tmp/data-v1
cd /tmp/data-v1 && sha256sum -c SHA256SUMS
tar --zstd -xf pilot_core-models-v1.tar.zst -C <pilot_core checkout>
tar --zstd -xf pilot_core-dataset-sim_track_seg-v1.tar.zst -C <pilot_core checkout>    # -> datasets/
tar --zstd -xf pilot_core-telemetry-showcase-v1.tar.zst -C <any dir>                    # -> export/, hud_full.mp4
```

## Bundles

| bundle | size | contents | needed for |
| ------ | ---- | -------- | ---------- |
| `pilot_core-models-v1.tar.zst` | 14 MB | `sim_track_seg.onnx` + `.json` (Fast-SCNN, sim track segmentation); `rl/checkpoints/PPO_1/` (policy ONNX at 4.5M steps, best model, VecNormalize stats, the two live fine-tune checkpoints, configs, eval log); `rl/checkpoints/PPO_2/` (experimental, see below) | `vision.model: onnx` in the sim; `control_mode: policy`; resuming RL training |
| `pilot_core-dataset-sim_track_seg-v1.tar.zst` | 76 MB | `datasets/sim_track_practice/` (1417 RGB + mask pairs, 4 runs) and `datasets/sim_track_compete/` (926 pairs, 2 runs, held out). `source.json` in each lists the source runs | retraining or evaluating the segmentation model |
| `pilot_core-telemetry-showcase-v1.tar.zst` | 115 MB | `export/` of the README run (mapping lap, then a pure_pursuit lap, every buffer on one clock, with its own `README.md`) and `hud_full.mp4` (the full-length HUD) | the README HUD, telemetry tooling, a worked example of every buffer format |

SHA-256:

```
b0a6b929f41f5a60e5d47a459657be4f5bd450017464aa406a5a4016c730e09e  pilot_core-models-v1.tar.zst
d7ac7edd1d590c170699478e61acb6066754f559790566d782a45d38d12df57b  pilot_core-dataset-sim_track_seg-v1.tar.zst
efcb62577c7d00e9e67429a4b3f19de8d67359d00a4432a39169552d5aaace0b  pilot_core-telemetry-showcase-v1.tar.zst
```

## Models

| file | config key | result | rebuild |
| ---- | ---------- | ------ | ------- |
| `sim_track_seg.onnx` | `vision.onnx_model` | IoU 0.941 on the unseen compete track (926 images), 0.939 live in `VisionWorker` | `scripts/train_vision.py`, [autodrive.md](autodrive.md#vision-visionworker-in-the-sim) |
| `rl/checkpoints/PPO_1/policy_4500000.onnx` | `track_control.policy.model` | bicycle-model pretrain, zero-shot on the live sim: lap 1 in 17.58 s vs pure_pursuit's 18.09 s at the same 2.0 m/s cap; ~30 % of live episodes collide | `rl/pretrain_bicycle.py`, then `rl/export_onnx.py` |
| `rl/checkpoints/PPO_2/policy_5500000.onnx` | – | **experimental, undocumented.** A second pretrain (9M steps, `rl/boundaries_easy`, 4.5 m/s cap); never benchmarked live | – |

`PPO_1/ppo_track_4510000_steps.zip` and `4520000` are the live fine-tune (26.5k steps,
121 episodes). It was stopped because it did not improve; the 4.5M checkpoint is untouched.

## Not bundled (rebuildable or too large)

| data | size | where | how to get it |
| ---- | ---- | ----- | ------------- |
| AutoDRIVE RoboRacer sim builds | – | `/workspaces/autodrive_sim` | `scripts/vast_autodrive_setup.sh` downloads the release; `scripts/autodrive_texture_sim.py` makes the textured copy |
| ORB-SLAM3 vocabulary and build | ~930 MB | `ORB_SLAM3/` submodule | `scripts/setup.sh --with-slam`, [slam.md](slam.md) |
| Saved track maps | 28 KB | `maps/autodrive_{practice,compete}` | in git |
| Benchmark raw runs, 2026-10-20 | 1.3 GB | maintainer archive | `scripts/autodrive_bench.py matrix bench/matrix.yaml`. Summary tables are in git (`bench/2026-10-20/`) |
| Validation evidence (runs, GIFs, logs) | 1.1 GB | maintainer archive | `scripts/autodrive_validate.py`; summary in [results.md](results.md) |
| Real-car recordings (OAK-D) | – | **none yet** | `start.py --recording`; see "Still to collect" |

## Still to collect

1. **Real-car OAK-D recordings** of the track: hand-driven mapping laps with RC
   (`--recording`, `frame.sample_k: 1`). These are the input for BC data
   (`recording_to_dataset.py --bc`) and for real-world vision labels
   (`--vision --bootstrap-masks`). Record over USB 3 ([oakd.md](oakd.md)).
2. **Real track segmentation set.** The sim set does not transfer to the real camera
   (different sensor, mount and lighting). Bootstrap the masks from (1), then hand-correct them.
3. **Joshua Choi's sim build** with a front depth camera. ORB-SLAM3 and the
   depth controllers need it in the sim.

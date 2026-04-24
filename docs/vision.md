# Vision

`VisionWorker` turns each camera frame into a **drivable-track mask**: a uint8
image at frame resolution where 1 = drivable track and 0 = everything else
(track walls, off-track floor, background). It publishes the mask to `seg_buf`.

Today `seg_buf` is recorded and published as telemetry; no controller or mapping step uses it yet.

## Segmenters

Choose one with `vision.model` in the config:

| `vision.model` | What it is | Needs |
| -------------- | ---------- | ----- |
| `pilot_vision` (default) | A model from the `pilot_vision` submodule (e.g. its DeepLabV3), chosen by `segmentation_config_path`. Its output is converted to the mask format | pilot_vision model files |
| `onnx` | A candidate network trained with `scripts/train_vision.py` (`vision.onnx_model`) | trained `.onnx` + `.json` |
| `ground_plane` | Training-free: fits the floor plane to the depth image (RANSAC) and marks the floor between the car and the first obstacle in each column | depth camera, intrinsics (`slam.settings`) |

## Candidate networks

All candidates take an RGB frame and output one channel of track probability (`vision/models.py`):

| `--model` | Network | Params | CPU latency |
| --------- | ------- | ------ | ----------- |
| `fast_scnn` (default) | Fast-SCNN | 1.1 M | 7 ms |
| `lraspp` | LR-ASPP, MobileNetV3-Large backbone | 3.2 M | 22 ms |
| `deeplabv3` | DeepLabV3, MobileNetV3-Large backbone | 11.0 M | 29 ms |
| `unet` | Compact U-Net | 2.0 M | 89 ms |
| `ground_plane` | RANSAC floor + free space (no network) | – | 7 ms |

Latency is ONNX Runtime on one CPU thread at a 320×192 input on a 4-core i5-1145G7
(`python scripts/vision_bench.py`). `lraspp` and `deeplabv3` can start from an
ImageNet-pretrained backbone (`--pretrained`), which usually helps with small datasets.
References are in [references.md](references.md).

The candidates have not been trained on track data yet, so there are no accuracy numbers.

## Training a candidate

Put labelled frames in this layout (masks from the SAM2 auto-labelling tools in
`pilot_vision` fit once exported as single-channel PNGs):

```
<data>/images/<name>.png|.jpg     camera frames
<data>/masks/<name>.png           non-zero = drivable track
```

```bash
python scripts/train_vision.py --data datasets/track --model fast_scnn
python scripts/train_vision.py --data datasets/track --model lraspp --pretrained --epochs 60 --out lraspp.onnx
```

Training uses flip and colour augmentation, a BCE + Dice loss and AdamW with a cosine
schedule. It holds out `--val` of the images and exports the weights with the best
validation IoU. The script prints the validation IoU and writes `<out>.onnx` plus a
`<out>.json` sidecar (input size, threshold, IoU). To use it:

```yaml
vision:
  model: onnx
  onnx_model: track_seg.onnx
```

ONNX Runtime uses the fastest available provider (CUDA/TensorRT on the Jetson with
`onnxruntime-gpu`, otherwise CPU), so the car does not need torch.

## Adding a candidate

Add the network to `vision/models.py` (input RGB `(N, 3, H, W)`, output logits
`(N, 1, H, W)`) and to `MODELS`. Training, export, the benchmark and the `onnx` mode
then work unchanged. Add a shape test in `tests/test_vision.py` and a reference in
[references.md](references.md).

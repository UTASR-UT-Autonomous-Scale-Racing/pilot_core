# Shared Buffer Schema

This document defines the binary schema used by the shared-memory ring buffers in `pilot_core`.

## Overview

Each slot in every buffer contains:

1. A fixed-size 64-byte slot header.
2. A payload region with buffer-specific binary layout.

Code references:

- Header schema: `shared_buffer/schema.py`
- Ring implementation: `shared_buffer/ring_buffer.py`
- Typed buffer wrappers: `shared_buffer/*.py`

## Slot Header (64 bytes)

Little-endian format string:

`<QddIHH32s`

Field layout:

| Offset | Size | Type | Field | Description |
|---:|---:|---|---|---|
| 0 | 8 | uint64 | `seq_no` | Monotonic write sequence number |
| 8 | 8 | float64 | `produced_ts` | Writer timestamp from `time.time()` |
| 16 | 8 | float64 | `read_ts` | Reader timestamp set on `read_latest()` |
| 24 | 4 | uint32 | `payload_size` | Actual bytes used in payload |
| 28 | 2 | uint16 | `schema_version` | Schema version (`1`) |
| 30 | 2 | uint16 | `dtype_tag` | Payload type tag |
| 32 | 32 | bytes | `_reserved` | Padding / future use |

## DType Tags

Defined in `shared_buffer/schema.py`:

| Value | Name |
|---:|---|
| 0 | `UNKNOWN` |
| 1 | `FRAME` |
| 2 | `IMU` |
| 3 | `SEGMENTATION` |
| 4 | `POSE` |
| 5 | `CONTROL` |

Note: `PointCloudBuffer` and `TrackBoundaryBuffer` currently use `UNKNOWN`.

## Buffer Payload Schemas

## FrameBuffer

Source: `shared_buffer/frame_buffer.py`

Defaults:

- shm name: `frame_buf`
- slots: `8`

Payload layout:

1. `uint32 rgb_height`
2. `uint32 rgb_width`
3. `uint32 depth_height`
4. `uint32 depth_width`
5. `rgb_height * rgb_width * 3` bytes RGB image (`uint8`, row-major)
6. Depth image bytes (`depth_height * depth_width`, either `uint16` or `float32`)

Read behavior infers depth dtype from remaining payload size.

## IMUBuffer

Source: `shared_buffer/imu_buffer.py`

Defaults:

- shm name: `imu_buf`
- slots: `32`

Payload layout (exactly 64 bytes):

- 16 x `float32`:
  - `w, x, y, z`
  - `ax, ay, az`
  - `gx, gy, gz`
  - `mx, my, mz`
  - `throttle_raw, steering_raw`
  - `timestamp`

## SegmentationBuffer

Source: `shared_buffer/segment_buffer.py`

Defaults:

- shm name: `seg_buf`
- slots: `4`

Payload layout:

1. `uint32 height`
2. `uint32 width`
3. `height * width` values of `float32` mask data (row-major)

## PoseBuffer

Source: `shared_buffer/pose_buffer.py`

Defaults:

- shm name: `pose_buf`
- slots: `8`

Payload layout:

1. `float32[4][4]` pose matrix (64 bytes)
2. `uint32 num_map_points` (`N`)
3. `N x float32[3]` map points `(x, y, z)`

Wrapper clamps map points to `2048` max by default.

## ControlBuffer

Source: `shared_buffer/control_buffer.py`

Defaults:

- shm name: `ctrl_buf`
- slots: `8`

Payload layout (exactly 8 bytes):

1. `float32 steering`
2. `float32 throttle`

## PointCloudBuffer

Source: `shared_buffer/pointcloud_buffer.py`

Defaults:

- shm name: `pc_buf`
- slots: `4`

Payload layout:

1. `uint32 num_points` (`N`)
2. `N x float32[3]` point cloud coordinates `(x, y, z)`

Wrapper default max points per payload: `50,000`.

## TrackBoundaryBuffer

Source: `shared_buffer/track_boundary_buffer.py`

Defaults:

- shm name: `boundary_buf`
- slots: `2`

Payload layout:

1. `uint32 outer_count` (`N_outer`)
2. `N_outer x float32[2]` outer boundary points `(x, z)`
3. `uint32 inner_count` (`N_inner`)
4. `N_inner x float32[2]` inner boundary points `(x, z)`

Wrapper default max points per side: `10,000`.

## Shared Ring Semantics

Source: `shared_buffer/ring_buffer.py`

- Single producer, multi-consumer design.
- Writer increments `seq_no` and writes header + payload.
- Readers scan all slots and return the slot with highest `seq_no`.
- Reads are pop-most-recent snapshots, not queue pops.
- Payload writes larger than configured slot payload size are truncated.

## Buffer Exports

Public shared buffer API is exported in:

- `shared_buffer/__init__.py`

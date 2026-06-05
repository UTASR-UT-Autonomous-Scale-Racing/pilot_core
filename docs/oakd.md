# OAK-D camera

`CameraWorker` (`workers/camera_worker.py`) runs the OAK-D through the depthai v3 API: RGB
(CAM_A), stereo depth aligned to RGB, sync within half a frame period, and the on-board IMU.

## Bench check, 2026-10-21 (development PC)

| item | value |
| ---- | ----- |
| Model | OAK-D-PRO-FF-97 (board DM9098), RVC2 / MyriadX |
| Device ID | `14442C10F13ED2D200`; matches the header of `orbslam_oakd_rgbd.yaml` |
| Sensors | CAM_A OV9782 colour global shutter; CAM_B / CAM_C OV9282 mono |
| IMU | BNO086 (firmware 3.9.9); rotation vector available |
| Calibration | RGB at 640×400: fx 396.01, fy 395.82, cx 327.00, cy 188.99; stereo baseline 7.50 cm. Both match `orbslam_oakd_rgbd.yaml` |
| depthai | 3.11.0 |
| Link | **USB 2 (HIGH, 480 Mb/s)** on that port and cable |

Measured with the `CameraWorker` pipeline:

| fps setting | RGB-D pairs | latency median / p95 | IMU |
| ----------- | ----------- | -------------------- | --- |
| 15 (`config_cpu.yaml`) | 15.0 Hz, 0 drops | 77 / 93 ms | 100 Hz |
| 30 (`config.yaml`) | **24.7 Hz, 26 drops in 6 s** | **319 / 356 ms** | 100 Hz |

- The RGB and depth stamps of a pair are at most 7 ms apart.
- Hardware tests (`OAKD_AVAILABLE=1 pytest -m oakd`) pass. The real-SLAM test skips
  without the ORB-SLAM3 build for this Python.
- `start.py --config config_cpu.yaml` brings the camera up in about 4 s and runs at
  15 fps with ~72 ms latency.

## Requirements

- **USB 3 for 30 fps.** On USB 2, 30 fps RGB-D drops frames and lags about 320 ms, too old
  for the 150 ms command watchdog. Use a SuperSpeed port and a USB 3 cable; with USB-C, try
  flipping the plug. `CameraWorker` logs a warning when the link is slower than USB 3 and
  fps is above 15. Check the link:

  ```bash
  python -c "import depthai as dai; print(dai.Device().getUsbSpeed())"   # want UsbSpeed.SUPER
  ```
- **udev rule** for non-root access: `/etc/udev/rules.d/80-movidius.rules` with
  `SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"`, then
  `sudo udevadm control --reload-rules && sudo udevadm trigger`.

## Open items

- `config.yaml` asks for 30 fps, but `orbslam_oakd_rgbd.yaml` says `Camera.fps: 15`.
  Regenerate it with `scripts/gen_orbslam_settings.py` for the rate the car runs at.
- Depth was only about 33 % valid in a dim room at the 4000 µs exposure limit. The Pro has an
  IR dot projector that nothing enables yet; try `device.setIrLaserDotProjectorIntensity(0.5)`
  and revisit `max_exposure_us` indoors.
- depthai 3.11 deprecates `IMUSensor.ACCELEROMETER` (`camera_worker.py`, `_setup_imu`);
  the replacement is `ACCELEROMETER_CALIBRATED`. Pin `depthai` once the Jetson's version is
  confirmed, so 3.x API changes don't land unannounced.

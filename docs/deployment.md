# Deployment on the Jetson

Natively or in Docker; both use `config.yaml` (I2C actuator on `/dev/i2c-1`) unless
you pass `--config`. The Arduino follows the Jetson only while commands are fresh
(250 ms) and the RC sticks are centred (`firmware/README.md`).

> **Firmware not yet bench-tested.** The command-source selection and the 250 ms timeout in
> the firmware have not been compiled and run on the Arduino (`firmware/README.md`). Do the
> wheels-off bench test there before the car drives under Jetson control.

## What runs

Both scripts check out the newest `car-v*` release tag (a version that passed the bench test
and a drive), or `main` if there is no such tag yet, with the submodules at the commits it
pins. `PILOT_REF=<tag|branch|commit>` picks another. Then `scripts/jetson_selftest.py` checks
the platform, power mode, CUDA, `/dev/shm`, the ring buffers, the ORB-SLAM3 bindings, the
OAK-D on USB 3 and the Arduino on I2C; any failure stops the start (`SKIP_SELFTEST=1` skips it).

## Native

```bash
scripts/jetson_run.sh [start.py arguments]
```

Sets maximum performance, checks out the release (refuses with uncommitted changes), runs
the self-test, activates `.venv` and runs `start.py` as root (I2C and USB). Needs the `.venv`
and the ORB-SLAM3 build ([slam.md](slam.md)).

## Docker

```bash
scripts/start.sh [start.py arguments]
```

Checks out the release, sets performance mode and the OAK-D udev rule, builds `Dockerfile`,
runs the self-test in the container, then runs it in the foreground with:

| Flag | Why |
| ---- | --- |
| `--runtime nvidia` | GPU access |
| `--privileged` | USB access for the OAK-D |
| `--device /dev/i2c-1` | Arduino over I2C |
| `--net=host` | Telemetry and debugging |
| `--ipc=host` | Room in `/dev/shm` for the shared-memory buffers (Docker's default is 64 MB) |

Arguments go to `start.py`. The image builds ORB-SLAM3 (`--target noslam` skips it);
the base is JetPack 6 (`L4T_BASE`), which must match the Jetson's L4T major version.
Logs: `sudo docker logs -f utasr-container`. CI builds the same image on an arm64 runner
(`.github/workflows/jetson-image.yml`) and pushes it to `ghcr.io` for each `car-v*` tag.

## Measuring on the Jetson

Record a run with the actuator disabled and read the per-stage rates and latencies off it:

```bash
python start.py --config config_cpu.yaml --enable-slam --recording     # 60 s, then q
python scripts/stage_latency.py /tmp/pilot_recordings/<run>
```

Frames, masks and poses are stamped with the camera's capture time, so the table gives each
stage's capture-to-output latency, and the age of the newest pose whenever a command went out.
`SLAMWorker` also logs its mean tracking time every 30 frames. The same table for the sim race
is in [results.md](results.md#stage-latency); the Jetson's is still to be measured.

## Releasing a car version

After the firmware bench test and a drive on the track with a version, tag it so the car runs it:

```bash
git tag -a car-v1.0.0 -m "<what was driven, where>" && git push origin car-v1.0.0
```

## One-time Jetson setup for Docker

Install Docker and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
then add your user to the `docker` group.

Use the NVIDIA runtime by default and keep Docker's data off the root partition:

```bash
sudo systemctl stop docker
sudo mkdir -p /mnt/docker
sudo cp -r /var/lib/docker /mnt/docker   # keep existing images
sudoedit /etc/docker/daemon.json
```

```json
{
    "runtimes": {
        "nvidia": { "path": "nvidia-container-runtime", "runtimeArgs": [] }
    },
    "default-runtime": "nvidia",
    "data-root": "/mnt/docker",
    "iptables": false,
    "firewall-backend": "nftables"
}
```

```bash
sudo systemctl start docker
sudo docker info | grep 'Docker Root Dir'
```

# Deployment on the Jetson

Natively or in Docker; both use `config.yaml` (I2C actuator on `/dev/i2c-1`) unless
you pass `--config`. The Arduino follows the Jetson only while commands are fresh
(250 ms) and the RC sticks are centred (`firmware/README.md`).

## Native

```bash
scripts/jetson_run.sh [start.py arguments]
```

Sets maximum performance, fast-forwards pilot_core (refuses with uncommitted
changes), checks out the pinned submodules, activates `.venv` and runs `start.py`.
Needs the `.venv` and the ORB-SLAM3 build ([slam.md](slam.md)).

## Docker

```bash
scripts/start.sh [start.py arguments]
```

Updates the repo, sets performance mode and the OAK-D udev rule, builds `Dockerfile`
and runs it in the foreground with:

| Flag | Why |
| ---- | --- |
| `--runtime nvidia` | GPU access |
| `--privileged` | USB access for the OAK-D |
| `--device /dev/i2c-1` | Arduino over I2C |
| `--net=host` | Telemetry and debugging |
| `--ipc=host` | Room in `/dev/shm` for the shared-memory buffers (Docker's default is 64 MB) |

Arguments go to `start.py`. The image builds ORB-SLAM3 (`--target noslam` skips it);
the base is JetPack 6 (`L4T_BASE`), which must match the Jetson's L4T major version.
Logs: `sudo docker logs -f utasr-container`.

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

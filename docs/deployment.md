# Deployment on the Jetson

There are two ways to run pilot_core on the car's Jetson Orin: natively, or in
Docker. Both use `config.yaml` unless you pass `--config`. It enables the I2C
actuator on `/dev/i2c-1`.

The actuator is silent on I2C until the car is armed, so the RC transmitter drives
while disarmed (the hand-driven mapping lap). Armed, the Arduino follows the Jetson
as long as commands keep arriving (within 250 ms) and the RC sticks are centred;
moving a stick takes over. See the "Jetson interface" section in `firmware/README.md`.

## Native

```bash
scripts/jetson_run.sh [start.py arguments]
```

The script sets the Jetson to maximum performance (`nvpmodel -m 0`, `jetson_clocks`),
fast-forwards pilot_core (it refuses to run with uncommitted changes, so nothing is
discarded), puts the submodules on the commits pilot_core pins, activates `.venv`
and launches `start.py` (with `--config config.yaml` unless you pass one). It expects
the `.venv` and the ORB-SLAM3 build described in [slam.md](slam.md).

## Docker

```bash
scripts/start.sh [start.py arguments]
```

`start.sh` fast-forwards pilot_core, puts the submodules on their pinned commits,
checks the config exists, sets performance mode and the OAK-D udev rule (once),
builds the image from `Dockerfile`, and runs it in the foreground with:

| Flag | Why |
| ---- | --- |
| `--runtime nvidia` | GPU access |
| `--privileged` | USB access for the OAK-D |
| `--device /dev/i2c-1` | Arduino over I2C |
| `--net=host` | Telemetry and debugging |
| `--ipc=host` | Room in `/dev/shm` for the shared-memory buffers (Docker's default is 64 MB) |

The image's entrypoint is `start.py`, and any arguments are forwarded to it. By
default the image builds ORB-SLAM3 and the Python wrapper (the `ORB_SLAM3` submodule
must be checked out). `docker build --target noslam .` skips SLAM for a quick build.
The base image is JetPack 6 (`L4T_BASE` build argument); the container's L4T major
version must match the Jetson's.

Useful commands:

```bash
sudo docker logs -f utasr-container     # follow the logs
sudo docker system prune -a             # remove unused images
```

## One-time Jetson setup for Docker

Prerequisites: NVIDIA driver, Docker and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

**Install Docker (Ubuntu):**

```bash
sudo apt remove $(dpkg --get-selections docker.io docker-compose docker-compose-v2 docker-doc podman-docker containerd runc | cut -f1)
sudo apt update && sudo apt install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
sudo tee /etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Signed-By: /etc/apt/keyrings/docker.asc
EOF
sudo apt update
sudo apt install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER            # then log out and back in
```

**Use the NVIDIA runtime by default and move Docker's data off the root partition:**

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

If large models run out of memory, add swap as described in the
[jetson-containers setup guide](https://github.com/dusty-nv/jetson-containers/blob/master/docs/setup.md).

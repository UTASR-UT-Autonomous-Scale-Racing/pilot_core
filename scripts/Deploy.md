# Docker Deployment Guide

## Prerequisites

- On JetSon Orin
- NVIDIA Driver
- [NVIDIA Container ToolKit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (includes NVDIA Container Runtime)
- Docker
- Git and GitHub Token at `~/Documents/git_token.git`

Suspect not required:

- [JetSon Container Utils](https://github.com/dusty-nv/jetson-containers/blob/master/docs/setup.md)

The container also requires hardware access to:

| Device         | Purpose             | Flag                  |
| -------------- | ------------------- | --------------------- |
| `/dev/i2c-1`   | Board communication | `--device /dev/i2c-1` |
| `/dev/nvidia*` | GPU inference       | `--runtime nvidia`    |
| USB devices    | OAK-D camera        | `--privileged`        |

```toml
# Not sure if X11 (JetSon Screen Display) and Network is required
  | Network    | Sensors/debugging | `--net=host` |
```

**File Structure**

```
pilot_core/
    ├── start.sh
    ├── Dockerfile
    ├── _start.py
    ├── config.yaml
    ├── pilot_ingest/
    ├── pilot_vision/
    ├── pilot_controller/
    └── ...
```

## Quick Start

Run the following script will clone/update all repositories ,validate configurations, build Docker image, start the container. The `<Arguments>` field is specified in `_start.py`.

```bash
chmod +x start.sh
./start.sh <Arguments>
```

**Helpful Docker Commands**

```bash
# Removes unused images
sudo docker system prune -a

# Checks log
sudo docker logs utasr-container
sudo docker logs -f utasr-container
```

## Docker Setup Guide

**Install Docker via the following command:**
Run the following commands to install docker on Ubuntu.

```bash
# Uninstall conflict packages
sudo apt remove $(dpkg --get-selections docker.io docker-compose docker-compose-v2 docker-doc podman-docker containerd runc | cut -f1)

# Add Docker's official GPG key:
sudo apt update
sudo apt install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
sudo tee /etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Signed-By: /etc/apt/keyrings/docker.asc
EOF

sudo apt update

sudo apt install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

**Update Default Docker Runtime and data root**

Run the following commands in order to align with NVIDIA JetSon Container and avoid running out of space on the root partition:

```bash
# Stops the docker from running
sudo systemctl stop docker

# Creates a new folder for docker root dir
sudo mkdir -p /mnt/docker

# Migrates the existing cache if necessary
sudo cp -r /var/lib/docker /mnt/docker

# Edit the daemon setting file (feel free to use any other editor)
sudo vi /etc/docker/daemon.json
```

Edit the file to look like the following:

```
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },

    "default-runtime": "nvidia",
    "data-root": "/mnt/docker",
    "iptables": false,
    "firewall-backend": "nftables"
}
```

Finally, restart the docker and confirm the changes by:

```bash
sudo systemctl start docker
sudo docker info | grep 'Docker Root Dir'
```

**Set sudo right for Docker**

```bash
sudo usermod -aG docker $USER
```

Log out and log back in to apply the changes.

**Mounting Swap**
If the memory usage is large (large models ..), refer to the official [jetson-container instruction](https://github.com/dusty-nv/jetson-containers/blob/master/docs/setup.md) to mount SWAP.

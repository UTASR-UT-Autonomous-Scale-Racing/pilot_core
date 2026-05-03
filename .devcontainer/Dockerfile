# .devcontainer/Dockerfile — VS Code dev container for developing on the
# Jetson Orin. Used by .devcontainer/devcontainer.json.
# Same base as the production Dockerfile (JetPack 6 / L4T r36, Ubuntu 22.04,
# Python 3.10); the host must run JetPack 6.x. The base image is large and slow to pull.
# The workspace is bind-mounted by VS Code, not copied; Python deps are installed
# into .venv by postCreateCommand (requirements.txt + `pip install -e .`).
ARG L4T_BASE=nvcr.io/nvidia/l4t-jetpack:r36.4.0
FROM ${L4T_BASE}

# Runtime (USB/I2C) + tools to build ORB-SLAM3 and its wrapper (docs/slam.md)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    libusb-1.0-0 udev i2c-tools \
    git wget ca-certificates sudo \
    build-essential cmake pkg-config \
    libeigen3-dev libepoxy-dev libglew-dev libgl1-mesa-dev libegl1-mesa-dev \
    libwayland-dev wayland-protocols libxkbcommon-dev \
    libopencv-dev libboost-serialization-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Non-root user with passwordless sudo and access to video / serial / USB / I2C devices
ARG USERNAME=developer
ARG USER_UID=1000
ARG USER_GID=1000
RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} -s /bin/bash \
    && (getent group i2c >/dev/null || groupadd --system i2c) \
    && usermod -aG sudo,video,dialout,plugdev,i2c ${USERNAME} \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >/etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}

ENV PYTHONUNBUFFERED=1

USER ${USERNAME}
CMD ["bash"]

# VS Code dev container for the Jetson Orin (same JetPack 6 base as ../Dockerfile).
# Workspace is bind-mounted; Python deps go into .venv via postCreateCommand.
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

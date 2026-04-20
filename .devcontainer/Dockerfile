# .devcontainer/Dockerfile — VS Code dev container for developing on the
# Jetson (JetPack r36.4.0; the base image is large and slow to pull).
# Used by .devcontainer/devcontainer.json.
FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libusb-1.0-0 \
    libusb-1.0-0-dev \
    udev \
    i2c-tools \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
RUN pip3 install --no-cache-dir \
    PyYAML \
    tqdm \
    pandas \
    matplotlib \
    smbus2 \
    depthai

WORKDIR /app/pilot_core
COPY . /app/pilot_core

# Non-root user with passwordless sudo and access to video / serial / USB devices
ARG USERNAME=developer
ARG USER_UID=1000
ARG USER_GID=1000
RUN apt-get update && apt-get install -y --no-install-recommends sudo \
    && groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} -s /bin/bash \
    && usermod -aG sudo,video,dialout,plugdev ${USERNAME} \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >/etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME} \
    && rm -rf /var/lib/apt/lists/* \
    && getent passwd ${USERNAME}

ENV PYTHONUNBUFFERED=1

USER ${USERNAME}
CMD ["bash"]

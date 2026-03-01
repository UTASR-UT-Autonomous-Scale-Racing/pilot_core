# Most libraries included
# this takes a lot of time
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

# Install Python dependencies
# numpy, opencv, torch, torchvision already included in the base image
RUN pip3 install --no-cache-dir \
    PyYAML \
    tqdm \
    pandas \
    matplotlib \
    smbus2 \
    depthai 
    # onnxruntime-gpu

# Creates application directory
WORKDIR /app/pilot_core

# Copies all application code
COPY . /app/pilot_core

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

# Most libraries included
FROM nvcr.io/nvidia/l4t-ml:r35.2.1-py3

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
    "depthai>=3.0" \
    scipy \
    onnxruntime-gpu

# Creates application directory
WORKDIR /app/pilot_core

# Copies all application code
COPY . /app/pilot_core

# Flush output to the terminal immediately
ENV PYTHONUNBUFFERED=1

# start.py is the entrypoint; container args are forwarded to it, e.g.
#   docker run ... pilot_core --config config.yaml --recording
ENTRYPOINT ["python3", "start.py"]
CMD ["--config", "config.yaml"]

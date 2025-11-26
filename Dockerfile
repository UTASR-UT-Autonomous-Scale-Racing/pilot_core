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
    depthai \
    onnxruntime-gpu

# Creates application directory
WORKDIR /app/pilot_core

# Copies all application code
COPY . /app/pilot_core


# Environment variables 
# flushes output to the terminal immediately
ENV PYTHONUNBUFFERED=1   
ENV SEGMENTATION_CONFIG=""
ENV CONTROL_CONFIG=""

# Executes _start.py as entrypoint, args forwarded via '$@'
RUN echo '#!/bin/bash\n\
set -e\n\
echo "Starting the container..."\n\
echo "Segmentation config: $SEGMENTATION_CONFIG"\n\
echo "Control config: $CONTROL_CONFIG"\n\
exec python3 _start.py \\\n\
    --segmentation_config "$SEGMENTATION_CONFIG" \\\n\
    --control_config "$CONTROL_CONFIG" \\\n\
    "$@"\n\
' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]

# defaults arg to the /app/entrypoint.sh
CMD []   



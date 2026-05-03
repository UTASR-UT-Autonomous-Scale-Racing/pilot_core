# Dockerfile — runtime image for the car (Jetson Orin).
# Built and run by scripts/start.sh:
#   docker build -t utasr-image -f Dockerfile .                   # with ORB-SLAM3 (default target)
#   docker build -t utasr-image -f Dockerfile --target noslam .   # without SLAM (fast build)
#
# Base: JetPack 6 / L4T r36 (Ubuntu 22.04, Python 3.10, which pilot_core requires).
# The container's L4T major version must match the host's, so the Jetson must run
# JetPack 6.x. The previous base (l4t-ml:r35.2.1, JetPack 5) ships Python 3.8.
#
# Stages:
#   deps        system + Python runtime dependencies (requirements.txt)
#   noslam      deps + the pilot_core sources; SLAMWorker falls back to its stub
#   slam-build  deps + Pangolin, ORB_SLAM3 and orbslam3_wrapper, built as in docs/slam.md
#   runtime     slam-build + the pilot_core sources (default)
#
# The SLAM stages need the ORB_SLAM3 submodule checked out in the build context
# (scripts/setup.sh --with-slam). The first SLAM build takes ~30+ min on the
# Jetson; it is cached afterwards because only ORB_SLAM3/ and orbslam3_wrapper/
# are copied into that stage.

ARG L4T_BASE=nvcr.io/nvidia/l4t-jetpack:r36.4.0

# ---------------------------------------------------------------------------
FROM ${L4T_BASE} AS deps

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    libusb-1.0-0 \
    udev \
    i2c-tools \
    && rm -rf /var/lib/apt/lists/*

# Python runtime deps. requirements.txt installs CPU onnxruntime: onnxruntime-gpu
# has no aarch64 wheel on PyPI, and the controllers (control/policy.py,
# control/imitation.py) use the CPU execution provider anyway. For GPU inference
# of vision.model: onnx, pass the Jetson AI Lab wheel index for your JetPack, e.g.
#   docker build --build-arg ORT_GPU_INDEX=https://pypi.jetson-ai-lab.io/jp6/cu126 ...
# (check https://pypi.jetson-ai-lab.io for the current URL).
ARG ORT_GPU_INDEX=""
WORKDIR /app/pilot_core
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt \
    && if [ -n "${ORT_GPU_INDEX}" ]; then \
         pip3 uninstall -y onnxruntime \
         && pip3 install --no-cache-dir --extra-index-url "${ORT_GPU_INDEX}" onnxruntime-gpu; \
       fi

# Flush output to the terminal immediately
ENV PYTHONUNBUFFERED=1
# ORB-SLAM3 + Pangolin shared libraries (present only in the SLAM stages)
ENV LD_LIBRARY_PATH=/app/pilot_core/ORB_SLAM3/lib:/usr/local/lib:${LD_LIBRARY_PATH}

# ---------------------------------------------------------------------------
FROM deps AS noslam

COPY . /app/pilot_core

# start.py is the entrypoint; container args are forwarded to it, e.g.
#   docker run ... utasr-image --config config.yaml --recording
ENTRYPOINT ["python3", "start.py"]
CMD ["--config", "config.yaml"]

# ---------------------------------------------------------------------------
FROM deps AS slam-build

# Build dependencies from docs/slam.md (Ubuntu 22.04). The -dev packages stay in
# the image so libORB_SLAM3.so finds the same OpenCV/Boost/GL libraries at runtime
# (a slimmer copy-only runtime stage would have to list every runtime .so).
# If your L4T base already ships NVIDIA's OpenCV (e.g. l4t-ml), drop libopencv-dev.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git pkg-config ca-certificates \
    libeigen3-dev libepoxy-dev libglew-dev libgl1-mesa-dev libegl1-mesa-dev \
    libwayland-dev wayland-protocols libxkbcommon-dev \
    libopencv-dev libboost-serialization-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Pangolin (ORB-SLAM3 dependency; the viewer is not used)
ARG PANGOLIN_REF=master
RUN git clone --depth 1 --branch ${PANGOLIN_REF} https://github.com/stevenlovegrove/Pangolin.git /tmp/Pangolin \
    && cmake -S /tmp/Pangolin -B /tmp/Pangolin/build \
         -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_TOOLS=OFF \
    && cmake --build /tmp/Pangolin/build -j3 \
    && cmake --install /tmp/Pangolin/build \
    && ldconfig \
    && rm -rf /tmp/Pangolin

# ORB_SLAM3: third-party libs, vocabulary, then only the library target (the
# example executables are not needed). Host build outputs are excluded by
# .dockerignore, so this always builds against the image's libraries.
COPY ORB_SLAM3 /app/pilot_core/ORB_SLAM3
RUN test -f /app/pilot_core/ORB_SLAM3/CMakeLists.txt \
    || { echo "ORB_SLAM3 submodule is empty: run scripts/setup.sh --with-slam" >&2; exit 1; }
RUN cd /app/pilot_core/ORB_SLAM3 \
    && cmake -S Thirdparty/DBoW2 -B Thirdparty/DBoW2/build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build Thirdparty/DBoW2/build -j3 \
    && cmake -S Thirdparty/g2o -B Thirdparty/g2o/build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build Thirdparty/g2o/build -j3 \
    && (cd Vocabulary && tar xf ORBvoc.txt.tar.gz) \
    && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build -j3 --target ORB_SLAM3

# pybind11 wrapper; writes orbslam3_python*.so to /app/pilot_core (repo root)
COPY orbslam3_wrapper /app/pilot_core/orbslam3_wrapper
RUN cmake -S /app/pilot_core/orbslam3_wrapper -B /app/pilot_core/orbslam3_wrapper/build \
         -Dpybind11_DIR="$(pybind11-config --cmakedir)" -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /app/pilot_core/orbslam3_wrapper/build -j3 \
    && python3 -c "import sys; sys.path.insert(0, '/app/pilot_core'); import orbslam3_python; print('orbslam3_python ok')"

# ---------------------------------------------------------------------------
FROM slam-build AS runtime

# Adds the sources on top of the SLAM build (COPY merges; the build outputs stay).
COPY . /app/pilot_core

ENTRYPOINT ["python3", "start.py"]
CMD ["--config", "config.yaml"]

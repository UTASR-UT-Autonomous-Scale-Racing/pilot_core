# Car runtime image (Jetson Orin, JetPack 6 / L4T r36 for Python 3.10; host must run JetPack 6.x).
# Stages: deps -> noslam (SLAM stub) | slam-build (needs scripts/setup.sh --with-slam) -> runtime (default).
#   docker build -t utasr-image -f Dockerfile [--target noslam] .

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

# CPU onnxruntime (no aarch64 GPU wheel on PyPI); for GPU pass a Jetson AI Lab index, e.g.
#   --build-arg ORT_GPU_INDEX=https://pypi.jetson-ai-lab.io/jp6/cu126
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

# container args are forwarded to start.py
ENTRYPOINT ["python3", "start.py"]
CMD ["--config", "config.yaml"]

# ---------------------------------------------------------------------------
FROM deps AS slam-build

# Build deps (docs/slam.md); -dev packages stay so libORB_SLAM3.so finds its libs at runtime.
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

# ORB_SLAM3: third-party libs, vocabulary, library target only (host builds are .dockerignored)
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
         -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=87 \
    && cmake --build build -j3 --target ORB_SLAM3

# pybind11 wrapper; writes orbslam3_python*.so to /app/pilot_core (repo root)
COPY orbslam3_wrapper /app/pilot_core/orbslam3_wrapper
RUN cmake -S /app/pilot_core/orbslam3_wrapper -B /app/pilot_core/orbslam3_wrapper/build \
         -Dpybind11_DIR="$(pybind11-config --cmakedir)" -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /app/pilot_core/orbslam3_wrapper/build -j3 \
    && python3 -c "import sys; sys.path.insert(0, '/app/pilot_core'); import orbslam3_python; print('orbslam3_python ok')"

# ---------------------------------------------------------------------------
FROM slam-build AS runtime

# COPY merges, so the SLAM build outputs stay
COPY . /app/pilot_core

ENTRYPOINT ["python3", "start.py"]
CMD ["--config", "config.yaml"]

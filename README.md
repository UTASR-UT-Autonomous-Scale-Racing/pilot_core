# pilot_core

Central control hub for UTASR's autonomous scale racing project — coordinates control logic, data ingestion, and vision processing for an RC-car.

## Overview

`pilot_core` integrates these component repositories as git submodules:

- [`pilot_controller`](https://github.com/UTASR-UT-Autonomous-Scale-Racing/pilot_controller) — steering & throttle control
- [`pilot_ingest`](https://github.com/UTASR-UT-Autonomous-Scale-Racing/pilot_ingest) — sensor data ingestion
- [`pilot_vision`](https://github.com/UTASR-UT-Autonomous-Scale-Racing/pilot_vision) — segmentation & vision processing

### Project Structure

```
pilot_core/
├── workers/          # Pipeline workers (camera, SLAM, mapping, control, …)
├── shared_buffer/    # Lock-free shared-memory ring buffers
├── synthetic/        # Synthetic data generators for testing
├── tests/            # pytest suite (189 tests)
├── src/              # C++ entry points (main, replay)
├── include/          # C++ headers
├── scripts/          # Helper shell scripts & extra docs
├── config*.yaml      # Runtime configuration files
├── start.py          # Python entry point
├── Dockerfile        # Jetson / GPU container
└── Dockerfile.dev-local  # Lightweight dev container (no GPU)
```

## Setup

### 1. Clone & init submodules

```bash
git clone <repo-url> && cd pilot_core
git submodule update --init --recursive
```

### 2a. Local Dev Container (recommended for development)

Open in VS Code with the **Dev Containers** extension — use the `.devcontainer/dev-local/` profile:

- No GPU / OAK-D camera required
- Runs in synthetic mode (`PILOT_SYNTHETIC=1`)
- Python-only — all tests run out of the box

```bash
# Or build manually:
docker build -f Dockerfile.dev-local -t pilot_core:dev-local .
docker run --rm -it pilot_core:dev-local
```

### 2b. Jetson / GPU Container

Use the root `.devcontainer/devcontainer.json` or build directly:

```bash
docker build -t pilot_core .
```

Requires NVIDIA runtime, OAK-D camera, and ORB_SLAM3 built in `ORB_SLAM3/lib/`.

### 3. Python venv (inside container)

```bash
python3 -m venv venv && source venv/bin/activate
pip install PyYAML numpy opencv-python-headless scipy pytest pytest-timeout matplotlib pandas tqdm zmq
```

### 4. Building ORB-SLAM3 + pybind11 wrapper

ORB-SLAM3 is included as a git submodule. The Python wrapper (`orbslam3_python`) uses pybind11 to expose SLAM functionality to the Python pipeline.

**Prerequisites** (install inside the container if not already present):

```bash
apt-get update && apt-get install -y \
    build-essential cmake git pkg-config \
    libeigen3-dev libglew-dev libgl1-mesa-dev libwayland-dev \
    libopencv-dev libboost-serialization-dev libssl-dev
```

**Step 1 — Install Pangolin** (3D visualization library required by ORB-SLAM3):

```bash
cd /tmp && git clone --depth 1 https://github.com/stevenlovegrove/Pangolin.git
cd Pangolin && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF
make -j3 && make install && ldconfig
```

**Step 2 — Build ORB-SLAM3** (thirdparty libs + main library):

```bash
cd ORB_SLAM3

# Build DBoW2
cd Thirdparty/DBoW2 && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j3 && cd ../../..

# Build g2o
cd Thirdparty/g2o && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j3 && cd ../../..

# Uncompress vocabulary
cd Vocabulary && tar xf ORBvoc.txt.tar.gz && cd ..

# Build ORB-SLAM3
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j3
```

This produces `ORB_SLAM3/lib/libORB_SLAM3.so`.

**Step 3 — Build the pybind11 Python wrapper:**

```bash
pip install pybind11

cd orbslam3_wrapper && mkdir -p build && cd build
cmake .. -Dpybind11_DIR=$(pybind11-config --cmakedir) -DCMAKE_BUILD_TYPE=Release
make -j3
```

This produces `orbslam3_python.cpython-*.so` in the project root, importable as `import orbslam3_python`.

**Step 4 — Verify:**

```bash
export LD_LIBRARY_PATH=$PWD/ORB_SLAM3/lib:/usr/local/lib:$LD_LIBRARY_PATH
python3 -c "import orbslam3_python; print('OK')"
```

### 5. C++ main build (Jetson container only)

Builds the full C++ pipeline (OAK-D + ORB-SLAM3 + DepthAI):

```bash
bash scripts/build.sh   # mkdir build && cmake .. && make
```

## Running Tests

```bash
PILOT_SYNTHETIC=1 python -m pytest tests/ -v --timeout=60
```

189 tests pass in the dev-local container; 7 hardware-dependent tests are auto-skipped.

**SLAM synthetic pipeline tests** (requires ORB-SLAM3 built — step 4 above):

```bash
export LD_LIBRARY_PATH=$PWD/ORB_SLAM3/lib:/usr/local/lib:$LD_LIBRARY_PATH
PILOT_SYNTHETIC=1 timeout 200 python3 -m pytest tests/test_slam_synthetic_pipeline.py -v -s --log-cli-level=INFO
```

Runs a full 2-lap closed-loop test: Phase A (mapping, 300 frames) → Phase B (localization, 300 frames with loop closure detection).

## Usage

**Synthetic / dev mode:**
```bash
python start.py
```

**Jetson (hardware):**
```bash
bash scripts/jetson_run.sh
```

## Scripts

Helper scripts live in `scripts/` — see [scripts/READMECPP.md](scripts/READMECPP.md) for C++ build details.

| Script | Purpose |
|--------|---------|
| `build.sh` | CMake + make C++ build |
| `jetson_run.sh` | Full Jetson deploy (power mode, git pull, build, launch) |
| `start.sh` | Git pull + validate config + launch |
| `run.sh` / `runCpp.sh` | Quick-launch Python / C++ entry points |
| `replay.sh` | Replay recorded data |
| `setup.sh` | Init git submodules |
| `branch_checkout.sh` | Switch all submodules to a branch |
| `dev_run.sh` | Dev launch with config auto-read |
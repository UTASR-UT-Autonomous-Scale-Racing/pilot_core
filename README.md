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
pip install PyYAML numpy opencv-python-headless scipy pytest pytest-timeout matplotlib pandas tqdm
```

### 4. C++ build (Jetson container only)

```bash
bash scripts/build.sh   # mkdir build && cmake .. && make
```

## Running Tests

```bash
PILOT_SYNTHETIC=1 python -m pytest tests/ -v --timeout=60
```

189 tests pass in the dev-local container; 7 hardware-dependent tests are auto-skipped.

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
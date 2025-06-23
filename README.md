# pilot_core

This repository serves as the central control hub for UTASR's autonomous scale racing project. It is responsible for coordinating the various components of the RC-car, including control logic, data ingestion, and vision processing.

## Overview

`pilot_core` coordinates and manages the main control logic for the RC-car. It acts as the central repository, integrating the following component repositories:

- [`pilot_controller`](https://github.com/UTASR-UT-Autonomous-Scale-Racing/pilot_controller)
- [`pilot_ingest`](https://github.com/UTASR-UT-Autonomous-Scale-Racing/pilot_ingest)
- [`pilot_vision`](https://github.com/UTASR-UT-Autonomous-Scale-Racing/pilot_vision)

## Setup

After cloning this repository, run the setup script:

```bash
chmod +x setup.sh
./setup.sh
```

This will clone the required repositories: pilot_ingest, pilot_vision, and pilot_control.

## Usage

To run the pilot core, execute the following command:

```bash
chmod +x jetson_run.sh
./jetson_run.sh
```
This script will pull the latest changes from our repositories and start the pilot core.

## More Information

For detailed documentation and usage instructions, please refer to the respective repositories listed above.